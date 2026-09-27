"""Tello I/O: the only process that talks to the Tello (plan 3.1, 3.3, 4.1 "Tello I/O" row, 6).

    python -m flyfollow.runtime.tello_io                  # dry run: connect, state + video, publish what WOULD be sent
    python -m flyfollow.runtime.tello_io --send           # really command the drone
    python -m flyfollow.runtime.tello_io --no-video --ip 192.168.10.1

In:  mode (current mode for RC_OWNER), rc, tello_cmd (takeoff | land | emergency | move | stop), kill.
Out: tello_state (every state packet, about 10 Hz), frame (+ FrameRing), rc_sent (20 Hz), tello_ack, tello_event.

Threads: main (bus, state, safety), sender (the single command queue AND the 20 Hz rc stream, so a blocking move
naturally pauses rc), video (PyAV decode into the FrameRing). Sticks go to djitellopy send_rc_control as final
integers (never FlyDrones' TelloDrone 60 % scaling). rc arbitration and safety rules: flyfollow.runtime.drone_common.

Dry run (default): connects, streams state and video, never sends takeoff, land, emergency, rc or moves; rc_sent
(dry_run True) shows what would go out and tello_ack answers "dry run". No cv2 import here: cv2 and av both bundle
FFmpeg dylibs and macOS warns about duplicate classes.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import logging
import socket
import sys
import threading
import time
import traceback

from flyfollow.interfaces import IMG_H, IMG_W
from flyfollow.runtime.drone_common import (
    HOVER,
    RcArbiter,
    SafetyLimits,
    move_timeout_s,
    parse_move,
    safety_reason,
    state_fields,
    takeoff_block_reason,
)
from flyfollow.runtime.messages import RC_HZ, msg
from flyfollow.runtime.node import Rate, log, stop_event

NAME = "tello_io"
TOPICS_IN = ["mode", "rc", "tello_cmd", "kill", "tello_state"]  # own tello_state = broker loopback heartbeat
CONTROL_PORT, STATE_PORT, VIDEO_PORT = 8889, 8890, 11111
VIDEO_URL = f"udp://@0.0.0.0:{VIDEO_PORT}"
TAKEOFF_TIMEOUT_S = 20
LAND_TIMEOUT_S = 7
STATE_LOST_S = 3.0  # no state packet this long while flying: try to land (the Tello also self-lands 15 s after rc stops)
VIDEO_OK_S = 1.0

FIREWALL_HINT = (
    "No state/video packets. Windows: allow Python through the firewall and open inbound UDP 8890 and 11111 "
    "(plan 3.3). macOS: allow incoming connections for python. Check that the laptop is on the TELLO-XXXXXX Wi-Fi."
)


def port_hint(port: int) -> str:
    return (f"UDP {port} is taken (macOS 'Errno 48 Address already in use' after a crash): find the stale process with "
            f"`lsof -nP -iUDP:{port}` and kill it (another tello_io, a lag test, or an R0 tool).")


def check_udp_port(port: int) -> str | None:
    """None if we can bind the port, else the OS error text."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("", port))
        return None
    except OSError as e:
        return str(e)
    finally:
        s.close()


def is_ok(resp) -> bool:
    return "ok" in str(resp).lower()


class TelloLink:
    """Thin wrapper over a djitellopy.Tello (or a fake with the same methods).

    Commands go out once (no djitellopy retries: a retried takeoff or move can execute twice), with stale
    responses cleared first so a late "ok" from an earlier timed-out command is not taken as this one's answer.
    """

    def __init__(self, tello):
        self.t = tello

    def connect(self) -> None:
        self.t.connect(wait_for_state=True)

    def state(self) -> dict:
        return self.t.get_current_state()

    def cmd(self, text: str, timeout_s: float) -> str:
        try:
            self.t.get_own_udp_object()["responses"].clear()
        except Exception:
            pass
        return str(self.t.send_command_with_return(text, timeout=max(1, int(round(timeout_s)))))

    def raw(self, text: str) -> None:
        self.t.send_command_without_return(text)

    def rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        self.t.send_rc_control(int(lr), int(fb), int(ud), int(yaw))


class VideoReader(threading.Thread):
    """PyAV decode of the Tello H.264 stream in a thread (plan 3.3: 150 to 300 ms vs about 1 s with OpenCV).

    Frames are discarded until warmup_s after opening and until the first key frame (or warmup_s + 2 s without one),
    then every decoded 960x720 RGB frame goes to on_frame(img, t_decoded). Reopens the stream after errors or stalls.
    """

    OPTIONS = {"fflags": "nobuffer", "flags": "low_delay", "fifo_size": "5000000", "overrun_nonfatal": "1",
               "probesize": "500000", "analyzeduration": "500000"}

    def __init__(self, on_frame, url: str = VIDEO_URL, warmup_s: float = 3.0, open_timeout_s: float = 5.0,
                 read_timeout_s: float = 2.0, fmt: str | None = "h264"):
        super().__init__(name="tello-video", daemon=True)
        self.on_frame = on_frame
        self.url, self.warmup_s, self.fmt = url, warmup_s, fmt
        self.timeouts = (open_timeout_s, read_timeout_s)
        self.stop_ev = threading.Event()
        self.n_decoded = self.n_discarded = self.n_bad_size = self.n_opens = 0
        self._hinted = False

    def stop(self) -> None:
        self.stop_ev.set()

    def run(self) -> None:
        import av  # only the video thread needs FFmpeg

        while not self.stop_ev.is_set():
            try:
                c = av.open(self.url, format=self.fmt, options=self.OPTIONS, timeout=self.timeouts)
            except Exception as e:
                log(NAME, f"video open failed: {e}")
                if not self._hinted:
                    log(NAME, FIREWALL_HINT)
                    log(NAME, port_hint(VIDEO_PORT))
                    self._hinted = True
                self.stop_ev.wait(1.0)
                continue
            self.n_opens += 1
            t_open = time.time()
            seen_key = False
            try:
                s = c.streams.video[0]
                try:
                    s.codec_context.thread_type = "SLICE"  # frame threading would add frames of latency
                except Exception:
                    pass
                for frame in c.decode(s):
                    if self.stop_ev.is_set():
                        break
                    self.n_decoded += 1
                    age = time.time() - t_open
                    if not seen_key:
                        if frame.key_frame or age > self.warmup_s + 2.0:
                            seen_key = True
                        else:
                            self.n_discarded += 1
                            continue
                    if age < self.warmup_s:
                        self.n_discarded += 1
                        continue
                    img = frame.to_ndarray(format="rgb24")
                    if img.shape != (IMG_H, IMG_W, 3):
                        self.n_bad_size += 1
                        continue
                    self.on_frame(img, time.time())
            except Exception as e:
                if not self.stop_ev.is_set():
                    log(NAME, f"video stream error ({type(e).__name__}: {e}); reopening")
            finally:
                try:
                    c.close()
                except Exception:
                    pass


class TelloIO:
    """Bus side of the Tello. step() runs in the main thread; the sender thread owns every packet to the drone."""

    def __init__(self, link: TelloLink, pub, sub, *, send: bool = False, video: bool = True,
                 limits: SafetyLimits | None = None, ring=None, rc_hz: float = RC_HZ):
        self.link, self.pub, self.sub = link, pub, sub
        self.send, self.video, self.ring = send, video, ring
        self.lim = limits or SafetyLimits()
        self.rc_hz = rc_hz
        self.arb = RcArbiter()
        self.flying = False
        self._arb_lock = threading.Lock()
        self._pub_lock = threading.Lock()
        self._jobs: collections.deque[dict] = collections.deque()
        self._jobs_lock = threading.Lock()
        self._busy: dict | None = None
        self._rc_hold = True  # no rc stream until a takeoff succeeds
        self._stop = threading.Event()
        self._sender: threading.Thread | None = None
        self._ids = itertools.count(1)
        self.t_start = time.time()
        self._last_raw: dict | None = None
        self._last_state_t = 0.0
        self._last_state_pub = 0.0
        self._bus_seen: float | None = None
        self._last_frame_t: float | None = None
        self._frame_id = 0
        self._last_cmd_t = time.time()
        self.safety_fired: str | None = None
        self.n_rc_sent = 0
        self.fatal: str | None = None

    # ------------------------------------------------------------------ publishing
    def _publish(self, m: dict) -> None:
        with self._pub_lock:
            self.pub.publish(m)

    def _event(self, kind: str, **fields) -> None:
        log(NAME, kind, fields)
        self._publish(msg("tello_event", kind=kind, **fields))

    def _ack(self, job: dict, ok: bool, detail: str, elapsed_s: float) -> None:
        self._publish(msg("tello_ack", id=job.get("id"), cmd=job.get("cmd"), ok=bool(ok), detail=str(detail),
                          elapsed_s=round(elapsed_s, 3), dry_run=not self.send))
        log(NAME, f"ack {job.get('cmd')} id={job.get('id')} ok={ok} {detail}")

    def _rc_sent(self, sticks, src: str, dry: bool) -> None:
        lr, fb, ud, yaw = sticks
        self._publish(msg("rc_sent", lr=lr, fb=fb, ud=ud, yaw=yaw, src=src, dry_run=dry, mode=self.arb.mode))

    # ------------------------------------------------------------------ video
    def on_frame(self, img, t_decoded: float) -> None:
        """Video thread: write into the ring and announce the slot."""
        self._frame_id += 1
        slot = self.ring.write(img, self._frame_id, t_decoded) if self.ring is not None else -1
        self._last_frame_t = t_decoded
        self._publish(msg("frame", frame_id=self._frame_id, t_decoded=t_decoded, slot=slot, w=IMG_W, h=IMG_H, src="tello"))

    def video_age(self, now: float) -> float:
        """Seconds since the last decoded frame; 999 before the first one (so no takeoff without video)."""
        return min(999.0, now - self._last_frame_t) if self._last_frame_t is not None else 999.0

    # ------------------------------------------------------------------ bus input (main thread)
    def step(self, now: float | None = None) -> None:
        for m in self.sub.drain():
            self.on_msg(m, time.time() if now is None else now)
        now = time.time() if now is None else now
        raw = self.link.state()
        if raw and raw is not self._last_raw:
            self._last_raw = raw
            self._last_state_t = now
            self._publish_state(now)
        elif now - self._last_state_pub > 0.5:
            self._publish_state(now)
        self.check_safety(now)
        if self._sender is not None and not self._sender.is_alive() and not self._stop.is_set():
            self.fatal = "sender thread died"

    def on_msg(self, m: dict, now: float) -> None:
        tp = m.get("topic")
        if m.get("replayed") and tp != "tello_state":
            return  # never act on a replayed command (messages.py)
        if tp == "rc":
            with self._arb_lock:
                if self.arb.offer(m, now):
                    self._last_cmd_t = now
        elif tp == "mode":
            with self._arb_lock:
                if self.arb.set_mode(str(m.get("to"))):
                    self._last_cmd_t = now
        elif tp == "kill":
            self.kill(str(m.get("action", "land")), f"kill from {m.get('src_node', '?')}")
        elif tp == "tello_cmd":
            self._last_cmd_t = now
            self.submit(m)
        elif tp == "tello_state" and m.get("src_node") == getattr(self.pub, "name", NAME):
            self._bus_seen = now

    def submit(self, m: dict) -> None:
        """Queue a tello_cmd. land and emergency preempt like kill; stop cancels queued commands first."""
        cmd = str(m.get("cmd", ""))
        job = {"id": m.get("id", f"auto-{next(self._ids)}"), "cmd": cmd, "args": m.get("args") or {}}
        if cmd in ("land", "emergency"):
            self.kill(cmd, f"tello_cmd from {m.get('src_node', '?')}", job)
            return
        dropped = []
        with self._jobs_lock:
            if cmd == "stop":
                dropped = list(self._jobs)
                self._jobs.clear()
            self._jobs.append(job)
        for j in dropped:
            self._ack(j, False, "cancelled by stop", 0.0)

    def kill(self, action: str, reason: str, job: dict | None = None) -> None:
        """Land or emergency now, in every mode: drops everything queued and stops the rc stream."""
        action = "emergency" if action == "emergency" else "land"
        job = dict(job or {"id": f"kill-{next(self._ids)}", "args": {}})
        job["cmd"] = action
        with self._jobs_lock:
            dropped = list(self._jobs)
            self._jobs.clear()
            self._jobs.appendleft(job)
            self._rc_hold = True
        for j in dropped:
            self._ack(j, False, f"preempted by {action} ({reason})", 0.0)
        if self.send and (action == "emergency" or self._busy is not None):
            try:  # a blocking command is in flight in the sender thread: do not wait for it
                self.link.raw(action)
            except Exception as e:
                log(NAME, f"raw {action} failed: {e}")
        self._event("kill", action=action, reason=reason, busy=(self._busy or {}).get("cmd"))

    # ------------------------------------------------------------------ state and safety
    def _state_raw(self) -> dict:
        return self._last_raw or {}

    def _publish_state(self, now: float) -> None:
        raw = self._state_raw()
        va = self.video_age(now)
        f = state_fields(raw)
        self._publish(msg("tello_state", **f, flying=self.flying, video_ok=self.video and va < VIDEO_OK_S,
                          video_age_s=round(va, 3), sending=self.send, state_ok=bool(raw) and now - self._last_state_t < 1.0,
                          state_age_s=round(now - self._last_state_t, 3) if raw else None, mode=self.arb.mode,
                          busy=(self._busy or {}).get("cmd"), queued=len(self._jobs), safety=self.safety_fired))
        self._last_state_pub = now

    def bus_age(self, now: float) -> float | None:
        if self._bus_seen is not None:
            return now - self._bus_seen
        return now - self.t_start if now - self.t_start > 5.0 else None

    def check_safety(self, now: float) -> str | None:
        if not (self.flying and self.send) or self.safety_fired:
            return None
        raw = self._state_raw()
        reason = safety_reason(self.lim, flying=True, bat_pct=raw.get("bat"), temph_c=raw.get("temph"),
                               video_enabled=self.video, video_age_s=self.video_age(now), bus_age_s=self.bus_age(now),
                               idle_s=now - self._last_cmd_t)
        if reason is None and raw and now - self._last_state_t > STATE_LOST_S:
            reason = f"no state packet for {now - self._last_state_t:.1f} s"
        if reason:
            self.safety_fired = reason
            self.kill("land", f"safety: {reason}")
        return reason

    # ------------------------------------------------------------------ sender thread
    def start(self) -> TelloIO:
        self._sender = threading.Thread(target=self._sender_loop, name="tello-sender", daemon=True)
        self._sender.start()
        return self

    def _sender_loop(self) -> None:
        rate = Rate(self.rc_hz)
        while not self._stop.is_set():
            try:
                self.sender_step()
            except Exception:
                log(NAME, "sender error:\n" + traceback.format_exc())
                self.fatal = "sender error"
                return
            rate.sleep()

    def sender_step(self) -> None:
        """One 20 Hz tick: run the next queued command (blocking), else send the arbitrated rc."""
        with self._jobs_lock:
            job = self._jobs.popleft() if self._jobs else None
        if job is not None:
            self._busy = job
            t0 = time.time()
            try:
                ok, detail = self._run(job)
            except Exception as e:
                ok, detail = False, f"{type(e).__name__}: {e}"
            finally:
                self._busy = None
            self._ack(job, ok, detail, time.time() - t0)
            return
        with self._arb_lock:
            sticks, src = self.arb.select(time.time())
        if not self.send:
            self._rc_sent(sticks, src, dry=True)
        elif self.flying and not self._rc_hold:
            self.link.rc(*sticks)
            self.n_rc_sent += 1
            self._rc_sent(sticks, src, dry=False)

    def _run(self, job: dict) -> tuple[bool, str]:
        cmd = job["cmd"]
        if cmd == "takeoff":
            return self._takeoff()
        if cmd == "land":
            return self._land()
        if cmd == "emergency":
            self._rc_hold = True
            if self.send:
                self.link.raw("emergency")
            self.flying = False
            return True, "emergency sent" if self.send else "dry run: emergency not sent"
        if cmd == "move":
            return self._move(job.get("args") or {})
        if cmd == "stop":
            with self._arb_lock:
                self.arb.clear()
            if self.send and self.flying and not self._rc_hold:
                self.link.rc(*HOVER)
                self._rc_sent(HOVER, "stop", dry=False)
            return True, "queue cleared, hovering"
        return False, f"unknown cmd {cmd!r}"

    def _takeoff(self) -> tuple[bool, str]:
        if self.flying:
            return True, "already flying"
        raw = self._state_raw()
        why = takeoff_block_reason(self.lim, bat_pct=raw.get("bat"), temph_c=raw.get("temph"), video_enabled=self.video,
                                   video_ok=self.video_age(time.time()) < VIDEO_OK_S)
        if why:
            return False, f"refused: {why}"
        if not self.send:
            # pretend to fly, so the mission enters FOLLOW and the controller computes sticks (shown as rc_sent
            # dry_run); nothing is ever sent: every packet to the drone is gated on self.send
            with self._arb_lock:
                self.arb.clear()
            self.flying = True
            self._rc_hold = False
            return True, "dry run: takeoff not sent (pretending to fly)"
        resp = self.link.cmd("takeoff", TAKEOFF_TIMEOUT_S)
        ok = is_ok(resp)
        h = self._state_raw().get("h", 0)
        if not ok and isinstance(h, (int, float)) and h >= 30:
            ok, resp = True, f"{resp} (but h = {h} cm: flying)"
        if ok:
            with self._arb_lock:
                self.arb.clear()
            self.safety_fired = None
            self.flying = True
            self._last_cmd_t = time.time()
            self._rc_hold = False
        return ok, resp

    def _land(self) -> tuple[bool, str]:
        self._rc_hold = True
        if not self.send:
            self.flying = False
            return True, "dry run: land not sent"
        h = self._state_raw().get("h", 0)
        if not self.flying and not (isinstance(h, (int, float)) and h >= 20):
            return True, "not flying"
        resp = ""
        for _ in range(3):
            resp = self.link.cmd("land", LAND_TIMEOUT_S)
            if is_ok(resp):
                break
            h = self._state_raw().get("h", 0)
            if isinstance(h, (int, float)) and h <= 0:
                break
        self.flying = False
        if is_ok(resp):
            return True, resp
        return False, f"land not confirmed ({resp}); rc stopped, so the Tello auto-lands within 15 s"

    def _move(self, args: dict) -> tuple[bool, str]:
        sdk, err = parse_move(args)
        if sdk is None:
            return False, err
        if not self.send:
            return True, f"dry run: would send '{sdk}'"
        if not self.flying or self._rc_hold:
            return False, "not flying"
        # The rc stream is paused for the whole move: this thread is the only one that sends.
        resp = self.link.cmd(sdk, move_timeout_s(sdk, args))
        if self.flying and not self._rc_hold:
            self.link.rc(*HOVER)
            self._rc_sent(HOVER, "move_end", dry=False)
        with self._arb_lock:
            self.arb.clear()
        return is_ok(resp), resp

    # ------------------------------------------------------------------ shutdown
    def shutdown(self) -> None:
        """Stop the sender; if flying, land directly (process exit, signal or exception)."""
        self._stop.set()
        busy = self._busy is not None
        if self._sender is not None:
            self._sender.join(timeout=0.5 if busy else 2.0)
        if self.send and self.flying:
            log(NAME, "exit while flying: landing")
            try:
                if busy:
                    self.link.raw("land")
                self._land()
            except Exception as e:
                log(NAME, f"land on exit failed: {e}; the Tello auto-lands 15 s after the last rc")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Tello I/O: the only process that talks to the Tello (dry run unless --send)")
    ap.add_argument("--send", action="store_true", help="really send takeoff, rc, moves and land (default: dry run)")
    ap.add_argument("--no-video", action="store_true", help="do not start the video stream")
    ap.add_argument("--ip", default="192.168.10.1")
    ap.add_argument("--min-battery", type=float, default=SafetyLimits.min_battery_pct)
    ap.add_argument("--max-temp", type=float, default=SafetyLimits.max_temp_c)
    ap.add_argument("--video-lost-s", type=float, default=SafetyLimits.video_lost_s)
    ap.add_argument("--bus-lost-s", type=float, default=SafetyLimits.bus_lost_s)
    ap.add_argument("--idle-land-s", type=float, default=0.0, help="land after this long with no accepted rc/command (0 = off)")
    ap.add_argument("--video-warmup-s", type=float, default=3.0)
    a = ap.parse_args(argv)
    logging.getLogger("djitellopy").setLevel(logging.WARNING)
    video = not a.no_video
    stop = stop_event()
    log(NAME, "SENDING: takeoff, rc and moves go to the drone" if a.send else "DRY RUN: nothing that moves the drone is sent")
    for port in [CONTROL_PORT, STATE_PORT] + ([VIDEO_PORT] if video else []):
        err = check_udp_port(port)
        if err:
            log(NAME, f"cannot bind UDP {port}: {err}")
            log(NAME, port_hint(port))
            sys.exit(2)

    from djitellopy import Tello

    from flyfollow.runtime.bus import FrameRing, Publisher, Subscriber

    pub = Publisher(NAME)
    sub = Subscriber(TOPICS_IN)
    tello = Tello(host=a.ip)
    link = TelloLink(tello)
    try:
        link.connect()
    except Exception as e:
        log(NAME, f"connect failed: {e}")
        log(NAME, FIREWALL_HINT)
        log(NAME, port_hint(STATE_PORT))
        sys.exit(2)
    raw = link.state()
    log(NAME, f"connected: battery {raw.get('bat')}%, temph {raw.get('temph')} C, h {raw.get('h')} cm")
    lim = SafetyLimits(a.min_battery, a.max_temp, a.video_lost_s, a.bus_lost_s, a.idle_land_s)
    ring = FrameRing.create(h=IMG_H, w=IMG_W) if video else None
    io = TelloIO(link, pub, sub, send=a.send, video=video, limits=lim, ring=ring)
    reader = None
    if video:
        resp = link.cmd("streamon", 7)
        log(NAME, f"streamon: {resp}")
        reader = VideoReader(io.on_frame, warmup_s=a.video_warmup_s)
        reader.start()
    io.start()
    t_log = time.time()
    try:
        while not stop.is_set() and io.fatal is None:
            io.step()
            now = time.time()
            if now - t_log > 5.0:
                t_log = now
                r = io._state_raw()
                log(NAME, f"mode {io.arb.mode} flying {io.flying} bat {r.get('bat')} temph {r.get('temph')} "
                          f"video_age {io.video_age(now):.2f}s frames {io._frame_id} rc_rejected {dict(io.arb.rejected)}")
            time.sleep(0.005)
        if io.fatal:
            log(NAME, f"fatal: {io.fatal}")
    except BaseException:
        log(NAME, "exception:\n" + traceback.format_exc())
        raise
    finally:
        io.shutdown()
        if reader is not None:
            reader.stop()
            try:
                link.cmd("streamoff", 2)
            except Exception:
                pass
        if ring is not None:
            ring.close()
            ring.unlink()
        pub.close()
        sub.close()
        log(NAME, "stopped")


if __name__ == "__main__":
    main()
