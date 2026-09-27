"""One command for the fly-controller test harness: broker, recorder, backend, controller, mission_lite, viz, operator.

    python -m flyfollow.runtime.launch --sim                      # simulated Tello + world, synthetic det
    python -m flyfollow.runtime.launch --dry                      # real Tello video/state + YOLO, never sends commands
    python -m flyfollow.runtime.launch --send                     # real flight (asks you to type "fly")
    python -m flyfollow.runtime.launch --replay recordings/<s>    # recorded session instead of the drone
    options: --detector yolo|external|synthetic  --viz  --no-operator  --script FILE  --session NAME  --port 5560
             --duration S  --no-video  --args MODULE="--flag value"  --plain

Children run as `python -m flyfollow.runtime.<module>` in their own process groups, so a terminal Ctrl+C reaches
only the launcher (or the operator console, which treats it as land + quit). Shutdown (Ctrl+C, operator quit,
--duration, or any child crash in --send): publish kill land, wait for the land ack or tello_state flying=False
(re-send once after 1.5 s, give up after 5 s), then stop children in reverse order (recorder last). If the
Tello I/O itself died in --send, land is sent straight to the drone over UDP. Child logs:
recordings/<session>/logs/<name>.log. mission_lite owns the modes (FOLLOW / HOLD / GUIDE / LAND / IDLE); an
optional flyfollow.runtime.guidance is started only if present.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from flyfollow.interfaces import REPO_ROOT, brains_dir
from flyfollow.runtime.bus import Broker, Publisher, Subscriber, default_frame_ring, default_pub_addr, shm_name
from flyfollow.runtime.messages import RC_OWNER, msg
from flyfollow.runtime.recorder import default_session, recordings_root

NAME = "launch"
LAND_WAIT_S = 5.0  # max wait for land confirmation at shutdown
LAND_WAIT_MAX_S = 8.0  # extended while the backend confirmed the kill and still reports flying (descending)
LAND_RESEND_S = 1.5  # re-send kill land once if nothing confirms by then
ACK_GRACE_S = 1.0  # after flying=False, keep the backend up this long for its land ack
STOP_WAIT_S = 3.0
TELLO_IP, TELLO_CMD_PORT = "192.168.10.1", 8889
OWNERS = {"tello_io": "R2", "sim_world": "R2", "controller_runner": "R4", "guidance": "R5"}
IS_WIN = os.name == "nt"


@dataclass
class Child:
    name: str
    module: str
    args: list[str]
    required: bool = False
    foreground: bool = False  # the interactive operator: keeps the terminal
    proc: subprocess.Popen | None = None
    log_path: Path | None = None
    expected_exit: bool = False
    reported: bool = False
    tail: list[str] = field(default_factory=list)


def module_exists(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def udp_port_free(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def who_holds(port: int, proto: str = "UDP") -> str:
    if IS_WIN:
        return f"netstat -ano | findstr :{port}"
    try:
        out = subprocess.run(["lsof", "-nP", f"-i{proto}:{port}"], capture_output=True, text=True, timeout=3).stdout
        return out.strip() or "(lsof shows nothing; try sudo lsof)"
    except Exception:
        return f"lsof -nP -i{proto}:{port}"


def direct_udp_land() -> None:
    """Last resort when the Tello I/O is dead: raw SDK "command" then "land" to the drone."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for cmd in (b"command", b"land", b"land"):
            s.sendto(cmd, (TELLO_IP, TELLO_CMD_PORT))
            time.sleep(0.15)
    except OSError:
        pass
    finally:
        s.close()


class Launcher:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.mode = "sim" if a.sim else "dry" if a.dry else "send" if a.send else "replay"
        self.session = a.session or default_session(self.mode)
        self.root = Path(a.record_root) if a.record_root else recordings_root()
        self.dir = self.root / self.session
        self.logs = self.dir / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.interactive = not (a.no_operator or a.script)
        self.children: list[Child] = []
        self.stop = threading.Event()
        self.stop_reason = ""
        self.exit_code = 0
        self._log = (self.logs / "launch.log").open("a")
        self._print_ok = True  # False while the curses operator owns the terminal
        self.env = dict(os.environ, PYTHONUNBUFFERED="1", FLYFOLLOW_SESSION=self.session,
                        FLYFOLLOW_SESSION_DIR=str(self.dir), FLYFOLLOW_MODE=self.mode.upper())
        if a.port:
            self.env["FLYFOLLOW_PUB_ADDR"] = f"tcp://127.0.0.1:{a.port}"
            self.env["FLYFOLLOW_SUB_ADDR"] = f"tcp://127.0.0.1:{a.port + 1}"
            os.environ.update({k: self.env[k] for k in ("FLYFOLLOW_PUB_ADDR", "FLYFOLLOW_SUB_ADDR")})
        ring = a.ring or (f"ff_frames_{a.port}" if a.port else None)  # parallel sessions must not share a ring
        if ring:
            self.env["FLYFOLLOW_FRAME_RING"] = os.environ["FLYFOLLOW_FRAME_RING"] = ring
        self.detector = a.detector or ("yolo" if self.mode in ("dry", "send") else "synthetic")
        self.flying = None
        self.mode_now = "IDLE"
        self.stale: set[str] = set()
        self.last_seen: dict[str, float | None] = {"tello_state": None, "det": None, "rc": None}
        self.t_start = time.time()

    # ---------------------------------------------------------------------------------------------- output
    def say(self, text: str, level: str = "info", key: str | None = None, ok: bool | None = None) -> None:
        line = f"{time.strftime('%H:%M:%S')} [launch] {text}"
        self._log.write(line + "\n")
        self._log.flush()
        if self._print_ok:
            print(line, flush=True)
        if level != "info" and getattr(self, "pub", None) is not None:
            self.pub.publish(msg("health", key=key or text, ok=bool(ok), level=level, text=text))

    # ---------------------------------------------------------------------------------------------- plan
    def plan(self) -> list[Child]:
        a = self.a
        extra = {}
        for spec in a.args or []:
            mod, _, rest = spec.partition("=")
            extra.setdefault(mod.strip(), []).extend(rest.split())
        x = lambda n: extra.get(n, [])  # noqa: E731
        kids = [Child("recorder", "flyfollow.runtime.recorder",
                      ["--session", self.session, "--root", str(self.root)]
                      + (["--video"] if self.mode in ("dry", "send") and not a.no_video else []) + x("recorder"))]
        if self.mode == "sim":
            sargs = ["--scenario", a.scenario, "--seed", str(a.seed)]
            if self.detector in ("external", "yolo"):  # render frames for the perception process, no ground-truth det
                sargs += ["--det", "none", "--frames"]
            kids.append(Child("sim_world", "flyfollow.runtime.sim_world", sargs + x("sim_world"), required=True))
        elif self.mode in ("dry", "send"):
            kids.append(Child("tello_io", "flyfollow.runtime.tello_io", (["--send"] if self.mode == "send" else [])
                              + x("tello_io"), required=True))
        else:
            src = Path(a.replay)
            topics = ["tello_state"] + (["det"] if self.detector == "synthetic" else [])
            rargs = [str(src), "--no-broker", "--speed", str(a.replay_speed), "--topics", ",".join(topics)]
            if (src / "video.mp4").exists():
                rargs.append("--frames")
            kids.append(Child("replay", "flyfollow.runtime.replay", rargs + x("replay"), required=True))
        if self.detector == "yolo":  # built-in person detector: FrameRing -> det
            kids.append(Child("detector", "flyfollow.runtime.detector", x("detector"), required=True))
        if not a.no_controller:
            kids.append(Child("controller_runner", "flyfollow.runtime.controller_runner",
                              (["--viz"] if a.viz else []) + x("controller_runner")))
        if not a.no_mission:  # the harness's mode owner (the ReachGlass stack owns the real mission)
            kids.append(Child("mission_lite", "flyfollow.runtime.mission_lite",
                              (["--takeoff-follow"] if a.takeoff_follow else []) + x("mission_lite")))
        if not a.no_guidance and module_exists("flyfollow.runtime.guidance"):  # optional, started only if present
            kids.append(Child("guidance", "flyfollow.runtime.guidance", x("guidance")))
        if a.viz:
            brain = Path(a.viz_brain) if a.viz_brain else brains_dir() / "pursuit_core1.npz"
            kids.append(Child("viz", "flyfollow.viz.live", (["--brain", str(brain)] if brain.exists() else []) + x("viz")))
        if not a.no_operator or a.script:
            oargs = (["--script", a.script, "--linger", str(a.linger)] if a.script else []) + (["--plain"] if a.plain else [])
            kids.append(Child("operator", "flyfollow.runtime.operator", oargs + x("operator"), foreground=self.interactive))
        return kids

    # ---------------------------------------------------------------------------------------------- processes
    def spawn(self, c: Child) -> None:
        c.log_path = self.logs / f"{c.name}.log"
        cmd = [sys.executable, "-m", c.module, *c.args]
        self.say(f"start {c.name}: {' '.join(cmd[1:])}")
        kw: dict = dict(cwd=str(REPO_ROOT), env=self.env)
        if c.foreground:  # owns the terminal; its stderr still goes to the log
            kw.update(stderr=c.log_path.open("a"))
        else:
            kw.update(stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True)
            if IS_WIN:
                kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kw["start_new_session"] = True  # terminal Ctrl+C does not reach it
        c.proc = subprocess.Popen(cmd, **kw)
        if not c.foreground:
            threading.Thread(target=self._pump, args=(c,), daemon=True, name=f"log-{c.name}").start()

    def _pump(self, c: Child) -> None:
        with c.log_path.open("a") as f:
            for line in c.proc.stdout:
                f.write(line)
                f.flush()
                c.tail = (c.tail + [line.rstrip()])[-15:]
                if self._print_ok and not self.a.quiet:
                    print(f"[{c.name}] {line.rstrip()}", flush=True)

    def signal_child(self, c: Child, how: str) -> None:
        p = c.proc
        if p is None or p.poll() is not None:
            return
        try:
            if how == "kill":
                p.kill()
            elif how == "term":
                p.terminate()
            elif IS_WIN:
                p.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                p.send_signal(signal.SIGINT)
        except OSError:
            pass

    def stop_children(self, kids: list[Child], wait_s: float) -> None:
        for c in kids:
            c.expected_exit = True
            self.signal_child(c, "int")
        for how, extra in (("term", wait_s), ("kill", 1.0)):
            deadline = time.monotonic() + (wait_s if how == "term" else extra)
            while time.monotonic() < deadline and any(c.proc and c.proc.poll() is None for c in kids):
                time.sleep(0.05)
            for c in kids:
                if c.proc and c.proc.poll() is None:
                    self.say(f"{c.name} did not stop, {how}")
                    self.signal_child(c, how)
        for c in kids:
            if c.proc:
                try:
                    c.proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass

    # ---------------------------------------------------------------------------------------------- health
    def watch_bus(self, m: dict) -> None:
        tp = m.get("topic")
        if tp == "tello_state":
            self.flying = m.get("flying")
        elif tp == "mode":
            self.mode_now = m.get("to") or self.mode_now
        if tp in self.last_seen:
            self.last_seen[tp] = time.time()

    def check_liveness(self) -> None:
        now = time.time()
        limits = {"tello_state": 1.0, "det": 1.5}
        if RC_OWNER.get(self.mode_now) and self.mode != "replay":
            limits["rc"] = 1.0
        for tp in self.last_seen:
            lim = limits.get(tp)
            t = self.last_seen[tp]
            age = None if t is None else now - t
            bad = lim is not None and ((age is None and now - self.t_start > 10.0) or (age is not None and age > lim))
            if bad and tp not in self.stale:
                self.stale.add(tp)
                what = f"no {tp} since start" if age is None else f"{tp} stale {age:.1f}s"
                hint = " (external detector running?)" if tp == "det" and self.detector == "external" else ""
                self.say(f"WARNING {what}{hint}", level="warn", key=f"stale:{tp}", ok=False)
            elif not bad and tp in self.stale:
                self.stale.discard(tp)
                self.say(f"{tp} back", level="ok", key=f"stale:{tp}", ok=True)

    def check_children(self) -> None:
        for c in self.children:
            if c.proc is None or c.reported or c.proc.poll() is None:
                continue
            c.reported = True
            code = c.proc.returncode
            if c.name == "operator":
                self.exit_code = code or 0
                self.request_stop(f"operator exited ({code})")
            elif c.name == "replay" and code == 0 and not self.interactive:
                self.request_stop("replay finished")
            elif c.name == "replay" and code == 0:
                self.say("replay finished (system stays up; quit the operator to stop)", level="warn", key="replay")
            else:
                tail = " | ".join(c.tail[-3:])
                self.say(f"ERROR {c.name} exited with code {code}; log {c.log_path}  {tail}", level="error",
                         key=f"crash:{c.name}", ok=False)
                if self.mode == "send" and c.name != "viz":  # the viewer is display only
                    self.request_stop(f"{c.name} crashed in --send")

    def request_stop(self, reason: str) -> None:
        if not self.stop.is_set():
            self.stop_reason = reason
            self.stop.set()

    # ---------------------------------------------------------------------------------------------- land
    def land_and_wait(self) -> str:
        """Publish kill land and wait (at most LAND_WAIT_S) until the landing is confirmed:
        - landed: a tello_state after the kill says flying=False; the backend is then kept alive up to ACK_GRACE_S
          more so its land ack (sent after touchdown) is still published and recorded;
        - or a land ack (ok) with no later tello_state saying flying=True for ACK_GRACE_S (a backend without state).
        An ack while states still say flying ("already landing") is NOT confirmation. The backend's tello_event
        kill (sent at once) means the kill arrived and the landing is in progress; if neither that, an ack nor
        flying=False arrives within LAND_RESEND_S, the kill is re-sent once. While the kill is confirmed and the
        drone still reports flying at LAND_WAIT_S, the wait extends to LAND_WAIT_MAX_S (a slow descent).
        Returns "ack", "state" or "none"."""
        if self.mode == "replay":
            return "replay"
        backend = next((c for c in self.children if c.name in ("tello_io", "sim_world")), None)
        alive = lambda: backend is not None and backend.proc is not None and backend.proc.poll() is None  # noqa: E731
        if not alive():
            if self.mode == "send":
                self.say("land: Tello I/O is not running, sending land directly over UDP", level="error", key="land")
                direct_udp_land()
                return "udp"
            self.say("land: no drone backend running, nothing to land")
            return "no-backend"
        t_kill = time.time()
        self.pub.publish(msg("kill", action="land", source="launch"))
        self.say(f"land: kill land sent, waiting for tello_state flying=False and the land ack (max {LAND_WAIT_S:g} s)")
        ack: dict | None = None
        t_ack = t_landed = t_event = None
        flying: bool | None = None  # newest tello_state after the kill
        resent = False
        path = "none"
        extended = False
        while (now := time.time()) - t_kill < (LAND_WAIT_MAX_S if extended else LAND_WAIT_S):
            if not extended and t_event is not None and flying is True and now - t_kill >= LAND_WAIT_S - 0.05:
                extended = True
                self.say(f"land: still descending at {LAND_WAIT_S:g} s, waiting up to {LAND_WAIT_MAX_S:g} s")
            for m in self.sub.drain(1000):
                self.watch_bus(m)
                tp, t = m.get("topic"), m.get("t", 0)
                if tp == "tello_ack" and ack is None and m.get("cmd") in ("land", "emergency") and m.get("ok") \
                        and t >= t_kill - 0.05:
                    ack, t_ack = m, now
                elif tp == "tello_state" and t > t_kill:
                    flying = m.get("flying")
                    if flying is False and t_landed is None:
                        t_landed = now
                elif tp == "tello_event" and m.get("kind") == "kill" and t_event is None and t >= t_kill - 0.05:
                    t_event = now
                    self.say(f"land: kill received by the backend at {now - t_kill:.2f} s, landing")
            if t_landed is not None and (ack is not None or now - t_landed >= ACK_GRACE_S):
                path = "ack" if ack is not None else "state"
                break
            if ack is not None and flying is not True and now - t_ack >= ACK_GRACE_S:
                path = "ack"
                break
            if not resent and ack is None and t_landed is None and t_event is None and now - t_kill >= LAND_RESEND_S:
                resent = True
                self.pub.publish(msg("kill", action="land", source="launch", resend=True))
                self.say(f"land: no response to the kill after {LAND_RESEND_S:g} s, kill land RE-SENT")
            if not alive():
                self.say("land: drone backend exited while waiting")
                break
            time.sleep(0.02)
        again = " (after re-send)" if resent else ""
        rel = lambda x: "-" if x is None else f"{x - t_kill:.2f} s"  # noqa: E731
        ack_txt = f"land ack {ack.get('id')} {ack.get('detail')!r} at {rel(t_ack)}" if ack else "no land ack"
        if path == "ack" and t_landed is not None:
            self.say(f"land: CONFIRMED by ack{again}: flying=False at {rel(t_landed)}, {ack_txt}")
        elif path == "ack":
            self.say(f"land: CONFIRMED by ack{again} ({ack_txt}; no tello_state saying flying after it)")
        elif path == "state":
            self.say(f"land: CONFIRMED by tello_state flying=False{again} at {rel(t_landed)} ({ack_txt} "
                     f"within {ACK_GRACE_S:g} s after)")
        else:
            self.say(f"land: NOT CONFIRMED after {time.time() - t_kill:.1f} s{again}: newest flying={flying}, {ack_txt}, "
                     f"kill received {rel(t_event)}", level="error", key="land")
        return path

    # ---------------------------------------------------------------------------------------------- run
    def preflight(self) -> bool:
        a = self.a
        if self.mode == "replay" and not (Path(a.replay) / "bus.jsonl").exists():
            print(f"ERROR: {a.replay} has no bus.jsonl")
            return False
        if a.takeoff_follow and self.mode != "sim":
            print("ERROR: --takeoff-follow is for --sim only (take off with the operator console on the real drone)")
            return False
        if self.detector == "synthetic" and self.mode in ("dry", "send"):
            print("ERROR: --detector synthetic needs --sim or --replay (no ground truth on the real drone)")
            return False
        if self.detector == "yolo" and not self.yolo_ready():
            return False
        if self.mode in ("dry", "send"):
            for port in (8889, 8890, 11111):
                if not udp_port_free(port):
                    print(f"ERROR: UDP port {port} is in use (a crashed run?). Holders:\n{who_holds(port)}\n"
                          "Kill that process (kill <PID>) and retry.")
                    return False
        if self.mode == "send":
            print("\n" + "!" * 78 + "\n!!  --send: REAL FLIGHT. Commands WILL be sent to the Tello.\n"
                  "!!  Props clear, spotter ready, operator hand on SPACE (emergency) and L (land).\n" + "!" * 78)
            if not sys.stdin.isatty():
                print("ERROR: --send needs an interactive terminal to confirm")
                return False
            try:
                answer = input('Type "fly" to continue: ').strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer != "fly":
                print("aborted")
                return False
        return True

    def yolo_ready(self) -> bool:
        """The built-in detector needs ultralytics and, on the Tello Wi-Fi (no internet), its weights on disk."""
        if not module_exists("ultralytics"):
            print('ERROR: --detector yolo needs ultralytics: pip install -e ".[perception]" (or --detector external)')
            return False
        if self.mode == "replay" and not (Path(self.a.replay) / "video.mp4").exists():
            print(f"ERROR: --detector yolo needs frames: {self.a.replay} has no video.mp4")
            return False
        from flyfollow.runtime.detector import DEFAULT_WEIGHTS, resolve_weights

        args = [w for spec in self.a.args or [] if spec.partition("=")[0].strip() == "detector"
                for w in spec.partition("=")[2].split()]
        name = args[args.index("--weights") + 1] if "--weights" in args[:-1] else DEFAULT_WEIGHTS
        w = resolve_weights(name)
        if not w.exists() and self.mode in ("dry", "send"):
            print(f"ERROR: detector weights {w} missing and the Tello Wi-Fi has no internet. Run once with internet:\n"
                  "  python -m flyfollow.runtime.detector --selftest")
            return False
        return True

    def run(self) -> int:
        if not self.preflight():
            return 2
        try:
            self.broker = Broker().start()
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return 2
        self.pub = Publisher(NAME)
        self.sub = Subscriber(["tello_state", "det", "rc", "tello_ack", "tello_event", "mode", "health"])
        self.say(f"session {self.session} ({self.mode}, detector {self.detector}); recording to {self.dir}")
        self.say(f"bus {self.broker.pub_addr} -> {self.broker.sub_addr}")
        if self.detector == "yolo" and self.mode == "sim":
            self.say("note: YOLO on the simulator's box drawings finds few people; use --detector synthetic to test the loop")
        if self.detector == "external" and self.mode != "replay":
            self.say(f"external detector: FrameRing.attach() (ring {shm_name(default_frame_ring())!r}) for frames, "
                     f"publish det to {default_pub_addr()} (flyfollow.runtime.bus.Publisher)")
        prev = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}

        def on_signal(signum, frame):
            if self.stop.is_set():
                self.say("second signal: stopping children now")
                for c in self.children:
                    self.signal_child(c, "kill")
                os._exit(130)
            self.request_stop("signal")

        for s in prev:
            signal.signal(s, on_signal)
        try:
            self._start_all()
            self.t_start = time.time()
            while not self.stop.wait(0.2):
                for m in self.sub.drain(2000):
                    self.watch_bus(m)
                self.check_children()
                self.check_liveness()
                if self.a.duration and time.time() - self.t_start > self.a.duration:
                    self.request_stop(f"--duration {self.a.duration} s")
        finally:
            self._shutdown()
            for s, h in prev.items():
                signal.signal(s, h)
        return self.exit_code

    def _start_all(self) -> None:
        self.t_start = time.time()
        for c in self.plan():
            if not module_exists(c.module):
                owner = OWNERS.get(c.name, "")
                self.say(f"ERROR module {c.module} is missing{f' ({owner})' if owner else ''}; "
                         + ("cannot run without it" if c.required else "continuing without it"), level="error",
                         key=f"missing:{c.name}", ok=False)
                if c.required:
                    self.exit_code = 2
                    self.request_stop(f"{c.name} missing")
                    return
                continue
            if c.foreground:
                self._print_ok = False  # curses takes the terminal from here on
            self.spawn(c)
            self.children.append(c)
            if c.name == "recorder":
                self._wait_recorder()

    def _wait_recorder(self, timeout_s: float = 5.0) -> None:
        """Start the rest only once the recorder is subscribed (it says so on the health topic)."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            m = self.sub.recv(0.1)
            if m and m.get("topic") == "health" and m.get("key") == "recorder":
                return
        self.say("recorder did not report ready within 5 s; continuing", level="warn", key="recorder", ok=False)

    def _shutdown(self) -> None:
        self._print_ok = True
        self.say(f"stopping: {self.stop_reason or 'exit'}")
        try:
            self.land_and_wait()
        except Exception as e:  # never skip stopping children
            self.say(f"land step failed: {e}")
        order = ["operator", "viz", "guidance", "mission_lite", "controller_runner", "replay", "sim_world", "tello_io"]
        by = {c.name: c for c in self.children}
        for name in order:
            if name in by:
                self.stop_children([by[name]], STOP_WAIT_S)
        if "recorder" in by:
            self.stop_children([by["recorder"]], 10.0)
        for c in self.children:
            if c.proc and c.proc.returncode not in (0, None, -2, 130, -signal.SIGINT) and not c.reported:
                self.say(f"{c.name} exit code {c.proc.returncode} (log {c.log_path})")
        try:
            self.pub.close()
            self.sub.close()
            self.broker.stop()
        except Exception:
            pass
        if not IS_WIN:  # the frame producer is gone: remove its ring if it crashed without unlinking
            from flyfollow.runtime.bus import _unlink_raw

            _unlink_raw(shm_name(default_frame_ring()))
        self.say(f"done. session {self.dir}")
        self._log.close()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="start the whole drone runtime with one command")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sim", action="store_true", help="simulated Tello and world (sim_world)")
    g.add_argument("--dry", action="store_true", help="real Tello video/state, never send commands")
    g.add_argument("--send", action="store_true", help="REAL FLIGHT: commands go to the Tello")
    g.add_argument("--replay", metavar="SESSION_DIR", help="replay a recording instead of the drone")
    ap.add_argument("--detector", choices=("yolo", "external", "synthetic"), default=None,
                    help="yolo: built-in person detector, flyfollow.runtime.detector (default for --dry/--send); "
                         "external: another perception process publishes det; synthetic: sim ground truth or the "
                         "recorded det (default for --sim/--replay)")
    ap.add_argument("--scenario", choices=("follow", "find", "full"), default="follow", help="--sim world scenario")
    ap.add_argument("--seed", type=int, default=0, help="--sim world seed")
    ap.add_argument("--viz", action="store_true", help="start the fly brain/body viewer (flyfollow.viz.live)")
    ap.add_argument("--viz-brain", default=None)
    ap.add_argument("--no-operator", action="store_true", help="no console (tests, headless)")
    ap.add_argument("--script", default=None, help="operator script file (non-interactive)")
    ap.add_argument("--linger", type=float, default=1.0, help="script mode: seconds to listen after the last command")
    ap.add_argument("--plain", action="store_true", help="operator line UI instead of curses")
    ap.add_argument("--session", default=None)
    ap.add_argument("--record-root", default=None, help="default recordings/ (or $FLYFOLLOW_RECORDINGS)")
    ap.add_argument("--no-video", action="store_true", help="do not record video in --dry/--send")
    ap.add_argument("--replay-speed", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=None, help="bus ports PORT (pub) and PORT+1 (sub) instead of 5550/5551")
    ap.add_argument("--ring", default=None, help="frame ring name (default flyfollow_frames)")
    ap.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--no-controller", action="store_true")
    ap.add_argument("--takeoff-follow", action="store_true", help="--sim: mission_lite takes off, then FOLLOW")
    ap.add_argument("--no-mission", action="store_true", help="do not start mission_lite (the harness mode owner)")
    ap.add_argument("--no-guidance", action="store_true")
    ap.add_argument("--args", action="append", metavar='MODULE="ARGS"', help='extra args, e.g. --args tello_io="--foo 1"')
    ap.add_argument("--quiet", action="store_true", help="do not echo child output (it is still logged)")
    return ap


def main(argv: list[str] | None = None) -> None:
    sys.exit(Launcher(build_parser().parse_args(argv)).run())


if __name__ == "__main__":
    main()
