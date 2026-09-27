"""ReachGlass glasses hardware link: the ESP32-CAM video source and the Nano ESP32 data link.

Drop-in for the perception repo (github.com/nathanwuzhao/jerkgt13). Two ways to use it:

  * copy this file to ``reachglass/sources/glasses.py`` -- the repo imports below resolve as-is; or
  * leave it in the firmware repo and run it with the perception repo on PYTHONPATH.

Either way the hardware team can run it standalone (``python reachglass_glasses.py``) for a bench
smoke test: the repo imports are optional and only ``GlassesVideoSource`` needs them.

Hardware, protocol v1
---------------------
The ESP32-CAM is a TRANSPARENT bridge and never parses the protocol: one UDP datagram in becomes one
UART line to the Nano, and one UART line from the Nano becomes one UDP datagram out. So everything
here is really a conversation with the Nano; the CAM only supplies Wi-Fi and the MJPEG stream.

  video  HTTP MJPEG  http://<cam-ip>/stream
  data   UDP 4210, bidirectional. The PC must speak first -- that is how the CAM learns where to
         send telemetry.
  Nano -> PC  T,<seq>,<uptime_ms>,<tof_l_mm>,<tof_r_mm>,<press_l>,<press_r>,<link>
         ~15 Hz, not 20: the telemetry timer asks for 50 ms but the two blocking VL53L0X
         single-shot reads cost ~33 ms each, so the loop period floors at ~66 ms.
  PC -> Nano  H / P / Z / C / S   (see GlassesLink's methods)

The Nano releases both servos if it hears nothing for 600 ms. That failsafe exists because the pads
push on a person's face: a latched servo after a Wi-Fi drop is the worst thing this system can do.
The consequence for this module is that the 20 Hz resend thread is mandatory, not a nicety.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import threading
import time
from dataclasses import dataclass, replace

# Must be set before the first VideoCapture is constructed, exactly as the repo's video_stream.py
# does it. Without it FFmpeg queues frames it cannot keep up with, so the glasses view drifts further
# behind reality the longer the run lasts -- fatal for a "fine adjustment" camera whose whole job is
# telling someone where their hand is right now. Dropping late frames is the correct trade here.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "fflags;nobuffer|flags;low_delay|framedrop;1",
)

try:  # in-repo: use the real base classes so behaviour matches every other camera
    from reachglass.sources.opencv_sources import _ThreadedCapture, _readonly
    from reachglass.types import Frame

    IN_REPO = True
except ImportError:  # standalone bench use: just enough to run the smoke test at the bottom
    IN_REPO = False

    import numpy as np

    @dataclass
    class Frame:  # type: ignore[no-redef]
        image: "np.ndarray"
        t: float
        seq: int
        source: str = ""

    def _readonly(img):  # type: ignore[no-redef]
        img.flags.writeable = False
        return img

    class _ThreadedCapture:  # type: ignore[no-redef]
        """Stand-in for reachglass.sources.opencv_sources._ThreadedCapture.

        Only the parts GlassesVideoSource relies on. It exists so the hardware team can exercise the
        link without the perception repo checked out; in the repo the real class is used.
        """

        def __init__(self, name: str, pace_fps: float | None = None, loop: bool = False):
            self.name = name
            self._pace, self._loop = pace_fps, loop
            self._cap = None
            self._lock = threading.Lock()
            self._frame: Frame | None = None
            self._seq = 0
            self._stop = threading.Event()
            self._thread: threading.Thread | None = None
            self.finished = False

        def _open(self):
            raise NotImplementedError

        def start(self):
            if self._thread is None:
                self._stop.clear()
                self.finished = False
                with self._lock:
                    self._frame = None
                self._cap = self._open()
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
            return self

        def _run(self) -> None:
            raise NotImplementedError

        def read(self) -> Frame | None:
            with self._lock:
                return self._frame

        def wait_first(self, timeout_s: float = 10.0) -> Frame | None:
            t_end = time.time() + timeout_s
            while time.time() < t_end:
                f = self.read()
                if f is not None:
                    return f
                time.sleep(0.01)
            return None

        def stop(self) -> None:
            self._stop.set()

        def __enter__(self):
            return self.start()

        def __exit__(self, *exc) -> None:
            self.stop()


# ---------------------------------------------------------------------------------------------
# addresses
# ---------------------------------------------------------------------------------------------
#
# TWO different boards now, and they are NOT the same address:
#
#   video  the ESP32-CAM, 192.168.4.1   HTTP  -> GlassesVideoSource
#   data   the Nano ESP32, 192.168.4.50 UDP   -> GlassesLink
#
# The CAM runs the access point and serves MJPEG; the Nano joins that AP as a
# station on a static address and speaks the rover protocol directly. There is
# no UART bridge between them any more, so telemetry does NOT come from the CAM.

DATA_PORT = 4210
DISCOVERY_PORT = 4211

# The CAM answers HTTP, so it can be probed. It always hosts its own AP at .1 --
# it never joins another network, so there is only ever one address to try.
CAM_CANDIDATES = ("192.168.4.1",)

# The Nano speaks UDP only, so there is nothing to probe: it either beacons or
# it is at its compiled-in static address.
NANO_DEFAULT = "192.168.4.50"


def _wrap_deg(a: float) -> float:
    """Wrap to (-180, 180]. Duplicated from reachglass.types so this module stands alone."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def discover_nano(timeout_s: float = 3.0, port: int = DISCOVERY_PORT) -> str | None:
    """Listen for the Nano's 1 Hz beacon, "ROVER,<ip>,<port>". None on timeout.

    The Nano stops beaconing as soon as a peer registers, so this only finds a board that
    nobody is talking to yet. That is the intended behaviour: it is for startup, not health.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("", port))
        s.settimeout(timeout_s)
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            data, _ = s.recvfrom(128)
            parts = data.decode(errors="ignore").strip().split(",")
            if len(parts) >= 2 and parts[0] == "ROVER":
                return parts[1]
    except OSError:  # includes socket.timeout, and "port already bound" on a second listener
        return None
    finally:
        s.close()
    return None


def probe(ip: str, port: int = 80, timeout_s: float = 1.0) -> bool:
    """True if something answers TCP on ip:port. Used for the CAM only -- the Nano has no
    TCP listener at all, so a failed probe there would mean nothing."""
    try:
        with socket.create_connection((ip, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def resolve_cam(host: str | None = None) -> str:
    """Address of the ESP32-CAM (video). Explicit host wins, else probe the known addresses."""
    if host:
        return host
    for cand in CAM_CANDIDATES:
        if probe(cand):
            return cand
    raise RuntimeError(
        "cannot reach the ESP32-CAM. It hosts the AP at 192.168.4.1 -- are you joined to "
        '"rover"? Pass host=... to skip probing.'
    )


def resolve_nano(host: str | None = None, discovery_timeout_s: float = 3.0) -> str:
    """Address of the Nano ESP32 (ToF + haptics).

    Explicit host wins, then the beacon, then the compiled-in static address. Falling back to
    the static address rather than raising is deliberate: the Nano's IP is fixed in firmware, so
    a missed beacon (broadcast is the first thing a loaded AP drops) is not a reason to refuse
    to start. If it is genuinely absent, the first telemetry read reports stale and says so.
    """
    if host:
        return host
    ip = discover_nano(discovery_timeout_s)
    return ip if ip else NANO_DEFAULT


# ---------------------------------------------------------------------------------------------
# 1. video
# ---------------------------------------------------------------------------------------------


class GlassesVideoSource(_ThreadedCapture):
    """The ESP32-CAM MJPEG stream as a FrameSource.

    Used for point-of-view fine adjustment at the end of a run, where the drone's camera can no
    longer see what the person's hands are doing. ~10 fps is plenty; smoothness and latency matter
    far more than rate.

    Newest-frame-only reading comes from _ThreadedCapture, which is what the rest of the codebase
    expects: consumers poll read() and compare Frame.seq.

    Reconnects on its own. The ESP32 drops the stream when another client connects, when the sensor
    wedges, and when Wi-Fi hiccups -- all routine, none of them worth failing a mission over.
    """

    name = "glasses"

    def __init__(
        self,
        host: str | None = None,
        url: str | None = None,
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 3000,
        reconnect_delay_s: float = 0.5,
        discovery_timeout_s: float = 3.0,
    ):
        super().__init__("glasses")
        self._host = host
        self._url = url
        self._discovery_timeout_s = discovery_timeout_s
        # Bounded open/read: on a wedged MJPEG stream an untimed read() blocks forever, which would
        # hang both the reconnect and stop(). With a timeout the read simply fails and we reopen.
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.reconnect_delay_s = reconnect_delay_s
        self.reconnects = 0
        self._cap_lock = threading.Lock()  # guards self._cap between the reader and stop()
        # seq bookkeeping: a fresh capture starts its own frame numbering, but consumers use seq only
        # to answer "is this new?" and must never see it go backwards -- so count per connection and
        # carry an offset across reconnects, the same trick TelloVideoSource uses.
        self._conn_seq = 0
        self._offset = 0

    @property
    def url(self) -> str:
        if self._url is None:
            self._url = f"http://{resolve_cam(self._host)}/stream"
        return self._url

    def _open(self):
        import cv2

        cap = cv2.VideoCapture(
            self.url,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                self.open_timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                self.read_timeout_ms,
            ],
        )
        if not cap.isOpened():
            raise RuntimeError(f"cannot open glasses stream {self.url}")
        return cap

    def start(self) -> GlassesVideoSource:
        # Unlike the other sources, a camera that is not up yet must not kill the mission: come up
        # in the "disconnected" state and let the reader thread keep trying.
        if self._thread is None:
            self._stop.clear()
            self.finished = False
            with self._lock:
                self._frame = None
            try:
                cap = self._open()
            except RuntimeError:
                cap = None
            with self._cap_lock:
                self._cap = cap
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cap_lock:
                cap = self._cap
            if cap is None:
                if self._stop.wait(self.reconnect_delay_s):
                    return
                self._reopen(count=False)  # never connected yet: not a reconnect
                continue

            ok, img = cap.read()
            if not ok:
                # Read timeout or stream closed. The FFmpeg capture never recovers on its own.
                if self._stop.wait(self.reconnect_delay_s):
                    return
                self._reopen()
                continue

            with self._lock:
                self._conn_seq += 1
                self._seq = self._offset + self._conn_seq
                self._frame = Frame(_readonly(img), time.time(), self._seq, self.name)

    def _reopen(self, count: bool = True) -> None:
        with self._cap_lock:
            old, self._cap = self._cap, None
        if old is not None:
            old.release()  # safe: our reader is here, not inside read()
        try:
            new = self._open()
        except RuntimeError:
            new = None  # CAM still rebooting / off the AP: the loop retries
        with self._lock:
            self._offset += self._conn_seq
            self._conn_seq = 0
            if count and new is not None:
                self.reconnects += 1
        with self._cap_lock:
            if self._stop.is_set():
                if new is not None:
                    new.release()
                return
            self._cap = new

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # Worst case the reader is inside a read() that has not timed out yet.
            self._thread.join(timeout=self.read_timeout_ms / 1000.0 + 2.0)
            alive = self._thread.is_alive()
            self._thread = None
        else:
            alive = False
        with self._cap_lock:
            cap, self._cap = self._cap, None
        if cap is not None and not alive:  # never release under a read that is still running
            cap.release()
        with self._lock:
            self._frame = None


# ---------------------------------------------------------------------------------------------
# 2. data link
# ---------------------------------------------------------------------------------------------


@dataclass
class GlassesState:
    """Latest telemetry packet from the Nano. Immutable snapshot: GlassesLink.state() copies."""

    t: float = 0.0  # time.time() when the packet arrived on the PC
    seq: int = -1  # Nano's uint32 counter, wraps
    uptime_ms: int = 0  # millis() on the Nano
    tof_left_mm: int = -1  # -1 = out of range OR that sensor failed to initialise
    tof_right_mm: int = -1
    press_left: int = 0  # 0..100, press depth the Nano currently holds
    press_right: int = 0
    link: int = 0  # 1 = the Nano heard a command in the last 600 ms
    packets_lost: int = 0  # cumulative, from gaps in seq

    @property
    def stale(self) -> bool:
        """No telemetry for >0.5 s. At ~15 Hz that is ~7 missed packets: treat the readings as
        unknown rather than current. Obstacle warnings must not fire off stale ToF data."""
        return self.t == 0.0 or (time.time() - self.t) > 0.5

    @property
    def tof_min_mm(self) -> int | None:
        """Nearest valid reading of the two, or None if neither sensor is reporting."""
        vals = [v for v in (self.tof_left_mm, self.tof_right_mm) if v >= 0]
        return min(vals) if vals else None


class GlassesLink:
    """UDP telemetry/command client for the glasses (ToF sensors + the two haptic pads).

    Two daemon threads:
      * receiver -- parses T lines into a GlassesState
      * sender   -- RESENDS the current setpoint at `send_hz`. This is required. The Nano releases
                    both servos after 600 ms of silence, so a press only persists while this stream
                    keeps flowing. It also makes command loss harmless: H is an absolute setpoint,
                    so a dropped packet costs one stale frame, never a wrong position.

    Typical perception use::

        with GlassesLink() as glasses:
            ...
            glasses.direction(cue)               # cue = -1 left, 0 none, +1 right
                                                 # (+1 presses the LEFT pad: sides crossed)
            if (mm := glasses.state().tof_min_mm) is not None and mm < 700:
                glasses.pulse("B", count=2)      # obstacle ahead
    """

    def __init__(
        self,
        host: str | None = None,
        port: int = DATA_PORT,
        send_hz: float = 20.0,
        discovery_timeout_s: float = 3.0,
        max_press: int = 100,
    ):
        self._host_hint = host
        self.port = port
        self.send_hz = send_hz
        self.discovery_timeout_s = discovery_timeout_s
        self.max_press = int(_clamp(max_press, 0, 100))  # host-side ceiling; the Nano has its own
        self.host: str | None = None
        self._sock: socket.socket | None = None
        self._state = GlassesState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        # The line the sender repeats. Start released: whatever happens next, nothing presses a face
        # until perception asks for it.
        self._setpoint = "H,0,0"
        self._quiet_until = 0.0  # while a burst plays, suppress the H resend (H would cancel it)
        self.sends = 0
        self.packets = 0

    # -- lifecycle ----------------------------------------------------------------------------

    def start(self) -> GlassesLink:
        if self._sock is not None:
            return self
        self.host = resolve_nano(self._host_hint, self.discovery_timeout_s)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.25)
        self._sock = sock
        self._stop.clear()
        # Speak first: the CAM only learns where to send telemetry from an inbound packet. Z is the
        # right thing to open with -- it releases both pads, so a reconnect can never inherit a
        # press from a previous run.
        self._send_raw("Z")
        for target in (self._recv_loop, self._send_loop):
            th = threading.Thread(target=target, daemon=True)
            th.start()
            self._threads.append(th)
        return self

    def stop(self) -> None:
        """Release the servos, then shut down. Release first and more than once: this is the last
        chance to take the pads off someone's face, and it is one unacknowledged datagram."""
        self._set_setpoint("H,0,0", urgent=True)
        for _ in range(3):
            self._send_raw("Z")
            time.sleep(0.02)
        self._stop.set()
        for th in self._threads:
            th.join(timeout=1.0)
        self._threads.clear()
        sock, self._sock = self._sock, None
        if sock is not None:
            sock.close()

    def __enter__(self) -> GlassesLink:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- telemetry ----------------------------------------------------------------------------

    def state(self) -> GlassesState:
        with self._lock:
            return replace(self._state)  # a copy: callers may hold it across frames

    @property
    def connected(self) -> bool:
        return not self.state().stale

    def wait_first(self, timeout_s: float = 5.0) -> GlassesState | None:
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            st = self.state()
            if st.seq >= 0:
                return st
            time.sleep(0.01)
        return None

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            sock = self._sock
            if sock is None:
                return
            try:
                data, _ = sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                return  # socket closed under us by stop()
            for line in data.decode(errors="ignore").splitlines():
                self._parse(line.strip())

    def _parse(self, line: str) -> None:
        parts = line.split(",")
        if len(parts) != 8 or parts[0] != "T":
            return  # not telemetry (or a truncated datagram): ignore, do not crash a mission for it
        try:
            seq, uptime, tl, tr, pl, pr, link = (int(p) for p in parts[1:])
        except ValueError:
            return
        with self._lock:
            prev, lost = self._state.seq, self._state.packets_lost
            if prev >= 0:
                gap = (seq - prev) & 0xFFFFFFFF  # uint32 wrap
                if 1 < gap < 1000:  # a huge gap means the Nano rebooted, not that we lost packets
                    lost += gap - 1
            self._state = GlassesState(time.time(), seq, uptime, tl, tr, pl, pr, link, lost)
            self.packets += 1

    # -- commands -----------------------------------------------------------------------------

    def _send_raw(self, line: str) -> None:
        sock, host = self._sock, self.host
        if sock is None or host is None:
            return
        try:
            sock.sendto((line + "\n").encode(), (host, self.port))
            self.sends += 1
        except OSError:
            pass  # AP dropped for a moment; the resend thread covers it

    def _set_setpoint(self, line: str, urgent: bool = False) -> None:
        """Store the line the sender repeats, and normally send it at once.

        While a burst is playing a NON-urgent setpoint is stored but NOT sent, because the Nano
        cancels any burst in flight the moment it sees an H. Callers steer every perception frame,
        so sending here would cut every obstacle pulse down to one frame -- the warning would never
        actually be felt. The sender picks the stored line up as soon as the burst window ends.

        `urgent` is for the safety paths (release/stop) and the bench path (raw angles): those must
        take effect now and are entitled to kill a burst.
        """
        with self._lock:
            self._setpoint = line
            quiet = (not urgent) and time.time() < self._quiet_until
            if urgent:
                self._quiet_until = 0.0
        if not quiet:
            self._send_raw(line)  # send immediately; the resend thread only maintains it

    def _send_loop(self) -> None:
        period = 1.0 / self.send_hz
        next_t = time.time()
        while not self._stop.is_set():
            with self._lock:
                line = self._setpoint
                quiet = time.time() < self._quiet_until
            if not quiet:
                self._send_raw(line)
            next_t += period
            delay = next_t - time.time()
            if delay > 0:
                if self._stop.wait(delay):
                    return
            else:
                next_t = time.time()  # fell behind (GIL, scheduler): resync rather than burst

    def hold(self, left: float, right: float) -> None:
        """Hold press depth on each pad, 0 = released, 100 = fully pressed. THE PRIMARY INTERFACE.

        Absolute setpoint. Call it as often as you like; the resend thread keeps it alive between
        calls, so a caller running at 5 Hz is fine.
        """
        lp = int(round(_clamp(left, 0, self.max_press)))
        rp = int(round(_clamp(right, 0, self.max_press)))
        self._set_setpoint(f"H,{lp},{rp}")

    def pulse(self, side: str, count: int = 2, on_ms: int = 120, off_ms: int = 120, depth: int = 80) -> None:
        """Fire a pulse burst on "L", "R" or "B" (both). The rhythm is generated on the Nano, so it
        is immune to Wi-Fi jitter -- a burst timed from the PC would come out as an arrhythmic mush.

        Keep a burst under ~400 ms of total playing time. While it plays this client stops resending
        H (H would cancel the burst), and the Nano's 600 ms failsafe will cut a longer burst short.
        """
        side = side.upper()
        if side not in ("L", "R", "B"):
            raise ValueError("side must be L, R or B")
        count = max(1, int(count))
        on_ms, off_ms = max(1, int(on_ms)), max(10, int(off_ms))  # the Nano floors off_ms at 10
        depth = int(round(_clamp(depth, 0, self.max_press)))
        # The Nano plays `count` ON phases separated by count-1 gaps -- it does NOT trail a final
        # gap. Getting this right matters twice over: too short and the resumed H cancels the tail
        # of the burst, too long and the quiet window eats into the 600 ms failsafe.
        span_s = (count * on_ms + (count - 1) * off_ms) / 1000.0
        with self._lock:
            # After the burst the Nano is back at the held setpoint, so nothing else to restore.
            # The 0.4 s cap keeps the worst-case command gap (window + one 50 ms sender tick) near
            # 450 ms, leaving real margin to the 600 ms failsafe instead of a 95 ms sliver.
            self._quiet_until = time.time() + min(span_s, 0.4)
        self._send_raw(f"P,{side},{count},{on_ms},{off_ms},{depth}")

    def release(self) -> None:
        """Safety stop: release both pads now, and keep them released."""
        self._set_setpoint("H,0,0", urgent=True)
        self._send_raw("Z")

    def calibrate(self, side: str, released_deg: int, pressed_deg: int) -> None:
        """Trim one side's travel at runtime, so the mechanism can be adjusted without reflashing.
        NOT persisted across a Nano reboot -- re-send it after any power cycle."""
        side = side.upper()
        if side not in ("L", "R"):
            raise ValueError("side must be L or R")
        self._send_raw(f"C,{side},{int(released_deg)},{int(pressed_deg)}")

    def raw_angles(self, left_deg: int, right_deg: int) -> None:
        """Bench/calibration only: raw servo angles, bypassing the press mapping. This skips the
        0..100 press semantics, so the compile-time angle clamp on the Nano is the only thing
        keeping a pad off someone's cheekbone. Do not use this with the glasses being worn."""
        self._set_setpoint(f"S,{int(left_deg)},{int(right_deg)}", urgent=True)

    # -- the primary interface ----------------------------------------------------------------

    def direction(self, d: int, depth: int | None = None) -> tuple[int, int]:
        """Send a discrete direction cue. Returns the (left, right) press actually commanded.

        This is what perception calls. `d` is the whole signal:

            d = -1   go LEFT   -> presses the RIGHT pad
            d = +1   go RIGHT  -> presses the LEFT pad
            d =  0   no cue    -> releases both

        THE SIDES ARE CROSSED ON PURPOSE. The pad opposite the turn presses, so the wearer
        feels a nudge from the far side pushing them the way they should go. This is the
        INVERSE of this module's earlier convention, which pressed the near side -- if you are
        reading older notes or an older copy of HARDWARE_INTERFACE.md, they disagree with the
        firmware, and the firmware is right.

        Get this backwards and a blind person walks away from the thing they asked for. If the
        cue metaphor ever changes again, change it in `nano_tof.ino`'s D handler and here, and
        say so loudly in the handoff doc.

        Anything other than -1/0/+1 releases both pads: an out-of-range value is a bug
        upstream, and no cue beats a wrong cue. Only one pad is ever driven.
        """
        try:
            di = int(d)
        except (TypeError, ValueError):
            di = 0
        if di not in (-1, 0, 1):
            di = 0
        dp = self.max_press if depth is None else int(_clamp(depth, 0, self.max_press))
        self._set_setpoint(f"D,{di},{dp}")
        return (dp, 0) if di > 0 else (0, dp) if di < 0 else (0, 0)

    def guide(self, turn_deg: float | None, deadband_deg: float = 12.0) -> tuple[int, int]:
        """Convenience wrapper for callers that still have a continuous bearing error.

        Converts `turn_deg` (from ``Guidance.relative_to()``, clockwise positive, so positive
        means THE PERSON TURNS RIGHT) into the discrete cue above:

            |turn_deg| <= deadband_deg  ->  0   (aimed well enough; release)
            turn_deg  >  deadband_deg   ->  +1  (go right -> LEFT pad)
            turn_deg  < -deadband_deg   ->  -1  (go left  -> RIGHT pad)

        There is no proportional ramp any more. The upstream algorithm now emits -1/0/+1 with
        no magnitude, so a ramp here would be inventing precision the signal does not carry.
        Use `hold()` directly if you genuinely want graded pressure.

        `turn_deg=None` (heading unknown) releases both.
        """
        if turn_deg is None or not math.isfinite(turn_deg):
            return self.direction(0)
        a = _wrap_deg(float(turn_deg))
        if abs(a) <= deadband_deg:
            return self.direction(0)
        return self.direction(1 if a > 0 else -1)

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# ---------------------------------------------------------------------------------------------
# manual smoke test -- verifies the whole chain without any perception code
# ---------------------------------------------------------------------------------------------


def _smoke_test(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ReachGlass glasses hardware smoke test")
    ap.add_argument("--host", default=None, help="CAM IP; skips discovery")
    ap.add_argument("--video", action="store_true", help="also open the MJPEG stream and report fps")
    ap.add_argument("--no-sweep", action="store_true", help="telemetry only, never move the servos")
    ap.add_argument("--max-press", type=int, default=100, help="ceiling on press depth (0..100)")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after N s (0 = until Ctrl+C)")
    args = ap.parse_args(argv)

    link = GlassesLink(host=args.host, max_press=args.max_press)
    print("resolving the CAM ...")
    link.start()
    print(f"CAM at {link.host}, data on UDP {link.port}")

    video = None
    if args.video:
        if not IN_REPO:
            print("--video needs the perception repo on PYTHONPATH (GlassesVideoSource)")
        else:
            video = GlassesVideoSource(host=link.host).start()
            print(f"stream: {video.url}")

    if link.wait_first(5.0) is None:
        print("NO TELEMETRY. Check: Nano powered, GND shared with the CAM, D2<->GPIO14 and "
              "D3<->GPIO15 not swapped, both at 230400 8N1.")
    else:
        print("telemetry OK")

    # A slow sweep, one pad at a time, so a bystander can watch which pad moves and confirm the
    # left/right wiring matches the left/right label. This is the step that catches a swapped pair.
    script: list[tuple[str, float, float]] = []
    if not args.no_sweep:
        for pct in list(range(0, 101, 10)) + list(range(100, -1, -10)):
            script.append((f"LEFT  {pct:3d}%", pct, 0))
        for pct in list(range(0, 101, 10)) + list(range(100, -1, -10)):
            script.append((f"RIGHT {pct:3d}%", 0, pct))

    t0 = time.time()
    last_seq, frames, t_fps = -1, 0, time.time()
    fps = 0.0
    i = 0
    try:
        while True:
            if args.seconds and time.time() - t0 > args.seconds:
                break

            label = "idle"
            if script:
                label, lp, rp = script[i % len(script)]
                link.hold(lp, rp)
                if i and i % len(script) == 0:
                    link.pulse("B", count=2, on_ms=100, off_ms=100, depth=70)
                    label += "  (+both-pad pulse)"
                i += 1

            if video is not None:
                f = video.read()
                if f is not None and f.seq != last_seq:
                    last_seq, frames = f.seq, frames + 1
                if time.time() - t_fps >= 1.0:
                    fps, frames, t_fps = frames / (time.time() - t_fps), 0, time.time()

            st = link.state()
            flags = []
            if st.stale:
                flags.append("STALE")
            if not st.link:
                flags.append("nano-watchdog-open")  # the Nano is not hearing us
            if st.tof_left_mm < 0:
                flags.append("tofL?")
            if st.tof_right_mm < 0:
                flags.append("tofR?")
            vid = f"  video {fps:4.1f}fps rc={video.reconnects}" if video is not None else ""
            print(
                f"\rtof L{st.tof_left_mm:5d}mm R{st.tof_right_mm:5d}mm  "
                f"press L{st.press_left:3d} R{st.press_right:3d}  "
                f"seq {st.seq:7d} lost {st.packets_lost:4d}  {label:<24}{vid}  "
                f"{' '.join(flags):<38}",
                end="",
                flush=True,
            )
            time.sleep(0.2)  # 5 Hz print; the 20 Hz resend thread holds the setpoint meanwhile
    except KeyboardInterrupt:
        pass
    finally:
        print()
        if video is not None:
            video.stop()
        link.stop()  # releases both pads
        print(f"stopped. sent {link.sends} commands, received {link.packets} telemetry packets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke_test())
