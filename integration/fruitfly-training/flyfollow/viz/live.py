"""Live Rerun dashboard fed by viz frames, and the non-blocking sink the control loop publishes to.

Control side (sim demo now, the live drone runtime later), at 20 Hz:

    from flyfollow.viz.live import VizSink
    from flyfollow.viz.frames import frame_from_controller
    sink = VizSink()                                   # ZeroMQ PUB bound to tcp://127.0.0.1:5557
    ...
    sink.publish(frame_from_controller(ctrl, t, tick=i, box=box, sticks=(yaw, fb), image=rgb))

publish() never blocks: the PUB socket drops messages once its small high-water mark is full, and a
missing or slow viewer costs the loop one pickle (well under 1 ms without an image).

Viewer side, a separate process (demo.py starts it for you):

    python -m flyfollow.viz.live --brain data/brains/pursuit_core1.npz [--connect tcp://127.0.0.1:5557] [--save run.rrd]

It spawns the Rerun viewer app, drains every queued frame each loop (so spike traces use all of them),
logs the brain and panels for the newest frame only, and renders the fly body on its own 30 fps clock.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from flyfollow.interfaces import OUTPUT_GROUPS
from flyfollow.viz import draw

DEFAULT_ADDRESS = "tcp://127.0.0.1:5557"
READY_LINE = "FLYFOLLOW VIEWER READY"

ABOUT_MD = """### What you are seeing

**Brain (center).** Every dot is a neuron's cell body at its real position in the MaleCNS connectome
(faint cloud: 20k of the fly's brain somata; bright: the 1,446-neuron pursuit circuit we simulate).
Neurons glow when they spike. The highlighted arcs are the connectome's strongest synapses of the
**LC10a -> AOTU025 / AOTU012 / AOTU019 -> DNa02** push-pull: each side's LC10a excites the same-side
DNa02 (amber) and, through the GABAergic AOTU019, inhibits the opposite DNa02 (violet).
Target on the right: DNa02 R fires, DNa02 L goes silent, the drone yaws right.

**Fly body (right).** TuragaLab's flybody model (a *female* fly; our connectome is male).
*Animated from the fly brain's live motor outputs; not a physics simulation.* Wing stroke amplitude
follows the DNa02 asymmetry (a right turn = larger left stroke), body yaw follows the yaw command,
the head turns toward the LC10a activity centroid; the wingbeat is slowed about 60x so you can see it.

**Biology:** wiring, cell types, spiking (LIF) model. **Engineering:** camera-to-neuron encoding,
readout to yaw, PID forward speed (the fly steers; the PID sets distance), safety governor, all animation choices.
"""


# --------------------------------------------------------------------------- sink (control side)
class VizSink:
    """Non-blocking frame publisher (ZeroMQ PUB, loopback only). Drops frames when the viewer lags."""

    def __init__(self, address: str = DEFAULT_ADDRESS, hwm: int = 4, max_image_width: int = 480):
        import zmq

        if not address.startswith(("tcp://127.0.0.1", "ipc://", "inproc://")):
            raise ValueError("VizSink binds to loopback / ipc only (frames are pickled)")
        self.zmq = zmq
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, hwm)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(address)
        self.address = address
        self.max_image_width = max_image_width
        self.sent = 0
        self.refused = 0  # rare: the socket said "try again"; PUB normally drops silently past the HWM
        self.publish_ms_max = 0.0
        self.publish_ms_sum = 0.0

    def publish(self, frame: dict) -> bool:
        t0 = time.perf_counter()
        img = frame.get("image")
        if img is not None and self.max_image_width and img.shape[1] > self.max_image_width:
            step = int(np.ceil(img.shape[1] / self.max_image_width))
            frame = {**frame, "image": np.ascontiguousarray(img[::step, ::step])}
        ok = True
        try:
            self.sock.send(pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL), self.zmq.NOBLOCK)
            self.sent += 1
        except self.zmq.Again:
            self.refused += 1
            ok = False
        ms = 1000.0 * (time.perf_counter() - t0)
        self.publish_ms_sum += ms
        self.publish_ms_max = max(self.publish_ms_max, ms)
        return ok

    def stats(self) -> dict:
        n = max(1, self.sent + self.refused)
        return {"sent": self.sent, "refused": self.refused, "publish_ms_mean": self.publish_ms_sum / n, "publish_ms_max": self.publish_ms_max}

    def close(self, end: bool = True) -> None:
        if end:
            try:
                self.sock.send(pickle.dumps({"t": -1.0, "end": True}), self.zmq.NOBLOCK)
            except Exception:  # noqa: BLE001, S110 (best effort end marker)
                pass
        self.sock.close(0)


def spawn_viewer(brain: str, address: str = DEFAULT_ADDRESS, save: str | None = None, timeout_s: float = 60.0,
                 extra: list[str] | None = None):
    """Start the viewer process and wait until it is subscribed. Returns the Popen."""
    import os
    import subprocess
    import threading

    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    paths = [str(root), str(root / "third_party" / "FlyDrones" / "src")]
    env["PYTHONPATH"] = os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    cmd = [sys.executable, "-m", "flyfollow.viz.live", "--brain", str(brain), "--connect", address]
    if save:
        cmd += ["--save", save]
    cmd += extra or []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    ready = threading.Event()

    def pump():
        for line in p.stdout:
            if READY_LINE in line:
                ready.set()
            else:
                print(f"[viewer] {line.rstrip()}", flush=True)
        ready.set()

    threading.Thread(target=pump, daemon=True).start()
    if not ready.wait(timeout_s):
        print(f"[viewer] not ready after {timeout_s:.0f} s; streaming anyway", flush=True)
    time.sleep(0.3)  # let the SUB connection settle (PUB/SUB slow joiner)
    return p


# --------------------------------------------------------------------------- dashboard (viewer side)
def blueprint():
    import rerun.blueprint as rrb

    rng = rrb.VisibleTimeRange("sim_time", start=rrb.TimeRangeBoundary.cursor_relative(seconds=-12.0),
                               end=rrb.TimeRangeBoundary.cursor_relative())
    brain3d = rrb.Spatial3DView(
        origin="brain", name="Fly brain: MaleCNS pursuit circuit (live spikes)",
        background=rrb.Background(kind=rrb.BackgroundKind.SolidColor, color=[7, 9, 13]),
        eye_controls=rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital, position=(0.0, -1050.0, 560.0), look_target=(0.0, 20.0, 10.0),
                                       eye_up=(0.0, 0.0, 1.0)),
        line_grid=rrb.LineGrid3D(visible=False),
    )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial2DView(origin="camera", name="Drone camera"),
                    rrb.Spatial2DView(origin="fly_eye", name="Fly's-eye view (LC10a bins)"),
                    row_shares=[2.2, 1.0],
                ),
                brain3d,
                rrb.Spatial2DView(origin="body", name="Fly body (flybody): animated from motor outputs"),
                column_shares=[1.0, 1.55, 1.45],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="plots/dn", name="Steering DN rates (Hz)", time_ranges=rng),
                rrb.TimeSeriesView(origin="plots/sticks", name="Sticks sent to the drone", time_ranges=rng),
                rrb.Spatial2DView(origin="topdown", name="Top view (sim)"),
                rrb.TextDocumentView(origin="about", name="What you are seeing"),
                column_shares=[1.2, 1.2, 1.0, 1.1],
            ),
            row_shares=[3.0, 1.35],
        ),
        collapse_panels=True,
    )


class LiveDashboard:
    """Logs frames into Rerun: native 3D brain, fly body images, panels, time series and the honesty text."""

    def __init__(self, brain: str | None = None, spawn: bool = True, save: str | None = None, body: bool = True,
                 app_id: str = "flyfollow_fly_brain", recording_name: str | None = None):
        import rerun as rr

        self.rr = rr
        rr.init(app_id, spawn=False)
        if recording_name:
            rr.send_recording_name(recording_name)
        if save:
            rr.save(save, default_blueprint=blueprint())
        elif spawn:
            # the venv's bin is often not on PATH (running .venv/bin/python directly): point at its rerun binary
            exe = Path(sys.executable).parent / "rerun"
            rr.spawn(default_blueprint=blueprint(), memory_limit="2GB", executable_path=str(exe) if exe.exists() else None)
        rr.send_blueprint(blueprint())
        self.geo = None
        self.activity = None
        self.body = None
        self.want_body = body
        self.panels = None
        self.last_t = None
        self.n_frames = 0
        self.body_ms = 0.0
        self.log_ms = 0.0
        if brain:
            self._setup(brain)

    def _setup(self, brain: str) -> None:
        from flyfollow.viz.brain_view import BrainActivity, BrainGeometry, log_static_rerun
        from flyfollow.viz.panels import CameraPanel, FlyEyePanel, TopDownPanel

        rr = self.rr
        self.geo = BrainGeometry.load(brain)
        self.activity = BrainActivity(self.geo)
        log_static_rerun(self.geo)
        rr.log("about", rr.TextDocument(ABOUT_MD, media_type=rr.MediaType.MARKDOWN), static=True)
        colors = {"DNa02_L": draw.LEFT, "DNa02_R": draw.RIGHT}
        for g in OUTPUT_GROUPS:
            if g.startswith(("DNa02", "DNa01", "DNg13")):
                c = colors.get(g, (120, 130, 150) if g.endswith("L") else (170, 150, 160))
                rr.log(f"plots/dn/{g}", rr.SeriesLines(colors=[c], names=[g], widths=[2.5 if g.startswith("DNa02") else 1.0]), static=True)
        rr.log("plots/sticks/yaw", rr.SeriesLines(colors=[[240, 240, 240]], names=["yaw stick (+ right)"], widths=[2.0]), static=True)
        rr.log("plots/sticks/fb", rr.SeriesLines(colors=[list(draw.TARGET)], names=["forward stick"], widths=[1.5]), static=True)
        rr.log("plots/sticks/bearing", rr.SeriesLines(colors=[[250, 204, 21]], names=["target bearing (deg)"], widths=[1.0]), static=True)
        self.panels = {"camera": CameraPanel(480, 360), "eye": FlyEyePanel(480, 170), "top": TopDownPanel(420, 320)}
        if self.want_body:
            from flyfollow.viz.fly_body import FlyBodyAnimator, RenderStyle, flybody_available

            if flybody_available():
                self.body = FlyBodyAnimator(RenderStyle(width=720, height=560))
            else:
                print("flybody assets missing (run scripts/setup_viz.sh); body panel disabled", flush=True)

    def ingest(self, frame: dict) -> None:
        """Every frame: update spike traces, body targets and trails (cheap). Does not log images."""
        if self.geo is None:
            self._setup((frame.get("meta") or {}).get("brain") or "pursuit_core1")
        t = float(frame["t"])
        dt = 0.05 if self.last_t is None else min(0.5, max(1e-3, t - self.last_t))
        self.last_t = t
        self._last_wall = time.perf_counter()
        if frame.get("counts") is not None:
            self.activity.update(frame["counts"], dt)
        else:
            self.activity.update_from_rates(frame, dt)
        if self.body is not None:
            from flyfollow.viz.frames import motor_state

            self.body.set_motor(motor_state(frame, self.geo.lc10a_bins))
        self.panels["top"].push(frame)
        rr = self.rr
        rr.set_time("sim_time", duration=t)
        dn = frame.get("dn") or {}
        for g, v in dn.items():
            if g.startswith(("DNa02", "DNa01", "DNg13")):
                rr.log(f"plots/dn/{g}", rr.Scalars(float(v)))
        st = frame.get("sticks") or {}
        if "yaw" in st:
            rr.log("plots/sticks/yaw", rr.Scalars(float(st["yaw"])))
        if "fb" in st:
            rr.log("plots/sticks/fb", rr.Scalars(float(st["fb"])))
        tg = frame.get("target") or {}
        if tg.get("bearing_deg") is not None:
            rr.log("plots/sticks/bearing", rr.Scalars(float(tg["bearing_deg"])))
        self.n_frames += 1

    def log_latest(self, frame: dict) -> None:
        """Brain and panels for the newest frame only (the viewer may skip older ones when behind)."""
        from flyfollow.viz.brain_view import log_frame_rerun

        t0 = time.perf_counter()
        rr = self.rr
        rr.set_time("sim_time", duration=float(frame["t"]))
        log_frame_rerun(self.geo, self.activity.glow())
        rr.log("camera", rr.Image(self.panels["camera"].render(frame)).compress(jpeg_quality=85))
        rr.log("fly_eye", rr.Image(self.panels["eye"].render(frame, self.geo.lc10a_bins)))
        rr.log("topdown", rr.Image(self.panels["top"].render(frame)).compress(jpeg_quality=85))
        self.log_ms = 1000 * (time.perf_counter() - t0)

    def log_body(self, dt: float) -> None:
        if self.body is None:
            return
        t0 = time.perf_counter()
        img = self.body.step(dt)
        if self.last_t is not None:  # between 20 Hz frames, advance by wall time so 30 fps body frames stay distinct
            since = min(0.1, time.perf_counter() - getattr(self, "_last_wall", time.perf_counter()))
            self.rr.set_time("sim_time", duration=self.last_t + since)
        self.rr.log("body", self.rr.Image(img).compress(jpeg_quality=88))
        self.body_ms = 1000 * (time.perf_counter() - t0)


def run_viewer(address: str = DEFAULT_ADDRESS, brain: str | None = None, save: str | None = None, spawn: bool = True,
               fps: float = 30.0, idle_exit_s: float = 0.0, body: bool = True) -> None:
    """Subscriber loop: drain, ingest all, log the newest, body at `fps`. Ends on an "end" frame or Ctrl-C."""
    import zmq

    dash = LiveDashboard(brain=brain, spawn=spawn, save=save, body=body)
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 16)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(address)
    print(READY_LINE, flush=True)
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)
    period = 1.0 / fps
    next_body = time.perf_counter()
    last_msg = time.perf_counter()
    received = 0
    skipped = 0
    try:
        while True:
            timeout = max(0.0, next_body - time.perf_counter())
            events = dict(poller.poll(int(timeout * 1000)))
            batch = []
            if sub in events:
                while True:
                    try:
                        batch.append(pickle.loads(sub.recv(zmq.NOBLOCK)))
                    except zmq.Again:
                        break
            ended = False
            for fr in batch:
                if fr.get("end"):
                    ended = True
                    continue
                dash.ingest(fr)
                received += 1
            real = [fr for fr in batch if not fr.get("end")]
            if real:
                skipped += len(real) - 1
                dash.log_latest(real[-1])
                last_msg = time.perf_counter()
            now = time.perf_counter()
            if now >= next_body:
                if dash.last_t is not None:
                    dash.log_body(period)
                next_body = max(next_body + period, now)
            if ended:
                print(f"stream ended: {received} frames received, {skipped} not drawn (viewer behind), "
                      f"log {dash.log_ms:.1f} ms, body {dash.body_ms:.1f} ms", flush=True)
                break
            if idle_exit_s and time.perf_counter() - last_msg > idle_exit_s:
                break
    except KeyboardInterrupt:
        pass
    finally:
        sub.close(0)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Rerun viewer process for flyfollow viz frames")
    ap.add_argument("--connect", default=DEFAULT_ADDRESS)
    ap.add_argument("--brain", default=None, help="brain .npz (default: from the first frame's meta)")
    ap.add_argument("--save", default=None, help="write an .rrd recording instead of spawning the viewer")
    ap.add_argument("--no-body", action="store_true")
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args(argv)
    from flyfollow.brain.build import ensure_flydrones

    ensure_flydrones()
    run_viewer(a.connect, a.brain, a.save, spawn=a.save is None, fps=a.fps, body=not a.no_body)


if __name__ == "__main__":
    main()
