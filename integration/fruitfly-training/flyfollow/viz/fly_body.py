"""FlyBodyAnimator: the TuragaLab flybody fly, posed each frame from the brain's motor outputs.

This is a KINEMATIC animation (qpos + mj_forward + offscreen render), not a physics simulation:
we set joint angles directly; no forces, no aerodynamics, no trained flight controller. The model
is flybody's female Drosophila melanogaster (Apache-2.0, fetched by scripts/setup_viz.sh into
data/flybody/). Our brain is a male connectome (MaleCNS); the body is a female fly. Say so.

Mapping from motor signals to pose (engineering choices, not biology; see docs/VIZ.md):

    steer  = tanh(readout yaw drive)                            + = turning right
             (the steering DN asymmetry as the readout weighs it: sum over DN types of w_t (R_t/norm_Rt -
             L_t/norm_Lt) + b_yaw, the quantity that sets the yaw stick. Per-side normalization cancels
             core1's right bias, so the wings always agree with the drone's turn. Without a readout, e.g.
             the synthetic stream: tanh((DNa02_R - DNa02_L) / dn_scale_hz).)
    drive  = clip(fb_stick / fb_full, -1, 1)                    + = flying forward
    amp_L  = amp0 * (1 + amp_steer * steer) * (1 + amp_drive * drive)   right turn -> larger LEFT stroke
    amp_R  = amp0 * (1 - amp_steer * steer) * (1 + amp_drive * drive)
    body yaw   = -yaw_stick / yaw_full * max_body_yaw            (clockwise from above for a right turn)
    bank       =  steer * max_bank                               (right wing down for a right turn)
    pitch      =  pitch0 - pitch_drive * drive                   (nose-up hover posture, lower when driving)
    head yaw   =  clip(look_bearing, +-head_max)                 (toward the LC10a activity centroid)
    legs       =  flybody's retracted flight posture (joint springrefs)
    wingbeat   =  flybody's approximate base stroke pattern, shown SLOWED to beat_hz (a few Hz; the real
                  ~200 Hz is invisible on video), with translucent ghost wings trailing the stroke as a sweep.

All pose channels are smoothed with critically damped second-order filters so the fly moves like an
animal, not like a 20 Hz step function.

CLI (renders a short test clip and a few stills):
    python -m flyfollow.viz.fly_body --out runs/viz_check/body_test.mp4 --seconds 6
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from flyfollow.interfaces import data_root
from flyfollow.viz import draw

HONESTY_LABEL = "Animated from the fly brain's live motor outputs; not a physics simulation."
BODY_CREDIT = "flybody model (TuragaLab, Apache-2.0): a female D. melanogaster"

FLYBODY_REPO = "https://github.com/TuragaLab/flybody.git"
FLYBODY_COMMIT = "d015e9bfe441bd90ae431bac24c55cb74bdbce26"


def flybody_dir() -> Path:
    """Folder holding fruitfly.xml and its .obj meshes (scripts/setup_viz.sh). Override with FLYFOLLOW_FLYBODY."""
    return Path(os.environ.get("FLYFOLLOW_FLYBODY", data_root() / "flybody"))


def flybody_available(path: str | Path | None = None) -> bool:
    return (Path(path) if path else flybody_dir()).joinpath("fruitfly.xml").exists()


# --------------------------------------------------------------------------- mapping (pure, testable)
@dataclass
class BodyMapping:
    """Gains of the motor-signal -> pose map. Engineering constants chosen for a readable animation."""

    dn_scale_hz: float = 150.0  # DNa02 R-L difference that gives steer = tanh(1) = 0.76
    yaw_full: float = 60.0  # yaw stick that gives the full body yaw offset (governor max_yaw_stick)
    fb_full: float = 35.0  # forward stick that counts as full drive (governor max_fwd_stick)
    amp0: float = 1.0  # stroke amplitude scale on flybody's base pattern (about 126 deg peak to peak)
    amp_steer: float = 0.22  # +-22 % left/right amplitude difference at full steer (exaggerated for visibility)
    amp_drive: float = 0.08
    max_body_yaw_deg: float = 32.0
    max_bank_deg: float = 22.0
    pitch0_deg: float = 42.0  # nose-up hover posture (flybody's hover_up_dir is 47.5 deg)
    pitch_drive_deg: float = 12.0
    head_max_deg: float = 18.0
    head_gain: float = 0.9
    beat_hz: float = 3.2  # displayed (slowed) wingbeat frequency
    beat_drive: float = 0.25  # beat frequency rises 25 % at full forward drive


@dataclass
class MotorState:
    """What the animator needs from one brain tick."""

    dna02_l: float = 0.0  # Hz
    dna02_r: float = 0.0
    yaw_stick: float = 0.0  # final (post-governor) stick, -100..100, + = turn right
    fb_stick: float = 0.0  # + = forward
    look_bearing: float | None = None  # rad, + = right; LC10a activity centroid (None: no visual target)
    activity: float = 0.0  # 0..1 overall DN activity, only used for small idle effects
    steer_drive: float | None = None  # readout yaw drive (argument of its tanh); None: use raw DNa02 L/R


@dataclass
class PoseTargets:
    amp_l: float = 1.0  # stroke amplitude scale per wing
    amp_r: float = 1.0
    body_yaw: float = 0.0  # rad, world z (counterclockwise +), so a right turn is negative
    bank: float = 0.0  # rad about the body x axis, + = right wing down
    pitch: float = math.radians(42.0)  # rad nose-up
    head_yaw: float = 0.0  # rad, + = head turned to the fly's right
    beat_hz: float = 3.2

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=np.float64)

    @classmethod
    def from_array(cls, a: np.ndarray) -> PoseTargets:
        return cls(*[float(v) for v in a])


def steer_signal(ms: MotorState, m: BodyMapping) -> float:
    """Steering DN asymmetry in [-1, 1]; + = the brain is turning right."""
    if ms.steer_drive is not None and math.isfinite(ms.steer_drive):
        return math.tanh(ms.steer_drive)
    return math.tanh((ms.dna02_r - ms.dna02_l) / m.dn_scale_hz)


def pose_targets(ms: MotorState, m: BodyMapping | None = None) -> PoseTargets:
    """Motor state -> target pose (before smoothing). Pure function; see the module docstring."""
    m = m or BodyMapping()
    steer = steer_signal(ms, m)
    drive = max(-1.0, min(1.0, ms.fb_stick / m.fb_full))
    yaw = max(-1.0, min(1.0, ms.yaw_stick / m.yaw_full))
    amp_common = m.amp0 * (1.0 + m.amp_drive * drive)
    head = 0.0
    if ms.look_bearing is not None and math.isfinite(ms.look_bearing):
        hm = math.radians(m.head_max_deg)
        head = max(-hm, min(hm, m.head_gain * ms.look_bearing))
    return PoseTargets(
        amp_l=amp_common * (1.0 + m.amp_steer * steer),
        amp_r=amp_common * (1.0 - m.amp_steer * steer),
        body_yaw=-yaw * math.radians(m.max_body_yaw_deg),
        bank=steer * math.radians(m.max_bank_deg),
        pitch=math.radians(m.pitch0_deg - m.pitch_drive_deg * drive),
        head_yaw=head,
        beat_hz=m.beat_hz * (1.0 + m.beat_drive * max(0.0, drive)),
    )


# time constants (s) of the critically damped smoothing, per PoseTargets field
SMOOTH_TAU = PoseTargets(amp_l=0.10, amp_r=0.10, body_yaw=0.35, bank=0.25, pitch=0.45, head_yaw=0.12, beat_hz=0.5)


class PoseSmoother:
    """Critically damped second-order filter per channel: x'' = w^2 (target - x) - 2 w x'."""

    def __init__(self, initial: PoseTargets | None = None, tau: PoseTargets = SMOOTH_TAU):
        self.x = (initial or PoseTargets()).as_array()
        self.v = np.zeros_like(self.x)
        self.w = 1.0 / np.maximum(tau.as_array(), 1e-3)

    def step(self, target: PoseTargets, dt: float) -> PoseTargets:
        # substeps keep the semi-implicit Euler stable for small tau at 30 fps
        n = max(1, math.ceil(dt * float(self.w.max()) / 0.25))
        h = dt / n
        tgt = target.as_array()
        for _ in range(n):
            a = self.w**2 * (tgt - self.x) - 2.0 * self.w * self.v
            self.v += h * a
            self.x += h * self.v
        return PoseTargets.from_array(self.x)


def wing_pattern(phase: float) -> tuple[float, float, float]:
    """flybody's approximate base wingbeat (yaw = stroke, roll = deviation, pitch = rotation), phase in cycles.

    Same formula as flybody.tasks.pattern_generators.WingBeatPatternGenerator without a data file.
    yaw -0.8 is the wing forward (ventral reversal), yaw 1.4 is the wing back (dorsal reversal).
    """
    x = 2.0 * math.pi * (phase % 1.0)
    return 1.1 * math.sin(x - math.pi / 2), 0.25 * math.sin(1.5 * x) - 0.1, 1.35 * math.sin(x) + 0.8


YAW_CENTER = 0.3


def wing_angles(phase: float, amp: float) -> tuple[float, float, float]:
    """Stroke with amplitude scaled around the stroke center; rotation and deviation keep their shape."""
    y, r, p = wing_pattern(phase)
    return YAW_CENTER + amp * y, r, p


# --------------------------------------------------------------------------- quaternions
def _qaxis(axis, ang: float) -> np.ndarray:
    a = np.asarray(axis, np.float64)
    return np.r_[math.cos(ang / 2), math.sin(ang / 2) * a]


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def body_quat(body_yaw: float, pitch_up: float, bank: float) -> np.ndarray:
    """World orientation of the thorax: yaw about world z, then nose-up pitch, then bank about the body x axis.

    flybody frame: x forward (head), y left, z up. Positive rotation about y pitches the nose DOWN,
    so nose-up is a negative y rotation. Positive rotation about x lifts the left side (right wing down).
    """
    return _qmul(_qmul(_qaxis((0, 0, 1), body_yaw), _qaxis((0, 1, 0), -pitch_up)), _qaxis((1, 0, 0), bank))


# --------------------------------------------------------------------------- model build
@dataclass
class RenderStyle:
    width: int = 640
    height: int = 480
    distance: float = 0.70  # model units are cm; the fly is about 0.3 cm long
    azimuth: float = 20.0  # 0 = looking along the fly's +x from behind; + swings the camera to the fly's right
    elevation: float = -16.0
    lookat: tuple[float, float, float] = (-0.01, 0.0, 0.0)
    ghosts: int = 3  # translucent trailing wing copies (the sweep)
    ghost_lag: float = 0.045  # cycles between ghosts
    ghost_alpha: tuple[float, ...] = (0.30, 0.18, 0.09)
    offsamples: int = 8
    sky_top: tuple[float, float, float] = (0.13, 0.16, 0.23)
    sky_bottom: tuple[float, float, float] = (0.0, 0.0, 0.0)
    label: bool = True
    title: bool = True
    hud: bool = True  # small L/R stroke amplitude gauge
    extra: dict = field(default_factory=dict)


_WING_BODY_RE = r'(<body name="wing_{side}" childclass="wing".*?</body>)'


def _ghost_wing_xml(block: str, side: str, k: int) -> str:
    """Copy of a wing body with renamed elements, only the visible mesh geoms, and ghost materials."""
    lines = []
    for ln in block.splitlines():
        s = ln.strip()
        if s.startswith("<geom") and 'mesh="' not in s:
            continue  # collision / fluid / inertial helpers
        ln = re.sub(r'name="([^"]+)"', lambda mo, k=k: f'name="{mo.group(1)}_g{k}"', ln)
        ln = ln.replace('material="brown"', f'material="viz_ghost_brown_{k}"')
        ln = ln.replace('material="membrane"', f'material="viz_ghost_membrane_{k}"')
        if s.startswith("<geom"):
            extra = "".join(f' {k}="{v}"' for k, v in (("contype", 0), ("conaffinity", 0), ("mass", 0)) if f'{k}="' not in ln)
            ln = ln.replace("<geom", "<geom" + extra, 1)
        lines.append(ln)
        if s.startswith("<body"):  # the massless visual copy still needs an inertia to compile
            lines.append('        <inertial pos="0 0 0" mass="1e-7" diaginertia="1e-10 1e-10 1e-10"/>')
    return "\n".join(lines)


def build_xml(assets: Path, style: RenderStyle) -> str:
    """fruitfly.xml with our render tweaks: gradient sky, studio lights, ghost wings, absolute meshdir."""
    xml = (assets / "fruitfly.xml").read_text()
    xml = xml.replace('<compiler autolimits="true" angle="radian"/>',
                      f'<compiler autolimits="true" angle="radian" meshdir="{assets.resolve()}"/>', 1)
    st, sb = style.sky_top, style.sky_bottom
    mats = [
        (f'<texture name="viz_sky" type="skybox" builtin="gradient" rgb1="{st[0]} {st[1]} {st[2]}" '
         f'rgb2="{sb[0]} {sb[1]} {sb[2]}" width="512" height="3072"/>')
    ]
    for k in range(1, style.ghosts + 1):
        a = style.ghost_alpha[min(k - 1, len(style.ghost_alpha) - 1)]
        mats.append(f'<material name="viz_ghost_membrane_{k}" specular="0.3" shininess="0.5" rgba="0.62 0.76 0.9 {a:.3f}"/>')
        mats.append(f'<material name="viz_ghost_brown_{k}" rgba="0.3 0.18 0.1 {a * 0.8:.3f}"/>')
    xml = xml.replace("<asset>", "<asset>\n    " + "\n    ".join(mats), 1)
    # ghost wings go right after each real wing inside the thorax body
    for side in ("left", "right"):
        mo = re.search(_WING_BODY_RE.format(side=side), xml, flags=re.DOTALL)
        if mo is None:
            raise ValueError(f"wing_{side} body not found in fruitfly.xml")
        block = mo.group(1)
        ghosts = "\n".join(_ghost_wing_xml(block, side, k) for k in range(1, style.ghosts + 1))
        xml = xml.replace(block, block + "\n" + ghosts, 1)
    # studio lights in the world frame (the fly stays at the origin): warm key, cool fill, rim from the front
    lights = (
        '<light name="viz_key" mode="fixed" pos="-0.6 -0.8 1.2" dir="0.6 0.8 -1.2" diffuse="0.55 0.5 0.45" specular="0.3 0.3 0.3"/>\n'
        '    <light name="viz_fill" mode="fixed" pos="-0.8 0.9 0.3" dir="0.8 -0.9 -0.3" diffuse="0.18 0.22 0.3" specular="0 0 0"/>\n'
        '    <light name="viz_rim" mode="fixed" pos="1.0 0.2 0.9" dir="-1.0 -0.2 -0.9" diffuse="0.45 0.5 0.6" specular="0.5 0.5 0.5"/>'
    )
    xml = xml.replace("<worldbody>", "<worldbody>\n    " + lights, 1)
    return xml


# --------------------------------------------------------------------------- animator
class FlyBodyAnimator:
    """Poses and renders the flybody fly from brain motor signals. One instance per viewer.

        anim = FlyBodyAnimator()
        anim.set_motor(MotorState(...))     # whenever a brain tick arrives (e.g. 20 Hz)
        rgb = anim.step(1 / 30)             # every video frame (e.g. 30 fps); returns (H, W, 3) uint8
    """

    def __init__(self, style: RenderStyle | None = None, mapping: BodyMapping | None = None, assets: str | Path | None = None):
        import mujoco

        self.mj = mujoco
        self.style = style or RenderStyle()
        self.mapping = mapping or BodyMapping()
        self.assets = Path(assets) if assets else flybody_dir()
        if not flybody_available(self.assets):
            raise FileNotFoundError(f"flybody model not found in {self.assets}; run scripts/setup_viz.sh")
        t0 = time.perf_counter()
        self.model = mujoco.MjModel.from_xml_string(build_xml(self.assets, self.style))
        self.load_s = time.perf_counter() - t0
        m = self.model
        m.vis.quality.offsamples = self.style.offsamples
        m.vis.headlight.ambient[:] = (0.22, 0.22, 0.24)
        m.vis.headlight.diffuse[:] = (0.38, 0.38, 0.38)
        m.vis.headlight.specular[:] = (0.05, 0.05, 0.05)
        self.data = mujoco.MjData(m)
        self.renderer = mujoco.Renderer(m, self.style.height, self.style.width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = self.style.lookat
        self.cam.distance = self.style.distance
        self.cam.azimuth = self.style.azimuth
        self.cam.elevation = self.style.elevation
        self.q0 = m.qpos_spring.copy()  # flybody's retracted (flight) leg posture and folded defaults
        self.q0[0:3] = 0.0
        self._adr = {m.joint(j).name: int(m.jnt_qposadr[j]) for j in range(m.njnt)}
        self._wing_adr = {}
        for side in ("left", "right"):
            for k in range(self.style.ghosts + 1):
                suf = "" if k == 0 else f"_g{k}"
                self._wing_adr[(side, k)] = tuple(self._adr[f"wing_{a}_{side}{suf}"] for a in ("yaw", "roll", "pitch"))
        self.motor = MotorState()
        self.targets = pose_targets(self.motor, self.mapping)
        self.smoother = PoseSmoother(self.targets)
        self.pose = self.targets
        self.phase = 0.0
        self.t = 0.0
        self.render_ms = 0.0

    # ---- inputs
    def set_motor(self, motor: MotorState) -> None:
        self.motor = motor
        self.targets = pose_targets(motor, self.mapping)

    # ---- pose
    def apply_pose(self, pose: PoseTargets, phase: float) -> None:
        d, a = self.data, self._adr
        q = self.q0.copy()
        # a little life: tiny body bob with the (slowed) wingbeat and a slow abdomen sway
        bob = 0.0025 * math.sin(2 * math.pi * phase)
        q[0:3] = (0.0, 0.0, bob)
        q[3:7] = body_quat(pose.body_yaw, pose.pitch, pose.bank)
        q[a["head_abduct"]] = pose.head_yaw * -1.0  # head frame z points down-ish: negative abduct looks right
        q[a["head"]] = 0.05 * math.sin(2 * math.pi * 0.3 * self.t)
        for j in ("abdomen", "abdomen_2", "abdomen_3"):
            q[a[j]] = 0.03 * math.sin(2 * math.pi * 0.25 * self.t + 0.6)
        for side, amp in (("left", pose.amp_l), ("right", pose.amp_r)):
            for k in range(self.style.ghosts + 1):
                yaw, roll, pitch = wing_angles(phase - k * self.style.ghost_lag, amp)
                ia, ib, ic = self._wing_adr[(side, k)]
                q[ia], q[ib], q[ic] = yaw, roll, pitch
        # antennae twitch slightly toward the look direction
        for side, sgn in (("left", 1.0), ("right", -1.0)):
            q[a[f"antenna_abduct_{side}"]] = 0.15 * sgn * pose.head_yaw
        d.qpos[:] = q
        self.mj.mj_forward(self.model, d)

    def step(self, dt: float) -> np.ndarray:
        """Advance smoothing and the wing phase by dt seconds, pose, render, overlay. Returns RGB uint8."""
        self.t += dt
        self.pose = self.smoother.step(self.targets, dt)
        self.phase = (self.phase + self.pose.beat_hz * dt) % 1.0
        self.apply_pose(self.pose, self.phase)
        return self.render()

    def render(self) -> np.ndarray:
        t0 = time.perf_counter()
        r = self.renderer
        r.update_scene(self.data, self.cam)
        r.scene.flags[self.mj.mjtRndFlag.mjRND_SHADOW] = False
        img = r.render().copy()
        self.render_ms = 1000.0 * (time.perf_counter() - t0)
        if self.style.hud or self.style.title or self.style.label:
            with draw.TextBatch(img) as tb:  # one PIL round trip for all overlays
                if self.style.hud:
                    self._hud(img, tb)
                if self.style.title:
                    tb.text((14, 12), "FLY BODY", size=15, color=draw.TEXT, bold=True, shadow=True)
                    tb.text((14, 32), BODY_CREDIT, size=11, color=draw.MUTED, shadow=True)
                if self.style.label:
                    tb.text((self.style.width // 2, self.style.height - 12), HONESTY_LABEL, size=13,
                            color=(250, 214, 120), anchor="ms", shadow=True)
        return img

    def _hud(self, img: np.ndarray, tb: draw.TextBatch) -> None:
        """Left / right stroke amplitude gauges (what the steering DNs are doing to the wings)."""
        H, W = img.shape[:2]
        p = self.pose
        y0 = H - 54
        bw = 110
        for side, amp, x0, col, anchor, tx in (("L", p.amp_l, 18, draw.LEFT, "ls", 18), ("R", p.amp_r, W - 18 - bw, draw.RIGHT, "rs", W - 18)):
            frac = float(np.clip((amp - 0.6) / 0.8, 0, 1))
            img[y0 : y0 + 5, x0 : x0 + bw] = (38, 44, 56)
            if side == "L":
                img[y0 : y0 + 5, x0 : x0 + int(bw * frac)] = col
            else:  # right gauge fills from the right edge, mirroring the fly
                img[y0 : y0 + 5, x0 + bw - int(bw * frac) : x0 + bw] = col
            tb.text((tx, y0 - 6), f"{side} wing stroke {amp * 126:.0f}\u00b0", size=12, color=draw.MUTED, anchor=anchor)

    def close(self) -> None:
        try:
            self.renderer.close()
        except Exception:  # noqa: BLE001, S110 (closing a renderer whose GL context is already gone)
            pass


# --------------------------------------------------------------------------- CLI: a scripted test clip
def scripted_motor(t: float) -> MotorState:
    """A readable test program: hover, turn right, turn left, surge forward."""
    if t < 1.5:
        steer_hz, yaw, fb, look = 0.0, 0.0, 0.0, 0.0
    elif t < 3.0:
        steer_hz, yaw, fb, look = 260.0, 45.0, 10.0, math.radians(20)
    elif t < 4.5:
        steer_hz, yaw, fb, look = -240.0, -45.0, 10.0, math.radians(-20)
    else:
        steer_hz, yaw, fb, look = 0.0, 0.0, 35.0, 0.0
    base = 40.0
    return MotorState(dna02_l=base + max(0.0, -steer_hz), dna02_r=base + max(0.0, steer_hz), yaw_stick=yaw, fb_stick=fb,
                      look_bearing=look)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Render a short scripted flybody test clip and stills")
    ap.add_argument("--out", default="runs/viz_check/body_test.mp4")
    ap.add_argument("--stills", default="runs/viz_check", help="folder for PNG stills (empty to skip)")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--assets", default=None)
    a = ap.parse_args(argv)
    import imageio.v2 as imageio

    anim = FlyBodyAnimator(RenderStyle(width=a.width, height=a.height), assets=a.assets)
    print(f"model loaded in {anim.load_s:.2f} s from {anim.assets}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=8)
    n = int(a.seconds * a.fps)
    ms = []
    still_at = {int(0.8 * a.fps): "hover", int(2.6 * a.fps): "turn_right", int(4.1 * a.fps): "turn_left", int(5.6 * a.fps): "forward"}
    for i in range(n):
        anim.set_motor(scripted_motor(i / a.fps))
        t0 = time.perf_counter()
        img = anim.step(1.0 / a.fps)
        ms.append(1000 * (time.perf_counter() - t0))
        w.append_data(img)
        if a.stills and i in still_at:
            imageio.imwrite(Path(a.stills) / f"body_{still_at[i]}.png", img)
    w.close()
    ms_arr = np.array(ms[5:])
    print(f"wrote {a.out}: {n} frames, step+render {ms_arr.mean():.1f} ms mean, {np.percentile(ms_arr, 95):.1f} ms p95")


if __name__ == "__main__":
    main()
