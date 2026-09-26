"""Simulated Tello: implements the Drone interface on top of the sim World.

Behaviour copied from the real drone where it matters for our code:
  * rc sticks -> body velocities after a dead time + first-order lag (numbers from the latency test)
  * discrete move/rotate execute on the drone with small random errors (the sim's odometry noise);
    rc is ignored while they run; the reply comes when they finish
  * auto-land after 15 s without any command; takeoff climbs to ~0.8 m
  * telemetry like djitellopy's state: yaw (deg, clockwise +, integer), pitch, h, tof, battery
  * the camera sees the world with a video delay (frames show the pose from `video_delay_s` ago)
Collisions stop the drone and are counted (tests require zero).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..drone.base import Drone, check_move, check_rotate, clamp_rc
from ..geometry import CameraModel
from ..sources.base import FrameSource
from ..types import Frame, Telemetry, wrap_deg
from .person_model import CameraPose
from .world import Renderer, World


@dataclass
class SimDroneParams:
    v_max: float = 1.0  # m/s horizontal at rc 100
    vz_max: float = 0.8  # m/s at rc 100
    yaw_rate_max: float = 100.0  # deg/s at rc 100
    tau: float = 0.35  # first-order response time constant (s)
    dead_time: float = 0.1  # s between a stick command and the response starting
    move_speed: float = 0.5  # m/s for discrete moves (Tello 'speed' command)
    move_accel: float = 1.0  # m/s^2
    rot_speed: float = 75.0  # deg/s for discrete rotations
    move_err_frac: float = 0.03  # 1-sigma distance error of a discrete move
    move_drift_frac: float = 0.02  # 1-sigma sideways error of a discrete move
    rot_err_deg: float = 1.5  # 1-sigma error of a discrete rotation
    hover_drift: float = 0.01  # m/s random walk while hovering
    takeoff_height: float = 0.8
    auto_land_s: float = 15.0
    radius: float = 0.12
    pitch_per_accel: float = 6.0  # deg of nose-down pitch per m/s^2 of forward acceleration
    battery_s: float = 600.0  # full -> empty
    video_delay_s: float = 0.15
    seed: int = 0


@dataclass
class _Discrete:
    kind: str  # takeoff | land | move | rotate
    target: np.ndarray  # target (x, y, z) for moves; unused for rotate
    remaining_deg: float = 0.0  # rotate: signed angle still to turn (cw 270 really turns 270 cw)
    speed: float = 0.0


class SimDrone(Drone):
    name = "sim"

    def __init__(self, world: World, x: float, y: float, heading_deg: float = 0.0, params: SimDroneParams | None = None,
                 camera: CameraModel | None = None):
        self.world = world
        self.p = params or SimDroneParams()
        self.rng = np.random.default_rng(self.p.seed)
        self.pos = np.array([x, y, 0.0])
        self.heading = heading_deg
        self.vel = np.zeros(3)  # world frame
        self.yaw_rate = 0.0
        self.t = 0.0
        self._flying = False  # physics: in the air (also during takeoff/landing)
        self._airborne_done = False  # takeoff completed
        self._rc_queue: deque = deque()  # (t_effective, (lr, fb, ud, yaw))
        self._rc = (0, 0, 0, 0)
        self._job: _Discrete | None = None
        self._result: str | None = "ok"
        self._last_cmd_t = 0.0
        self._accel_fwd = 0.0
        self.collisions = 0
        self.events: list[tuple[float, str]] = []
        self.battery = 100.0
        self.commands: list[tuple[float, str]] = []  # log of every command received
        self._speed = self.p.move_speed
        self.camera_model = camera
        self._camera: SimCamera | None = None

    # ------------------------------------------------------------------ Drone interface
    @property
    def flying(self) -> bool:
        """True once the takeoff has completed (like the Tello adapter), until landed."""
        return self._flying and self._airborne_done

    def _cmd(self, s: str) -> None:
        self._last_cmd_t = self.t
        self.commands.append((self.t, s))

    def _check_idle(self, cmd: str) -> None:
        # same contract as TelloDrone: a second discrete command while one runs is a programming error
        if self._job is not None:
            raise RuntimeError(f"'{cmd}' while '{self._job.kind}' is still running")

    def takeoff(self) -> None:
        self._check_idle("takeoff")
        self._cmd("takeoff")
        if self._flying:
            self._result = "error: already flying"
            return
        self._flying = True
        self._airborne_done = False
        self._result = None
        self._job = _Discrete("takeoff", np.array([self.pos[0], self.pos[1], self.p.takeoff_height]), speed=0.4)

    def land(self) -> None:
        if self._job is not None and self._job.kind == "land":
            return  # already landing (idempotent, like the adapter)
        if self._job is not None and self._job.kind in ("move", "rotate"):
            self.stop()  # like the adapter: landing cancels a running move
        self._check_idle("land")
        self._cmd("land")
        if not self._flying:
            self._result = "ok"
            return
        self._result = None
        floor = self.world.floor_below(self.pos[0], self.pos[1], self.pos[2])
        self._job = _Discrete("land", np.array([self.pos[0], self.pos[1], floor]), speed=0.5)

    def emergency(self) -> None:
        self._cmd("emergency")
        self._flying = False
        self._airborne_done = False
        self._job = None
        self.vel[:] = 0
        self.pos[2] = self.world.floor_below(self.pos[0], self.pos[1], self.pos[2])
        self._result = "ok"

    def rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        self._cmd(f"rc {lr} {fb} {ud} {yaw}")
        if self._job is not None:
            return  # the real Tello would abort the move; our interface forbids mixing
        self._rc_queue.append((self.t + self.p.dead_time, (clamp_rc(lr), clamp_rc(fb), clamp_rc(ud), clamp_rc(yaw))))

    def move(self, direction: str, cm: int) -> None:
        cm = check_move(direction, cm)
        self._check_idle(f"{direction} {cm}")
        self._cmd(f"{direction} {cm}")
        if not self.flying:
            self._result = "error: not flying"
            return
        d = cm / 100.0 * (1.0 + self.rng.normal(0, self.p.move_err_frac))
        side = cm / 100.0 * self.rng.normal(0, self.p.move_drift_frac)
        h = math.radians(self.heading)
        fwd = np.array([math.cos(h), math.sin(h), 0.0])
        right = np.array([-math.sin(h), math.cos(h), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        axis = {"forward": fwd, "back": -fwd, "right": right, "left": -right, "up": up, "down": -up}[direction]
        ortho = right if direction in ("forward", "back") else fwd
        target = self.pos + axis * d + ortho * side
        self._rc = (0, 0, 0, 0)
        self._rc_queue.clear()
        self._result = None
        self._job = _Discrete("move", target, speed=self._speed)

    def rotate(self, deg: int) -> None:
        deg = check_rotate(deg)
        cmd = f"{'cw' if deg > 0 else 'ccw'} {abs(deg)}"
        self._check_idle(cmd)
        self._cmd(cmd)
        if not self.flying:
            self._result = "error: not flying"
            return
        actual = deg + self.rng.normal(0, self.p.rot_err_deg)
        self._rc = (0, 0, 0, 0)
        self._rc_queue.clear()
        self._result = None
        self._job = _Discrete("rotate", self.pos.copy(), remaining_deg=actual, speed=self.p.rot_speed)

    def stop(self) -> None:
        """Hover: the running move is cancelled, the drone brakes with its normal lag (overshoots)."""
        self._cmd("stop")
        self._rc = (0, 0, 0, 0)
        self._rc_queue.clear()
        if self._job is not None and self._job.kind in ("move", "rotate"):
            self._job = None
            self._result = "ok"

    def busy(self) -> bool:
        return self._job is not None

    def last_result(self) -> str | None:
        return self._result

    def set_speed(self, cm_s: int) -> None:
        self._cmd(f"speed {cm_s}")
        self._speed = max(0.1, min(1.0, cm_s / 100.0))

    def telemetry(self) -> Telemetry:
        floor = self.world.floor_below(self.pos[0], self.pos[1], self.pos[2])
        h = math.radians(self.heading)
        v_fwd = self.vel[0] * math.cos(h) + self.vel[1] * math.sin(h)
        v_right = -self.vel[0] * math.sin(h) + self.vel[1] * math.cos(h)
        return Telemetry(
            t=self.t,
            yaw_deg=float(round(wrap_deg(self.heading))),
            pitch_deg=float(round(self.pitch_deg())),
            roll_deg=0.0,
            height_m=round(self.pos[2], 2),
            tof_m=round(self.pos[2] - floor, 2) if self._flying else None,
            vx=round(v_fwd, 1), vy=round(v_right, 1), vz=round(self.vel[2], 1),
            battery_pct=round(self.battery),
            flight_time_s=self.t,
        )

    def pitch_deg(self) -> float:
        """Nose-up positive (the sim's own convention)."""
        return -self.p.pitch_per_accel * self._accel_fwd

    def frame_source(self) -> FrameSource:
        if self._camera is None:
            raise RuntimeError("attach a SimCamera first (see sim.scenario.Sim)")
        return self._camera

    # ------------------------------------------------------------------ physics
    def camera_pose(self, pitch_deg: float = 0.0) -> CameraPose:
        return CameraPose(float(self.pos[0]), float(self.pos[1]), float(self.pos[2]), float(self.heading), pitch_deg)

    def step(self, dt: float) -> None:
        self.t += dt
        if self._flying:
            self.battery = max(0.0, self.battery - 100.0 * dt / self.p.battery_s)
            if self._job is None and self.t - self._last_cmd_t > self.p.auto_land_s:
                self.events.append((self.t, "auto-land: no command for 15 s"))
                self.land()
        while self._rc_queue and self._rc_queue[0][0] <= self.t:
            self._rc = self._rc_queue.popleft()[1]
        v_prev = self.vel.copy()
        if not self._flying and self._job is None:
            self.vel[:] = 0
            self.yaw_rate = 0
            return
        if self._job is not None:
            self._step_job(dt)
        else:
            lr, fb, ud, yw = self._rc
            h = math.radians(self.heading)
            fwd = np.array([math.cos(h), math.sin(h)])
            right = np.array([-math.sin(h), math.cos(h)])
            v_xy = (fb * fwd + lr * right) / 100.0 * self.p.v_max
            target = np.array([v_xy[0], v_xy[1], ud / 100.0 * self.p.vz_max])
            k = 1.0 - math.exp(-dt / self.p.tau)
            self.vel += (target - self.vel) * k
            if not any(self._rc):
                self.vel[:2] += self.rng.normal(0, self.p.hover_drift, 2) * math.sqrt(dt)
            self.yaw_rate += (yw / 100.0 * self.p.yaw_rate_max - self.yaw_rate) * k
            self.heading = wrap_deg(self.heading + self.yaw_rate * dt)
            self._advance(self.pos + self.vel * dt)
        h = math.radians(self.heading)
        a = (self.vel - v_prev) / dt
        self._accel_fwd = 0.7 * self._accel_fwd + 0.3 * (a[0] * math.cos(h) + a[1] * math.sin(h))

    def _advance(self, new: np.ndarray) -> bool:
        if self.world.collides(new, self.p.radius):
            self.collisions += 1
            self.events.append((self.t, f"COLLISION at {np.round(new, 2).tolist()}"))
            self.vel[:] = 0
            return False
        self.pos = new
        return True

    def _step_job(self, dt: float) -> None:
        j = self._job
        if j.kind == "rotate":
            step = math.copysign(min(abs(j.remaining_deg), j.speed * dt), j.remaining_deg)
            self.heading = wrap_deg(self.heading + step)
            self.yaw_rate = step / dt
            j.remaining_deg -= step
            if abs(j.remaining_deg) < 1e-9:
                self._finish("ok")
            return
        delta = j.target - self.pos
        dist = float(np.linalg.norm(delta))
        if dist < 1e-3:
            self._finish("ok")
            return
        # trapezoidal speed: accelerate, cruise, brake to stop at the target
        v_now = float(np.linalg.norm(self.vel))
        v_brake = math.sqrt(2 * self.p.move_accel * dist)
        v = min(j.speed, v_now + self.p.move_accel * dt, v_brake)
        step = min(dist, max(v, 0.02) * dt)
        new = self.pos + delta / dist * step
        self.vel = delta / dist * (step / dt)
        if not self._advance(new):
            self._finish("error: collision")
            return
        if step >= dist - 1e-9:
            self._finish("ok")

    def _finish(self, result: str) -> None:
        kind = self._job.kind if self._job is not None else ""
        if kind == "land" or (kind == "takeoff" and result != "ok"):
            self._flying = False  # landed, or never got off the ground
        if kind == "takeoff" and result == "ok":
            self._airborne_done = True
        self._job = None
        self._result = result
        self.vel[:] = 0
        self.yaw_rate = 0


class SimCamera(FrameSource):
    """The sim drone's forward camera: renders what it saw `video_delay_s` ago."""

    name = "sim"

    def __init__(self, drone: SimDrone, world: World, cam: CameraModel, fps: float = 15.0, seed: int = 0):
        self.drone, self.world, self.cam = drone, world, cam
        self.renderer = Renderer(cam, seed)
        self.fps = fps
        self._poses: deque = deque(maxlen=200)  # (t, pose, person snapshot)
        self._seq = 0
        self._last_capture = -1e9
        self._cache: Frame | None = None
        self._cache_t_cap: float | None = None
        self._recent: deque = deque(maxlen=30)  # (image, (pose, t_capture, person)) for the oracle detectors
        drone._camera = self

    def capture(self) -> None:
        """Called by the simulator every step: record the pose if a new frame is due."""
        t = self.drone.t
        if t - self._last_capture >= 1.0 / self.fps - 1e-9:
            self._last_capture = t
            p = self.world.person
            snap = None if p is None else (p.x, p.y, p.heading_deg)
            self._poses.append((t, self.drone.camera_pose(self.drone.pitch_deg()), snap))

    def _pose_for(self, t: float):
        best = None
        for item in self._poses:
            if item[0] <= t + 1e-9:
                best = item
        return best

    def read(self) -> Frame | None:
        if not self._poses:
            return None
        item = self._pose_for(self.drone.t - self.drone.p.video_delay_s)
        if item is None:
            return None
        t_cap, pose, person = item
        if self._cache is not None and self._cache_t_cap == t_cap:
            return self._cache
        saved = None
        if person is not None and self.world.person is not None:
            saved = (self.world.person.x, self.world.person.y, self.world.person.heading_deg)
            self.world.person.x, self.world.person.y, self.world.person.heading_deg = person
        img = self.renderer.render(self.world, pose.x, pose.y, pose.z, pose.heading_deg, pose.pitch_deg)
        if saved is not None:
            self.world.person.x, self.world.person.y, self.world.person.heading_deg = saved
        self._seq += 1
        img.flags.writeable = False  # shared frame: consumers must copy before drawing
        # frame time = arrival on the laptop (like the real reader), content from t_cap
        self._cache = Frame(img, self.drone.t, self._seq, self.name)
        self._cache_t_cap = t_cap
        self._recent.append((img, (pose, t_cap, person)))
        return self._cache

    def meta_for(self, image: np.ndarray):
        """(camera pose, capture time, person snapshot) of a frame this camera produced, else None."""
        for img, meta in reversed(self._recent):
            if img is image:
                return meta
        return None
