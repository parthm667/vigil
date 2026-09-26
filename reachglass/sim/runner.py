"""The whole stack on the simulator: sim + perception + safety governor + mission + query inbox.

    r = SimRunner(load_config(), inbox=ScriptedInbox([(20.0, "find my water bottle")]))
    r.run(120)                    # or step() in your own loop (the app's --sim mode does)

Same loop as on the real drone (reachglass.app): sense -> safety check -> queries -> mission step.
Person and furniture come from oracle detectors (YOLO cannot see rendered boxes); the target is found by
the REAL colour-blob detector on rendered frames.
"""

from __future__ import annotations

import math

from ..behaviors import Ctx, sense
from ..config import Config
from ..detect import ColorBlobDetector, Detector
from ..drone.safety import SafetyGovernor
from ..mission import Mission, ScriptedInbox
from ..perception import Perception
from ..query import KeywordQueryParser
from .drone_sim import SimDroneParams
from .person_model import SimPerson
from .scenario import Sim, SimFreeSpace
from .world import PersonScript, World, demo_world


def demo_scenario(seed: int = 0) -> tuple[World, tuple[float, float], float]:
    """The demo room; the person stands at (2.6, 1.6) facing the room, the drone is behind them."""
    w = demo_world()
    script = PersonScript([(0.0, 2.6, 1.6, 20.0), (6.0, 2.6, 1.6, 20.0), (9.0, 3.1, 1.9, 35.0), (40.0, 3.1, 1.9, 35.0)])
    w.person = SimPerson(*script.at(0.0))
    w.person_script = script
    return w, (0.9, 1.0), 20.0


class SimRunner:
    def __init__(self, cfg: Config, world: World | None = None, drone_xy=None, drone_heading=None, inbox=None,
                 seed: int = 0, target_detector: Detector | None = None, oracle_freespace: bool = False,
                 dt: float = 1.0 / 15, announce=None):
        if world is None:
            world, xy, hd = demo_scenario(seed)
            drone_xy = drone_xy or xy
            drone_heading = hd if drone_heading is None else drone_heading
        self.cfg, self.dt = cfg, dt
        self.sim = Sim(world, drone_xy, drone_heading or 0.0, params=SimDroneParams(seed=seed), seed=seed)
        self.drone = SafetyGovernor(self.sim.drone, cfg.safety)
        self.perception = Perception(cfg, self.sim.person_detector, target_detector or ColorBlobDetector(),
                                     self.sim.context_detector, self.sim.cam)
        self.ctx = Ctx(cfg, self.drone, self.perception, freespace=SimFreeSpace(self.sim) if oracle_freespace else None)
        self.inbox = inbox or ScriptedInbox([])
        self.mission = Mission(self.ctx, KeywordQueryParser(), announce=announce or (lambda s: None))
        self.safety_events: list[str] = []
        self._started = False

    @property
    def t(self) -> float:
        return self.sim.t

    def step(self) -> None:
        if not self._started:
            self._started = True
            self.mission.start()
        self.sim.step(self.dt)
        sense(self.ctx, self.sim.camera, self.sim.t)
        last_frame_t = self.ctx.frame.t if self.ctx.frame is not None else None
        if self.drone.check(self.sim.t, last_frame_t) == "land":
            self.mission.force_land("safety")
        for text in self.inbox.poll(self.sim.t):
            self.mission.query(text)
        self.mission.step()

    def run(self, seconds: float, until=None) -> None:
        t_end = self.sim.t + seconds
        while self.sim.t < t_end:
            self.step()
            if until is not None and until(self):
                return

    # ------------------------------------------------------------------ ground truth helpers (tests, dashboard)
    def truth_person(self):
        p = self.sim.world.person
        return None if p is None else (p.x, p.y, p.heading_deg)

    def distance_drone_to(self, xy) -> float:
        return math.hypot(self.sim.drone.pos[0] - xy[0], self.sim.drone.pos[1] - xy[1])
