"""Perception pipeline: one frame (+ telemetry) in, one PerceptionResult out.

    per = Perception.from_config(cfg)
    per.set_mode("follow")               # follow | search | approach | guide | idle  (which detectors run)
    per.set_target("bottle")             # class to lock for search/approach
    res = per.update(frame, telemetry)   # PerceptionResult (types.py)

Detectors per role, each swappable in config:
  person   people (+ keypoints) -> locked person, range, facing
  target   the thing we look for (dummy colour blob now, trained bottle model later)
  context  furniture etc. for the exploration prior; also a second source of targets (e.g. "chair")
"""

from __future__ import annotations

import time

from .config import Config
from .detect import DETECTORS, Detector
from .geometry import CameraModel, camera_from_config
from .person import PersonEstimator
from .track import SimpleTracker, TargetLock, largest, most_confident
from .types import Detection, Frame, ObjectObs, PerceptionResult, Telemetry, TargetObs

MODES = ("follow", "search", "approach", "guide", "idle")
MIN_WIDTH_PX = 25  # below this a width-based range is too coarse and blur-inflated (2 px blur = 8 % at 25 px)


class Perception:
    def __init__(self, cfg: Config, person: Detector | None = None, target: Detector | None = None,
                 context: Detector | None = None, camera: CameraModel | None = None):
        self.cfg = cfg
        pc, tc = cfg.perception, cfg.tracking
        self.detectors = {"person": person, "target": target, "context": context}
        self.camera = camera or camera_from_config(cfg.camera)
        self.tracker = SimpleTracker(tc.iou_match, tc.center_match_frac, tc.max_age_s)
        self.person_lock = TargetLock(self.tracker, tc.confirm_hits, tc.confirm_window, 0.99, tc.lost_after_s, select=largest)
        self.person_lock.set_class("person")
        self.target_lock = TargetLock(self.tracker, tc.confirm_hits, tc.confirm_window, tc.confirm_conf, tc.lost_after_s,
                                      select=most_confident, plausible=self._plausible_target)
        # without telemetry pitch (pitch_sign 0) the head-elevation cue must be trusted less
        self.estimator = PersonEstimator(pc.person_height_m, pc.shoulder_width_m, pc.body_width_m,
                                         height_sd_m=pc.person_height_sd_m, pitch_sd_deg=1.0 if pc.pitch_sign else 3.0)
        self._person_boxes: list = []
        self._person_boxes_t = -1e9
        self.mode = "idle"
        self.target_cls: str | None = None
        self._frame_i = 0
        self._mode_i = 0
        self._cam_for: CameraModel = self.camera
        self.last: PerceptionResult | None = None

    @classmethod
    def from_config(cls, cfg: Config, camera: CameraModel | None = None) -> Perception:
        pc = cfg.perception
        return cls(cfg, DETECTORS.build(pc.person_detector), DETECTORS.build(pc.target_detector),
                   DETECTORS.build(pc.context_detector), camera)

    # ------------------------------------------------------------------ control
    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != self.mode:
            self._mode_i = 0  # every detector runs on the first frame of a new mode, then by stride
        self.mode = mode

    def set_target(self, cls: str | None) -> None:
        self.target_cls = cls
        self.target_lock.set_class(cls)

    def vocabulary(self) -> list[str]:
        """Classes a user may ask for: everything the target and context detectors know (not people)."""
        names = set()
        for role in ("target", "context"):
            d = self.detectors[role]
            if d is not None:
                names.update(d.classes)
        names.discard("person")
        return sorted(names)

    def reset_tracks(self) -> None:
        """After a big discontinuity (e.g. a 45 deg rotation) old boxes cannot be associated anyway."""
        self.tracker.reset()
        self.person_lock.release()
        self.target_lock.release()

    # ------------------------------------------------------------------ helpers
    def _plausible_target(self, det: Detection) -> bool:
        """Single-frame confirmation needs a plausible distance AND shape (a blue blob of any shape is not
        a bottle; upright or lying, the long/short side ratio must be within 2x of the prior's)."""
        pc = self.cfg.perception
        h, w = pc.object_heights_m.get(det.cls), pc.object_widths_m.get(det.cls)
        if h is None and w is None:
            return True
        if h and w and det.w > 0 and det.h > 0:
            prior = max(h, w) / min(h, w)
            seen = max(det.h, det.w) / min(det.h, det.w)
            if not prior / 2.0 <= seen <= prior * 2.0:
                return False
        r, _ = self.object_range(det, self._cam_for)
        lo, hi = pc.target_range_m
        return r is not None and lo <= r <= hi

    def owner_of(self, cls: str | None) -> str | None:
        """Which detector role reports this class (the target detector first)."""
        if cls is None:
            return None
        for role in ("target", "context", "person"):
            d = self.detectors[role]
            if d is not None and cls in d.classes:
                return role
        return None

    def object_range(self, det: Detection, cam: CameraModel, floor_alt: float | None = None,
                     pitch: float = 0.0) -> tuple[float | None, str]:
        """Distance from the class size priors. Height is preferred (more pixels); for round objects the
        width takes over when the height looks truncated (partly hidden, or cut by the image edge).
        floor_alt (objects standing on the floor, i.e. furniture): from well above, the bottom is below the
        frame; the depression of the TOP edge then gives it, like a person's head."""
        pc = self.cfg.perception
        h, w = pc.object_heights_m.get(det.cls), pc.object_widths_m.get(det.cls)
        edge = 3
        full_v = det.bbox[1] > edge and det.bbox[3] < cam.height - edge
        full_h = det.bbox[0] > edge and det.bbox[2] < cam.width - edge
        r_h = cam.range_from_height(det.h, h, det.cx) if (h and full_v) else None
        # a thin far object is only a few pixels wide and blur widens it: width is only trusted when wide enough
        r_w = cam.range_from_width(det.w, w, det.cx) if (w and full_h and det.w >= MIN_WIDTH_PX) else None
        if r_h is not None and r_w is not None and r_h > 1.3 * r_w:
            return r_w, "width_prior(occluded)"
        if r_h is not None:
            return r_h, "height_prior"
        if r_w is not None:
            return r_w, "width_prior"
        if h and floor_alt is not None and floor_alt - h >= 0.3 and det.bbox[1] > edge:
            r_t = cam.range_to_height(det.cx, det.bbox[1], floor_alt - h, pitch)
            if r_t is not None:
                return r_t, "top_elevation"
        return None, ""

    def _object_obs(self, det: Detection, cam: CameraModel, pitch: float, floor_alt: float | None = None) -> ObjectObs:
        rng, src = self.object_range(det, cam, floor_alt, pitch)
        return ObjectObs(det, cam.bearing_deg(det.cx), cam.elevation_deg(det.cy, pitch, det.cx), rng, src)

    @staticmethod
    def _touches_border(det: Detection, width: int, height: int, margin_px: float = 2.0) -> bool:
        x1, y1, x2, y2 = det.bbox
        return x1 <= margin_px or y1 <= margin_px or x2 >= width - margin_px or y2 >= height - margin_px

    def _inside_person(self, det: Detection, persons: list[tuple[float, float, float, float]]) -> bool:
        """Mostly (>= 50 %) inside a person's box: e.g. blue jeans must not pass as the blue dummy."""
        for p in persons:
            ix = max(0.0, min(det.bbox[2], p[2]) - max(det.bbox[0], p[0]))
            iy = max(0.0, min(det.bbox[3], p[3]) - max(det.bbox[1], p[1]))
            if ix * iy >= 0.5 * max(det.area, 1e-9):
                return True
        return False

    def _stride(self, role: str) -> int:
        """Run this detector every Nth frame in the current mode (0 = never)."""
        if self.detectors[role] is None:
            return 0
        strides = self.cfg.perception.stride.get(self.mode, {})
        stride = strides.get(role, 0)
        if self.mode in ("search", "approach") and role == self.owner_of(self.target_cls):
            # whoever reports the target runs at least at the target's rate (e.g. the context detector for "chair")
            stride = min(v for v in (stride, strides.get("target", 1)) if v) if (stride or strides.get("target")) else 1
        return stride

    def _run(self, role: str) -> bool:
        stride = self._stride(role)
        return bool(stride) and self._mode_i % stride == 0

    def active_roles(self) -> set[str]:
        """Detector roles that run in the current mode (each at its own stride)."""
        return {r for r in ("person", "target", "context") if self._stride(r)}

    # ------------------------------------------------------------------ main
    def update(self, frame: Frame, telemetry: Telemetry | None = None) -> PerceptionResult:
        if self.last is not None and frame.seq == self.last.seq and frame.t == self.last.t:
            return self.last  # already processed: a repeated frame must not count as a new sighting
        t0 = time.perf_counter()
        self._frame_i += 1
        cam = self._cam_for = self.camera.for_frame(frame.width, frame.height)
        tel = telemetry or Telemetry(frame.t)
        altitude = tel.floor_altitude_m()  # person range needs height above the FLOOR, not above a table
        pitch = (tel.pitch_deg or 0.0) * self.cfg.perception.pitch_sign

        dets: list[Detection] = []
        ran = {}
        for role in ("person", "target", "context"):
            ran[role] = self._run(role)
            if ran[role]:
                found = self.detectors[role].detect(frame.image)
                if role == "person":
                    self._person_boxes = [d.bbox for d in found if d.cls == "person"]
                    self._person_boxes_t = frame.t
                if role == "context":  # the person/target detectors own those classes
                    found = [d for d in found if d.cls != "person"]
                if role == "target":
                    # cut by the frame edge: cannot be sized or shape-checked, and a slice of someone's blue
                    # jeans at the edge is exactly what the person detector misses. A real target gets
                    # picked up a moment later, once the drone turns toward it.
                    found = [d for d in found if not self._touches_border(d, frame.width, frame.height)]
                if role == "target" and frame.t - self._person_boxes_t < 1.5:
                    # people from the latest person-detector run (it runs only every Nth frame)
                    found = [d for d in found if not self._inside_person(d, self._person_boxes)]
                dets.extend(found)
        dets = self.tracker.update(dets, frame.t, (frame.width, frame.height))

        res = PerceptionResult(frame.t, frame.seq, 0.0, (frame.width, frame.height), dets)
        persons = [d for d in dets if d.cls == "person"]
        res.persons = [self.estimator.estimate(d, cam, altitude, pitch) for d in persons]
        if ran["person"] or persons:
            ps = self.person_lock.update(dets, frame.t, (frame.width, frame.height), ran=ran["person"])
            if ps.det is not None:
                res.person = next((p for p in res.persons if p.det.track_id == ps.det.track_id), None)
        owner = self.owner_of(self.target_cls) or "target"
        res.target_ran = ran.get(owner, False) or self.detectors.get(owner) is None
        if self.target_cls:
            ts = self.target_lock.update(dets, frame.t, (frame.width, frame.height), ran=ran.get(owner, False))
            for d in dets:
                if d.cls == self.target_cls:
                    o = self._object_obs(d, cam, pitch)
                    tob = TargetObs(o.det, o.bearing_deg, o.elevation_deg, o.range_m, o.range_src)
                    if ts.det is not None and d.track_id == ts.det.track_id:
                        tob.confirmed = ts.confirmed
                        res.target = tob
                    res.targets.append(tob)
        res.context = [self._object_obs(d, cam, pitch, altitude) for d in dets if d.cls not in ("person", self.target_cls)]
        res.ran = ran
        res.person_unseen_s = self.person_lock.unseen_s(frame.t)
        res.target_unseen_s = self.target_lock.unseen_s(frame.t) if self.target_cls else float("inf")
        res.latency_ms = 1000.0 * (time.perf_counter() - t0)
        self._mode_i += 1
        self.last = res
        return res
