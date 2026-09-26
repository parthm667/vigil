"""Person geometry from a detection (+ COCO-17 keypoints when the detector is a pose model).

Range (horizontal distance), first estimate that applies:
  full_height    whole body in frame: height prior (most reliable)
  head_elev      top of the head visible and the drone's altitude known: (altitude - person height) /
                 tan(depression). The only cue that does NOT change when the person turns, so it is the
                 one that matters when following from above/behind (feet out of frame). Set
                 perception.person_height_m to the wearer's real height: 5 cm error ~ 10-20 % range error.
  shoulders      both shoulders visible: shoulder-width prior, corrected by |cos facing|
  bbox_width     body-width prior (fallback, shrinks when the person turns)

Facing (PersonObs.facing_deg, see types.py: 0 = back to the camera, +90 = facing image-right,
+-180 = facing the camera):
  cos  from the signed shoulder vector: seen from behind, the person's RIGHT shoulder is on the image
       right (dx = x_rs - x_ls > 0); from the front it is on the image left (dx < 0). |dx| shrinks toward
       profile views; normalised by the shoulder width expected at the estimated range.
  sign from the nose when the face is visible (nose right of the shoulder midpoint -> facing right),
       otherwise from the ears (turning right from a back view shows the RIGHT ear).
"""

from __future__ import annotations

import math

import numpy as np

from ..geometry import CameraModel
from ..types import Detection, PersonObs

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH, L_HIP, R_HIP = 0, 1, 2, 3, 4, 5, 6, 11, 12
MIN_RANGE_M, MAX_RANGE_M = 0.3, 15.0
BODY_DEPTH_M = 0.28  # chest-to-back incl. arms, for profile views


class PersonEstimator:
    def __init__(self, person_height_m: float = 1.75, shoulder_width_m: float = 0.40, body_width_m: float = 0.48,
                 kp_conf: float = 0.4, edge_px: float = 3.0, min_height_diff_m: float = 0.15,
                 height_sd_m: float = 0.05, altitude_sd_m: float = 0.03, pitch_sd_deg: float = 1.0):
        self.person_height_m = person_height_m
        self.pitch_sd_deg = pitch_sd_deg  # uncertainty of the camera's vertical angle (3 deg if pitch is not used)
        self.height_sd_m = height_sd_m  # 0.05 for an unknown person; ~0.02 once the wearer is measured
        self.altitude_sd_m = altitude_sd_m
        self.shoulder_width_m = shoulder_width_m
        self.body_width_m = body_width_m
        self.kp_conf = kp_conf
        self.edge_px = edge_px
        self.min_height_diff_m = min_height_diff_m

    # ------------------------------------------------------------------ helpers
    def _kp(self, det: Detection, i: int):
        k = det.keypoints
        if k is None or k.shape[0] <= i or k[i, 2] < self.kp_conf:
            return None
        return float(k[i, 0]), float(k[i, 1])

    @staticmethod
    def _ok(r: float | None) -> bool:
        return r is not None and MIN_RANGE_M <= r <= MAX_RANGE_M and math.isfinite(r)

    # ------------------------------------------------------------------ facing
    def facing(self, det: Detection, cam: CameraModel, range_hint_m: float | None) -> tuple[float | None, float]:
        """(facing_deg, confidence 0..1), or (None, 0) without both shoulders."""
        ls, rs = self._kp(det, L_SH), self._kp(det, R_SH)
        if ls is None or rs is None:
            return None, 0.0
        dx = rs[0] - ls[0]
        if range_hint_m:
            expected = cam.fx * self.shoulder_width_m / range_hint_m
        else:
            expected = 0.83 * det.w  # shoulders ~83 % of a back-view body box
        c = float(np.clip(dx / max(expected, 1e-6), -1.0, 1.0))
        k = det.keypoints
        face = float(np.mean(k[[NOSE, L_EYE, R_EYE], 2]))
        sign, sign_conf = 0.0, 0.0
        nose = self._kp(det, NOSE)
        mid_x = 0.5 * (ls[0] + rs[0])
        if nose is not None and face >= self.kp_conf:
            off = (nose[0] - mid_x) / max(abs(dx), 0.25 * expected, 1.0)
            if abs(off) > 0.08:
                sign, sign_conf = math.copysign(1.0, off), min(1.0, abs(off) * 2)
        if sign == 0.0:
            ear_diff = float(k[R_EAR, 2] - k[L_EAR, 2])
            if abs(ear_diff) > 0.15:
                sign, sign_conf = math.copysign(1.0, ear_diff), min(1.0, abs(ear_diff) * 2)
        mag = math.degrees(math.acos(c))  # 0 (back) .. 180 (front)
        conf = (0.8 if mag < 30 or mag > 150 else 0.3) if sign == 0.0 else 0.5 + 0.5 * sign_conf
        # independent front/back vote from the face: a left/right shoulder label swap would flip the answer
        # by 180 deg with full confidence, so disagreement means "don't trust it"
        if (dx < 0 and face < 0.45) or (dx > 0 and face > 0.6 and mag < 60):
            conf = min(conf, 0.25)
        if not range_hint_m:
            conf = min(conf, 0.3)  # magnitude normalised by the box width: rough
        return (mag if sign == 0.0 else sign * mag), conf

    # ------------------------------------------------------------------ range
    def ranges(self, det: Detection, cam: CameraModel, altitude_m: float | None, body_pitch_deg: float = 0.0,
               facing_deg: float | None = None, facing_ok: bool = True) -> dict[str, float]:
        """facing_ok=False: the facing estimate is not trustworthy, so the shoulder cue is left out."""
        out: dict[str, float] = {}
        x1, y1, x2, y2 = det.bbox
        top_ok = y1 > self.edge_px
        bottom_ok = y2 < cam.height - self.edge_px
        side_ok = x1 > self.edge_px and x2 < cam.width - self.edge_px
        if top_ok and bottom_ok:
            r = cam.range_from_height(det.h, self.person_height_m, det.cx)
            if self._ok(r):
                out["full_height"] = r
        ls, rs = self._kp(det, L_SH), self._kp(det, R_SH)
        if ls is not None and rs is not None and facing_deg is not None and facing_ok:
            px = abs(rs[0] - ls[0])
            cos_f = abs(math.cos(math.radians(facing_deg)))
            if cos_f > 0.6:  # near front/back view: foreshortening is small and correctable
                r = cam.range_from_width(px / cos_f, self.shoulder_width_m, 0.5 * (ls[0] + rs[0]))
                if self._ok(r):
                    out["shoulders"] = r
        if top_ok and altitude_m is not None:
            diff = altitude_m - self.person_height_m
            if abs(diff) >= self.min_height_diff_m:
                r = cam.range_to_height(det.cx, y1, diff, body_pitch_deg)
                if self._ok(r):
                    out["head_elev"] = r
        if side_ok and bottom_ok:  # a box cut by the frame bottom may be narrower than the body
            # the body is ~0.28 m deep: seen in profile the box is much narrower than from behind
            if facing_deg is not None and facing_ok:
                f = math.radians(facing_deg)
                width = math.hypot(self.body_width_m * math.cos(f), BODY_DEPTH_M * math.sin(f))
            else:
                width = self.body_width_m
            r = cam.range_from_width(det.w, width, det.cx)
            if self._ok(r):
                out["bbox_width"] = r
        return out

    def rel_sigma(self, src: str, det: Detection, cam: CameraModel, altitude_m: float | None, r: float,
                  facing_deg: float | None, body_pitch_deg: float) -> float:
        """Expected relative error (1 sigma) of one range cue in the current geometry."""
        if src == "full_height":
            return 0.07  # people's heights vary ~4 %, box fit ~3 %
        if src == "shoulders":
            cos_f = abs(math.cos(math.radians(facing_deg))) if facing_deg is not None else 0.7
            return 0.10 + 0.5 * (1.0 - cos_f)  # shoulder widths vary ~10 %; worse when turned
        if src == "head_elev":
            diff = abs((altitude_m or 0.0) - self.person_height_m)
            e = math.radians(abs(cam.elevation_deg(det.bbox[1], body_pitch_deg, det.cx)))
            dz = math.hypot(self.height_sd_m, self.altitude_sd_m)  # height prior + altimeter
            # elevation error: pitch residual (or the whole pitch when telemetry pitch is not used) + box edge
            return math.hypot(dz / max(diff, 1e-3), math.radians(self.pitch_sd_deg) / max(math.sin(e) * math.cos(e), 1e-3))
        return 0.2 if facing_deg is not None else 0.35  # bbox width (facing-corrected or not)

    def fuse(self, ranges: dict[str, float], sigmas: dict[str, float]) -> tuple[float | None, str]:
        """Inverse-variance average in log space (errors are multiplicative)."""
        if not ranges:
            return None, ""
        w = {k: 1.0 / sigmas[k] ** 2 for k in ranges}
        tot = sum(w.values())
        r = math.exp(sum(w[k] * math.log(v) for k, v in ranges.items()) / tot)
        src = "+".join(sorted(ranges, key=lambda k: -w[k]))
        return r, src

    # ------------------------------------------------------------------ main
    def estimate(self, det: Detection, cam: CameraModel, altitude_m: float | None = None,
                 body_pitch_deg: float = 0.0) -> PersonObs:
        """`cam` must match the image the detection came from (use cam.for_frame(w, h)).
        range_src lists the fused cues, most trusted first."""
        bearing = cam.bearing_deg(det.cx)
        elevation = cam.elevation_deg(det.cy, body_pitch_deg)
        # pass 1: rotation-independent cues give the range hint that normalises the shoulder vector
        r0 = self.ranges(det, cam, altitude_m, body_pitch_deg)
        s0 = {k: self.rel_sigma(k, det, cam, altitude_m, v, None, body_pitch_deg) for k, v in r0.items()}
        hint, _ = self.fuse({k: v for k, v in r0.items() if k in ("full_height", "head_elev")}, s0)
        facing, fconf = self.facing(det, cam, hint)
        # pass 2: shoulders corrected by the facing estimate (only when it is trustworthy), then fuse
        ok = fconf >= 0.5
        r = self.ranges(det, cam, altitude_m, body_pitch_deg, facing, ok)
        sig = {k: self.rel_sigma(k, det, cam, altitude_m, v, facing if ok else None, body_pitch_deg) for k, v in r.items()}
        rng, src = self.fuse(r, sig)
        obs = PersonObs(det, bearing, elevation, rng, src, facing, fconf)
        # conservative lower bound (use it before moving TOWARD the person): the smallest cue, and a
        # width bound: with a trusted facing, the facing-corrected body width minus 15 % (people vary);
        # otherwise the narrowest a person can look (0.28 m deep, whatever way they face)
        lows = list(r.values())
        x1, _, x2, _ = det.bbox
        if x1 > self.edge_px and x2 < cam.width - self.edge_px:
            if ok and facing is not None:
                f = math.radians(facing)
                width = 0.85 * math.hypot(self.body_width_m * math.cos(f), BODY_DEPTH_M * math.sin(f))
            else:
                width = BODY_DEPTH_M
            wb = cam.range_from_width(det.w, width, det.cx)
            if wb is not None:
                lows.append(wb)
        obs.range_lo_m = min(lows) if lows else None
        return obs
