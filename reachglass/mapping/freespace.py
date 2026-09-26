"""Free space in front of the drone, per horizontal sector of the image.

    fs = estimator.estimate(image, cam, altitude_m)
    fs.sectors -> [(bearing_deg, openness 0..1 or None, free_dist_m or None), ...]

Implementations (config explore.freespace):
  null    nothing metric known (DEFAULT). The explorer then uses free space proven by detections: every
          object seen at distance d certifies the line of sight to it (see mapping/grid.py).
  depth   EXPERIMENTAL, NOT RELIABLE on the Tello: Depth-Anything-V2-Small gives disparity up to an unknown
          scale/offset. The floor-geometry scale fit needs visible floor, but at 1.2 m altitude with the
          stream's 43 deg vertical FOV the floor only appears beyond ~3.1 m; and the model's learned
          "lower = closer" prior makes walls look like floor. Measured on simulator renders: a wall 1.5 m
          away scored as open as a 6 m room. Kept as a slot for a better model / a down-tilted camera.
  oracle  (simulator) ray-cast ground truth
The band that matters is the drone's own flight level: rows within +-band_deg of the horizon.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..geometry import CameraModel


@dataclass
class FreeSpace:
    sectors: list[tuple[float, float | None, float | None]]  # (bearing_deg, openness, free_dist_m)
    metric: bool = False

    def as_free_list(self) -> list[tuple[float, float | None]]:
        return [(b, d) for b, _, d in self.sectors]


class FreeSpaceEstimator(ABC):
    @abstractmethod
    def estimate(self, image: np.ndarray, cam: CameraModel, altitude_m: float | None) -> FreeSpace: ...


def sector_bearings(cam: CameraModel, n: int) -> list[float]:
    half = cam.hfov_deg / 2.0
    w = cam.hfov_deg / n
    return [-half + w * (k + 0.5) for k in range(n)]


class NullFreeSpace(FreeSpaceEstimator):
    def __init__(self, sectors: int = 5):
        self.n = sectors

    def estimate(self, image, cam, altitude_m) -> FreeSpace:
        return FreeSpace([(b, None, None) for b in sector_bearings(cam, self.n)])


class DepthFreeSpace(FreeSpaceEstimator):
    MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, sectors: int = 5, band_deg: float = 6.0, max_m: float = 5.0, input_px: int = 364,
                 device: str = "auto", cache_dir: str | None = None, min_floor_px: int = 400):
        from transformers import pipeline

        from ..detect.yolo import MODELS_DIR, auto_device

        dev = auto_device() if device == "auto" else device
        self.pipe = pipeline("depth-estimation", model=self.MODEL, device=dev if dev != "cpu" else -1,
                             model_kwargs={"cache_dir": cache_dir or str(MODELS_DIR / "hf")})
        self.n, self.band_deg, self.max_m, self.input_px, self.min_floor_px = sectors, band_deg, max_m, input_px, min_floor_px
        self.rng = np.random.default_rng(0)

    def disparity(self, image: np.ndarray) -> np.ndarray:
        """Relative disparity (bigger = closer) at the image's resolution."""
        import cv2
        from PIL import Image

        h, w = image.shape[:2]
        s = self.input_px / max(h, w)
        small = cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        out = self.pipe(Image.fromarray(small[..., ::-1]))
        d = np.asarray(out["predicted_depth"], dtype=np.float32).squeeze()
        return cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)

    def fit_floor(self, disp: np.ndarray, cam: CameraModel, altitude_m: float) -> tuple[float, float] | None:
        """RANSAC fit disp ~ a / D_floor + b on the lower image band; None if the floor is not visible enough."""
        h, w = disp.shape
        ys = np.arange(int(cam.y_of_elevation(-8.0)), h, 4)
        ys = ys[(ys > 0) & (ys < h)]
        if ys.size < 3:
            return None
        xs = np.arange(4, w - 4, 8)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        yy, xx = yy.ravel(), xx.ravel()
        inv_d = np.array([1.0 / (cam.ground_distance(float(y), altitude_m, x=float(x)) or np.inf) for y, x in zip(yy, xx)])
        val = disp[yy, xx]
        ok = np.isfinite(inv_d) & (inv_d > 0)
        inv_d, val = inv_d[ok], val[ok]
        if inv_d.size < self.min_floor_px // 16:
            return None
        best, best_in = None, 0
        tol = 0.05 * (np.percentile(val, 95) - np.percentile(val, 5) + 1e-6)
        for _ in range(60):
            i, j = self.rng.choice(inv_d.size, 2, replace=False)
            if abs(inv_d[i] - inv_d[j]) < 1e-3:
                continue
            a = (val[i] - val[j]) / (inv_d[i] - inv_d[j])
            if a <= 0:
                continue
            b = val[i] - a * inv_d[i]
            inl = np.abs(val - (a * inv_d + b)) < tol
            if inl.sum() > best_in:
                best_in, best = inl.sum(), (a, b)
        if best is None or best_in < 0.5 * inv_d.size:
            return None  # not mostly floor: do not trust a metric scale
        a, b = best
        inl = np.abs(val - (a * inv_d + b)) < tol
        A = np.column_stack([inv_d[inl], np.ones(inl.sum())])
        a, b = np.linalg.lstsq(A, val[inl], rcond=None)[0]
        return (float(a), float(b)) if a > 0 else None

    def estimate(self, image: np.ndarray, cam: CameraModel, altitude_m: float | None) -> FreeSpace:
        cam = cam.for_frame(image.shape[1], image.shape[0])
        disp = self.disparity(image)
        y0 = int(max(0, cam.y_of_elevation(self.band_deg)))
        y1 = int(min(image.shape[0], cam.y_of_elevation(-self.band_deg)))
        fit = self.fit_floor(disp, cam, altitude_m) if altitude_m else None
        bearings = sector_bearings(cam, self.n)
        near = []  # per sector: disparity of the closest things at flight level (high percentile)
        for b in bearings:
            xa = int(cam.x_of_bearing(b - cam.hfov_deg / (2 * self.n)))
            xb = int(cam.x_of_bearing(b + cam.hfov_deg / (2 * self.n)))
            near.append(float(np.percentile(disp[y0:y1, max(0, xa):min(image.shape[1], xb)], 90)))
        near = np.array(near)
        if fit is not None:
            a, bb = fit
            dist = [float(np.clip(a / max(v - bb, 1e-6), 0.1, self.max_m)) for v in near]
            return FreeSpace([(b, d / self.max_m, d) for b, d in zip(bearings, dist)], metric=True)
        lo, hi = float(np.percentile(disp, 2)), float(np.percentile(disp, 98))
        open_ = [float(np.clip(1.0 - (v - lo) / max(hi - lo, 1e-6), 0.0, 1.0)) for v in near]
        return FreeSpace([(b, o, None) for b, o in zip(bearings, open_)], metric=False)
