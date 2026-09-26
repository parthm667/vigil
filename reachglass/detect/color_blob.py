"""Colour target detector: finds brightly coloured blobs and reports them under a chosen label.

The fallback for the bottle (`--target color`): the default colour is the team's blue water bottle, so with it
in the room the whole pipeline behaves as if a bottle model had found it. The default target detector is
YOLO-World (see config.TARGET_PRESETS); `hsv_mask` is also its colour check.

Tune the colour on site: run `python -m reachglass.tools.hsv_picker` (step 12) or adjust `hsv_ranges`
(OpenCV HSV: H 0-180, S 0-255, V 0-255; each range is [h_lo, s_lo, v_lo, h_hi, s_hi, v_hi]).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..types import Detection
from .base import Detector


def parse_hsv_ranges(hsv_ranges) -> list[tuple[int, ...]]:
    """Validate [h_lo, s_lo, v_lo, h_hi, s_hi, v_hi] ranges; a wrap-around hue (h_lo > h_hi, e.g. red) is split."""
    out = []
    for r in hsv_ranges:
        if len(r) != 6:
            raise ValueError(f"hsv range needs 6 numbers [h_lo, s_lo, v_lo, h_hi, s_hi, v_hi], got {r}")
        h0, s0, v0, h1, s1, v1 = (int(v) for v in r)
        if not (0 <= h0 <= 180 and 0 <= h1 <= 180 and 0 <= s0 <= s1 <= 255 and 0 <= v0 <= v1 <= 255):
            raise ValueError(f"hsv range out of bounds (H 0-180, S/V 0-255, lo <= hi): {r}")
        if h0 > h1:  # written wrap-around style, e.g. [170, ..., 10, ...] for red
            out += [(h0, s0, v0, 180, s1, v1), (0, s0, v0, h1, s1, v1)]
        else:
            out.append((h0, s0, v0, h1, s1, v1))
    return out


def hsv_mask(image: np.ndarray, ranges: list[tuple[int, ...]]) -> np.ndarray:
    """255 where a BGR image's pixel is inside any of the parsed ranges."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for h0, s0, v0, h1, s1, v1 in ranges:
        m |= cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1))
    return m


class ColorBlobDetector(Detector):
    name = "color_blob"

    def __init__(self, label: str = "bottle", hsv_ranges=((100, 100, 70, 122, 255, 255),),
                 min_area_frac: float = 0.0003, min_fill: float = 0.35, max_aspect: float = 6.0,
                 ref_area_px: float = 900.0, max_detections: int = 5, blur: int = 5):
        self.label = label
        self.ranges = parse_hsv_ranges(hsv_ranges)
        self.min_area_frac = min_area_frac
        self.min_fill = min_fill
        self.max_aspect = max_aspect
        self.ref_area_px = ref_area_px  # blobs smaller than this get proportionally lower confidence
        self.max_detections = max_detections
        self.blur = blur | 1  # odd kernel

    @property
    def classes(self) -> list[str]:
        return [self.label]

    def mask(self, image: np.ndarray) -> np.ndarray:
        img = cv2.GaussianBlur(image, (self.blur, self.blur), 0) if self.blur > 1 else image
        m = hsv_mask(img, self.ranges)
        k = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        min_area = self.min_area_frac * w * h
        mask = self.mask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            # count real mask pixels: a coloured outline/ring has a large contour area but few pixels
            area = float(np.count_nonzero(mask[y:y + bh, x:x + bw]))
            if area < min_area:
                continue
            fill = area / float(bw * bh)
            aspect = max(bw / bh, bh / bw)
            if fill < self.min_fill or aspect > self.max_aspect:
                continue
            size = min(1.0, math.sqrt(area / self.ref_area_px))
            conf = float(np.clip((0.45 + 0.5 * fill) * (0.5 + 0.5 * size), 0.0, 0.99))
            out.append(Detection(self.label, conf, (float(x), float(y), float(x + bw), float(y + bh)), source=self.name))
        out.sort(key=lambda d: -d.conf)
        return out[: self.max_detections]
