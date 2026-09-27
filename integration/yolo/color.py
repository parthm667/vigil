"""HSV colour check behind `require_color` (YOLO-World finds bottles, this keeps the team's blue one).

Copied from reachglass/detect/color_blob.py (only the two helpers the YOLO detector uses).
OpenCV HSV: H 0-180, S 0-255, V 0-255; each range is [h_lo, s_lo, v_lo, h_hi, s_hi, v_hi].
"""

from __future__ import annotations

import cv2
import numpy as np


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
