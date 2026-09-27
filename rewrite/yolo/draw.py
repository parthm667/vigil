"""Draw detections onto a BGR image (for windows and recordings)."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection

KEYPOINT_MIN_CONF = 0.4


def draw(image: np.ndarray, detections: list[Detection], color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Boxes with "class conf" labels, plus confident person keypoints, drawn in place. Returns the image."""
    for d in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in d.bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{d.cls} {d.conf:.2f}", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if d.keypoints is not None:
            for x, y, c in d.keypoints:
                if c >= KEYPOINT_MIN_CONF:
                    cv2.circle(image, (int(round(x)), int(round(y))), 3, (0, 200, 255), -1)
    return image
