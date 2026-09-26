"""What a detector returns, and the interface every detector implements.

Copied from reachglass/types.py (Detection) and reachglass/detect/base.py (Detector) so this package stands
alone. The fields match reachglass's Detection, but it is a separate class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    cls: str
    conf: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    keypoints: np.ndarray | None = None  # (K, 3): x, y, confidence; COCO-17 order for people
    track_id: int | None = None
    source: str = ""  # detector that produced it

    @property
    def cx(self) -> float:
        return 0.5 * (self.bbox[0] + self.bbox[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.bbox[1] + self.bbox[3])

    @property
    def w(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def h(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.w * self.h


class Detector(ABC):
    name = "detector"

    @property
    @abstractmethod
    def classes(self) -> list[str]:
        """Class names this detector can report (the vocabulary the query parser may choose from)."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detections in pixel coordinates of `image`, most confident first."""
