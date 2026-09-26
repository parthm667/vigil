"""Detector interface. Anything that turns a BGR image into Detections implements it."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import Detection


class Detector(ABC):
    name = "detector"

    @property
    @abstractmethod
    def classes(self) -> list[str]:
        """Class names this detector can report (the vocabulary the query parser may choose from)."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detections in pixel coordinates of `image`, most confident first."""


class NullDetector(Detector):
    """Detects nothing. Used to disable a slot without special cases elsewhere."""

    name = "null"

    @property
    def classes(self) -> list[str]:
        return []

    def detect(self, image: np.ndarray) -> list[Detection]:
        return []
