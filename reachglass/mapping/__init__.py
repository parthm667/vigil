"""Mapping: odometry, coverage grid, semantic memory, free space."""

from ..registry import Registry
from .freespace import FreeSpace, FreeSpaceEstimator, NullFreeSpace
from .grid import Grid
from .odometry import Odometry
from .semantic import MemObject, SemanticMemory, Sighting

FREESPACE = Registry("free-space estimator")
FREESPACE.register("null")(NullFreeSpace)


@FREESPACE.register("depth")
def _depth(**params):
    from .freespace import DepthFreeSpace  # imports transformers only when used

    return DepthFreeSpace(**params)


__all__ = ["FREESPACE", "FreeSpace", "FreeSpaceEstimator", "NullFreeSpace", "Grid", "Odometry", "SemanticMemory",
           "MemObject", "Sighting"]
