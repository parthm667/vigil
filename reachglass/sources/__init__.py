"""Frame sources. Build one by name: SOURCES.build("tello") / SOURCES.build(ComponentSpec("file", {"path": ...}))."""

from ..registry import Registry
from .base import FrameSource
from .opencv_sources import (
    TELLO_UDP_URL,
    StaticImageSource,
    SteppedVideoSource,
    TelloVideoSource,
    VideoFileSource,
    WebcamSource,
)

SOURCES = Registry("frame source")
SOURCES.register("tello")(TelloVideoSource)
SOURCES.register("webcam")(WebcamSource)
SOURCES.register("file")(VideoFileSource)
SOURCES.register("stepped")(SteppedVideoSource)

__all__ = [
    "SOURCES", "FrameSource", "TelloVideoSource", "WebcamSource", "VideoFileSource", "SteppedVideoSource",
    "StaticImageSource", "TELLO_UDP_URL",
]
