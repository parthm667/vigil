"""Step 2: frame sources."""

import os
import shutil
import subprocess
import time

import cv2
import numpy as np
import pytest

from reachglass.config import ComponentSpec
from reachglass.registry import Registry
from reachglass.sources import SOURCES, FrameSource, StaticImageSource, SteppedVideoSource, TelloVideoSource, VideoFileSource

N_FRAMES = 40


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """A 30 fps MJPG video whose frame i is a flat grey of level 5*i (so order is checkable)."""
    path = str(tmp_path_factory.mktemp("vid") / "ramp.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 120))
    assert w.isOpened()
    for i in range(N_FRAMES):
        w.write(np.full((120, 160, 3), 5 * i + 10, np.uint8))
    w.release()
    return path


def level(frame) -> int:
    return int(round((float(frame.image.mean()) - 10) / 5))


def test_stepped_source_is_ordered_and_finishes(video):
    src = SteppedVideoSource(video)
    got = []
    while (f := src.read()) is not None:
        got.append((f.seq, level(f)))
        assert f.image.shape == (120, 160, 3) and f.image.dtype == np.uint8
    assert [s for s, _ in got] == list(range(1, N_FRAMES + 1))
    assert [lv for _, lv in got] == list(range(N_FRAMES))
    assert src.finished
    src.stop()


def test_file_source_plays_in_real_time_and_serves_newest(video):
    with VideoFileSource(video) as src:
        first = src.wait_first(3.0)
        assert first is not None
        time.sleep(0.5)
        f = src.read()
        # ~15 frames in 0.5 s at 30 fps: newest frame, not the next queued one
        assert f.seq >= first.seq + 8
        assert abs(level(f) - (f.seq - 1)) <= 1
        t0 = time.time()
        while not src.finished and time.time() - t0 < 3:
            time.sleep(0.05)
        assert src.finished and src.read().seq == N_FRAMES


def test_file_source_loops(video):
    with VideoFileSource(video, loop=True, fps=200) as src:
        src.wait_first(3.0)
        time.sleep(0.6)  # 120 frames at 200 fps > 40 frames in the file
        assert src.read().seq > N_FRAMES and not src.finished


def test_team_videostream_reader_via_tello_source(video):
    # the Tello source wraps the team's VideoStream; point it at a file to exercise it without a drone
    import reachglass.sources.video_stream  # noqa: F401

    assert "nobuffer" in os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
    src = TelloVideoSource(url=video, fps=60, backend="opencv").start()
    try:
        f = src.wait_first(5.0)
        assert f is not None and f.source == "tello" and f.seq >= 1
        seqs = [f.seq]
        t0 = time.time()
        while time.time() - t0 < 0.5:
            g = src.read()
            if g.seq != seqs[-1]:
                seqs.append(g.seq)
            time.sleep(0.002)
        assert seqs == sorted(seqs)  # never goes backwards
        assert abs(f.t - time.time()) < 5  # stamped with wall-clock time
    finally:
        src.stop()
    assert src.read() is None


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize("backend", ["pyav", "opencv"])
def test_tello_source_waits_for_a_stream_that_starts_after_the_open_timeout(backend):
    # the "can't find camera" bugs: video starting > 5 s after start() raised; keyframes 5 s apart never decoded
    port = 11178
    src = TelloVideoSource(url=f"udp://@0.0.0.0:{port}", backend=backend).start()  # must not raise
    time.sleep(7.0)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-re", "-f", "lavfi", "-i", "testsrc=size=960x720:rate=30",
                           "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-g", "150",
                           "-f", "h264", f"udp://127.0.0.1:{port}?pkt_size=1460"])
    try:
        f = src.wait_first(15.0)
        assert f is not None and f.image.shape == (720, 960, 3)
    finally:
        ff.kill()
        ff.wait()
        src.stop()
    assert src.read() is None


def test_tello_source_before_start_returns_none():
    assert TelloVideoSource().read() is None


def test_static_and_wait_first_timeout():
    s = StaticImageSource(np.zeros((10, 10, 3), np.uint8))
    assert s.read().seq == 1 and s.read().seq == 2

    class Never(FrameSource):
        def read(self):
            return None

    t0 = time.time()
    assert Never().wait_first(0.2) is None and time.time() - t0 >= 0.19


def test_registry(video):
    assert {"tello", "webcam", "file", "stepped"} <= set(SOURCES.names())
    src = SOURCES.build(ComponentSpec("stepped", {"path": video}))
    assert isinstance(src, SteppedVideoSource) and src.read().seq == 1
    assert SOURCES.build(None) is None and SOURCES.build("") is None and SOURCES.build(ComponentSpec("", {})) is None
    with pytest.raises(KeyError, match="unknown frame source 'nope'"):
        SOURCES.build("nope")
    r = Registry("thing")
    r.register("a")(lambda **k: k)
    with pytest.raises(ValueError):
        r.register("a")(lambda: 0)
    assert r.build(ComponentSpec("a", {"x": 1}), y=2) == {"x": 1, "y": 2}
