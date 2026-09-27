"""Regression tests for the confirmed findings of the steps 1-3 review."""

import shutil
import subprocess
import sys
import textwrap
import time

import cv2
import numpy as np
import pytest

from reachglass.config import CameraCfg, load_config
from reachglass.detect import ColorBlobDetector
from reachglass.geometry import camera_from_config, tello_camera
from reachglass.sources import StaticImageSource, TelloVideoSource, VideoFileSource


# ------------------------------------------------------------------ camera intrinsics (finding 1)
def test_default_camera_is_the_calibrated_stream_not_the_stills_fov():
    cam = camera_from_config(load_config().camera)
    assert (cam.width, cam.height, cam.fx, cam.fy) == (960, 720, 920.0, 920.0)
    assert cam.hfov_deg == pytest.approx(55.1, abs=0.3) and cam.vfov_deg == pytest.approx(42.7, abs=0.3)
    assert tello_camera().fx == 920.0 and tello_camera(480, 360).fx == 460.0
    assert tello_camera(f=None).fx == pytest.approx(683, abs=2)  # stills-FOV fallback still available


def test_camera_calibration_from_config():
    cfg = load_config(overrides={"camera": {"fx": 915, "fy": 910, "cx": 470, "cy": 350, "pitch_deg": -2}})
    cam = camera_from_config(cfg.camera)
    assert (cam.fx, cam.fy, cam.cx, cam.cy, cam.pitch_deg) == (915.0, 910.0, 470.0, 350.0, -2.0)
    half = cam.for_frame(480, 360)
    assert (half.fx, half.cx) == (457.5, 235.0)
    cam = camera_from_config(CameraCfg(fx=None, dfov_deg=66.2))
    assert cam.fx == pytest.approx(920, abs=1)


# ------------------------------------------------------------------ config validation (findings 3, 4, 13, 14)
def test_empty_yaml_section_keeps_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("follow:\n  # distance_m: 2.0\nexplore:\n")
    cfg = load_config(p)
    assert cfg.follow.distance_m == 1.0 and cfg.explore.max_vantage_points == 5


def test_wrong_types_fail_at_load_time_not_in_flight():
    with pytest.raises(TypeError, match="follow"):
        load_config(overrides={"follow": 3})
    with pytest.raises(TypeError, match="follow.distance_m"):
        load_config(overrides={"follow": {"distance_m": "far"}})
    with pytest.raises(TypeError, match="safety.max_rc"):
        load_config(overrides={"safety": {"max_rc": 50.5}})
    with pytest.raises(TypeError, match="mission.takeoff"):
        load_config(overrides={"mission": {"takeoff": "yes"}})
    with pytest.raises(TypeError, match="distance_m"):
        load_config(overrides={"follow": {"distance_m": True}})
    with pytest.raises(TypeError, match="object_heights_m"):
        load_config(overrides={"perception": {"object_heights_m": 3}})
    cfg = load_config(overrides={"follow": {"distance_m": "1.5"}, "safety": {"max_rc": 40.0}, "camera": {"fx": "1e3"}})
    assert cfg.follow.distance_m == 1.5 and cfg.safety.max_rc == 40 and isinstance(cfg.safety.max_rc, int)
    assert cfg.camera.fx == 1000.0
    assert load_config(overrides={"camera": {"fx": None}}).camera.fx is None


def test_kind_switch_without_params_gets_fresh_params_and_builds():
    from reachglass.detect import DETECTORS

    cfg = load_config(overrides={"perception": {"target_detector": {"kind": "null"}}})
    assert cfg.perception.target_detector.params == {}
    assert DETECTORS.build(cfg.perception.target_detector).detect(np.zeros((4, 4, 3), np.uint8)) == []
    cfg = load_config(overrides={"perception": {"target_detector": "null"}})  # shorthand
    assert cfg.perception.target_detector.kind == "null"
    cfg = load_config(overrides={"perception": {"context_detector": {"kind": None}}})
    assert cfg.perception.context_detector.kind == "" and DETECTORS.build(cfg.perception.context_detector) is None


def test_nested_dicts_deep_merge():
    cfg = load_config(overrides={"explore": {"semantic_weights": {"bottle": {"desk": 0.5}}}})
    w = cfg.explore.semantic_weights["bottle"]
    assert w["desk"] == 0.5 and w["dining table"] == 1.0  # sibling keys kept
    cfg = load_config(overrides={"perception": {"stride": {"search": {"context": 3}}}})
    assert cfg.perception.stride["search"] == {"person": 5, "target": 3, "context": 3}


# ------------------------------------------------------------------ colour blob (findings 6, 10)
def test_coloured_outline_or_ring_is_not_a_blob():
    blue = (167, 92, 52)  # the dummy's colour
    img = np.full((720, 960, 3), 120, np.uint8)
    cv2.rectangle(img, (300, 200), (420, 440), blue, 4)  # debug-overlay box in the dummy's colour
    cv2.circle(img, (700, 300), 60, blue, 5)  # ring
    assert ColorBlobDetector().detect(img) == []
    cv2.rectangle(img, (100, 500), (140, 590), blue, -1)  # a solid object is still found
    assert len(ColorBlobDetector().detect(img)) == 1


def test_hue_ranges_validated_and_wraparound_accepted():
    img = np.full((720, 960, 3), 120, np.uint8)
    cv2.rectangle(img, (100, 100), (160, 220), (60, 20, 220), -1)  # hue ~174
    cv2.rectangle(img, (500, 100), (560, 220), (30, 30, 210), -1)  # hue ~0
    wrap = ColorBlobDetector(hsv_ranges=[[172, 150, 140, 8, 255, 255]])
    assert len(wrap.detect(img)) == 2
    for bad in ([0, 0, 0, 200, 255, 255], [0, 200, 0, 10, 100, 255], [0, 0, 0, 10, 256, 255]):
        with pytest.raises(ValueError):
            ColorBlobDetector(hsv_ranges=[bad])


# ------------------------------------------------------------------ frames are read-only (finding 6)
def test_frames_are_read_only():
    f = StaticImageSource(np.zeros((10, 10, 3), np.uint8)).read()
    with pytest.raises(ValueError):
        f.image[0, 0] = 1
    with pytest.raises(cv2.error):
        cv2.rectangle(f.image, (0, 0), (5, 5), (0, 0, 255), 1)


# ------------------------------------------------------------------ sources restart / loop (findings 9, 16)
@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("v") / "v.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for i in range(20):
        w.write(np.full((48, 64, 3), 10 * i, np.uint8))
    w.release()
    return path


def test_file_source_restarts_cleanly(video):
    src = VideoFileSource(video, fps=200).start()
    src.wait_first(3)
    time.sleep(0.3)
    src.stop()
    assert src.read() is None
    seq_before = src._seq
    src.start()
    f = src.wait_first(3)
    assert f is not None and f.seq > seq_before and not src.finished
    src.stop()


def test_loop_on_unreadable_file_finishes_instead_of_spinning(tmp_path):
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"not a video")
    src = VideoFileSource(str(bad), loop=True)
    try:
        src.start()
    except RuntimeError:
        return  # refused at open: also fine
    t0 = time.time()
    while not src.finished and time.time() - t0 < 2:
        time.sleep(0.01)
    assert src.finished
    src.stop()


# ------------------------------------------------------------------ Tello UDP reader (findings 5, 7, 11, 12)
UDP_SCRIPT = textwrap.dedent("""
    import subprocess, sys, time
    sys.path.insert(0, {root!r})
    from reachglass.sources import TelloVideoSource
    port = {port}
    def sender():  # like the Tello: raw H.264 over UDP at 30 fps, 960x720
        return subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-re", "-f", "lavfi", "-i", "testsrc=size=960x720:rate=30",
                                 "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-g", "15",
                                 "-f", "h264", f"udp://127.0.0.1:{{port}}"])
    ff = sender()
    src = TelloVideoSource(url=f"udp://@0.0.0.0:{{port}}").start()
    first = src.wait_first(15)
    assert first is not None, "no frame"
    s0, t0 = src.read().seq, time.time()
    time.sleep(3.0)
    s1, t1 = src.read().seq, time.time()
    print("RATE", (s1 - s0) / (t1 - t0), flush=True)
    ff.kill(); ff.wait()
    time.sleep(5.0)                       # 5 s Wi-Fi gap: frames must flow again afterwards
    ff = sender()
    time.sleep(8.0)
    s2 = src.read().seq
    print("RECOVERED", s2 - s1, "RECONNECTS", src.reconnects, flush=True)
    ff.kill(); ff.wait()
    time.sleep(1.5)                       # stream stalled: reader is inside grab()
    t = time.time(); src.stop(); print("STOP", time.time() - t, flush=True)
""")


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_tello_reader_full_rate_and_safe_stop_on_stalled_udp_stream():
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = UDP_SCRIPT.format(root=root, port=11177)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    out = p.stdout
    assert p.returncode == 0, f"crashed (code {p.returncode}):\n{out}\n{p.stderr[-2000:]}"
    rate = float(out.split("RATE")[1].split()[0])
    recovered = int(out.split("RECOVERED")[1].split()[0])
    stop_s = float(out.split("STOP")[1].split()[0])
    assert rate > 25, f"delivered {rate:.1f} fps of a 30 fps stream"
    assert recovered > 30, out  # frames flow again after the gap (grab() waits it out, or the stream is reopened)
    assert stop_s < 6.0


def test_tello_source_default_cap_above_stream_rate():
    assert TelloVideoSource().fps >= 60
