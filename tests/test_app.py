"""Step 12b: dashboard, run log, CLI (sim mode) and the calibration tool."""

import json

import cv2
import numpy as np
import pytest

from reachglass.config import load_config
from reachglass.dashboard import BAR_H, CAM_H, CAM_W, MAP_W, render
from reachglass.mission import ScriptedInbox
from reachglass.runlog import RunLog
from reachglass.sim.runner import SimRunner


def test_dashboard_and_runlog_on_the_simulator(tmp_path):
    r = SimRunner(load_config(), inbox=ScriptedInbox([(4.0, "find my water bottle")]), seed=0)
    log = RunLog(tmp_path / "run.jsonl")
    for _ in range(int(8.0 / r.dt)):
        r.step()
        log.tick(r.ctx, r.mission)
    img = render(r.ctx, r.mission, fps=15.0, extra="test")
    assert img.shape == (CAM_H + BAR_H, CAM_W + MAP_W, 3) and img.dtype == np.uint8
    assert img[:CAM_H, :CAM_W].std() > 5  # camera panel has content
    log.close()
    lines = [json.loads(s) for s in (tmp_path / "run.jsonl").read_text().splitlines()]
    events = [x for x in lines if "event" in x]
    assert any(e["event"] == "state" and e["state"] == "FOLLOW" for e in events)
    assert any(e["event"] == "say" and "Looking for your bottle" in e["text"] for e in events)
    ticks = [x for x in lines if "pose" in x]
    assert len(ticks) > 30 and "tel" in ticks[-1]


@pytest.mark.slow
def test_cli_sim_headless_records_video(tmp_path):
    from reachglass.app import main

    video = tmp_path / "run.mp4"
    rc = main(["sim", "--headless", "--seconds", "12", "--query", "what's around me", "--at", "8",
               "--record", str(video), "--log", str(tmp_path / "log.jsonl"), "--udp", "0"])
    assert rc == 0 and video.exists() and video.stat().st_size > 10000
    cap = cv2.VideoCapture(str(video))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) > 100


def test_calibrate_camera_auto_recovers_focal_length(tmp_path, capsys):
    from reachglass.tools.calibrate_camera import main

    f_true, dist, h = 920.0, 2.0, 0.22
    img = np.full((720, 960, 3), 115, np.uint8)
    px_h, px_w = f_true * h / dist, f_true * 0.07 / dist
    cv2.rectangle(img, (int(480 - px_w / 2), int(360 - px_h / 2)), (int(480 + px_w / 2), int(360 + px_h / 2)), (167, 92, 52), -1)
    path = tmp_path / "dummy.png"
    cv2.imwrite(str(path), img)
    assert main([str(path), "--height", str(h), "--distance", str(dist), "--auto"]) == 0
    out = capsys.readouterr().out
    f = float(out.split("at 960x720")[0].split("(")[-1].split()[0])
    assert f == pytest.approx(f_true, rel=0.03)
