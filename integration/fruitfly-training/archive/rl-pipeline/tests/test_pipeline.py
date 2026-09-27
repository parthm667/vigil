"""Fast checks for the simulator, controllers, parameter mapping, trainer and publisher (no brain file needed)."""

import json
import math
import subprocess
from argparse import Namespace

import numpy as np

from flyfollow.config import load_config
from flyfollow.pilot.pid import PID_HAND, PIDController
from flyfollow.rl.backends import SerialBackend
from flyfollow.rl.env import run_episode
from flyfollow.rl.params import decode, encode, specs_for_arm
from flyfollow.rl.publish import Publisher
from flyfollow.rl.train import Trainer, parse_args
from flyfollow.sim.camera import project
from flyfollow.sim.drone_model import DroneModel
from flyfollow.sim.scenario import sample_scenario


def test_scenario_is_deterministic():
    cfg = load_config()
    a = sample_scenario(cfg, 5, "follow")
    b = sample_scenario(cfg, 5, "follow")
    assert a == b
    assert a["person_speed_max"] <= a["drone_v_max"]


def test_projection_signs():
    cfg = load_config()
    s = sample_scenario(cfg, 1, "follow")
    drone = DroneModel(s, 0.05, np.random.default_rng(0))
    drone.z = 1.5
    right = project(drone, 2.0, -0.5, 1.5, 0.23, 900, 900, 960, 720)
    left = project(drone, 2.0, 0.5, 1.5, 0.23, 900, 900, 960, 720)
    assert right["cx"] > 480 and left["cx"] < 480
    assert right["bearing"] > 0 and left["bearing"] < 0
    assert project(drone, -2.0, 0.0, 1.5, 0.23, 900, 900, 960, 720) is None


def test_positive_yaw_stick_turns_right():
    cfg = load_config()
    s = sample_scenario(cfg, 1, "follow")
    drone = DroneModel(s, 0.05, np.random.default_rng(0))
    for i in range(40):
        drone.step(0.0, 40.0, 0.0)
    assert drone.psi < 0  # clockwise seen from above


def test_pid_follows_reasonably():
    cfg = load_config()
    in_band = []
    for seed in range(10):
        r = run_episode(PIDController(PID_HAND), sample_scenario(cfg, 100 + seed, "follow"), cfg)
        in_band.append(r["in_band_frac"])
        assert math.isfinite(r["return"])
    assert np.mean(in_band) > 0.2


def test_param_round_trip():
    cfg = load_config()
    for arm in ("fly", "nobrain", "pid"):
        specs = specs_for_arm(arm, cfg["brain"]["dn_types"])
        x = np.random.default_rng(0).random(len(specs))
        params = decode(specs, x)
        assert np.allclose(encode(specs, params), x, atol=1e-9)
    assert len(specs_for_arm("fly", cfg["brain"]["dn_types"])) == 47


def test_trainer_smoke(tmp_path):
    args = parse_args(["--arm", "nobrain", "--backend", "serial", "--generations", "2", "--popsize", "4",
                       "--follow-s", "5", "--run-name", "pytest-smoke"])
    trainer = Trainer(args, backend=SerialBackend())
    trainer.run()
    latest = json.loads((trainer.run_dir / "latest.json").read_text())
    assert latest["generation"] == 2
    assert len(latest["mean_unit"]) == 47
    # the published mean must be where CMA-ES actually samples, not the clipped internal mean
    import pickle

    with open(trainer.run_dir / "state.pkl", "rb") as f:
        es = pickle.load(f)["es"]
    mean_unit = np.array(latest["mean_unit"])
    assert np.allclose(mean_unit, np.clip(es.result.xfavorite, 0, 1))
    # away from the bounds, samples average to the published mean (at a bound, folding pulls them inward)
    samples = np.array(es.ask(number=2000))
    inside = (mean_unit > 0.25) & (mean_unit < 0.75)
    assert np.abs(samples.mean(axis=0)[inside] - mean_unit[inside]).max() < 0.05


def test_publisher_survives_failed_push(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "remote", "add", "origin", "https://example.invalid/nope.git"], cwd=tmp_path, check=True)
    (tmp_path / "unrelated.py").write_text("half written code")
    pub = Publisher(tmp_path)
    ok = pub.publish("pid", {"arm": "pid", "generation": 1, "sigma": 0.1, "latest_train_score": -1.0,
                             "run_name": "t", "backend": "local", "updated_utc": "now"}, "results: test")
    assert ok is False
    committed = subprocess.run(["git", "show", "--name-only", "--format=", "HEAD"], cwd=tmp_path,
                               capture_output=True, text=True).stdout.split()
    assert sorted(committed) == ["HANDOFF.md", "checkpoints/pid/latest.json"]
