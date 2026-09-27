"""Fast tests for the trainer (plan item 7) with fake episodes: no env, controllers or brains needed."""

from __future__ import annotations

import json
import math
import os
import pickle

import numpy as np
import pytest

from flyfollow.interfaces import REPO_ROOT, SELECTION_SEEDS, TEST_SEEDS, TRAIN_SEED_BASE
from flyfollow.rl import cma_train as ct
from flyfollow.rl import evaluate as ev

DIM = 6
TARGET = np.linspace(0.2, 0.8, DIM)


# --------------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------------


def fake_runner(item: dict) -> dict:
    """Deterministic fake episode: PID-HAND returns about -10; candidates are scored by distance to TARGET.

    Seeds ending in 13 raise, to exercise error capture.
    """
    seed, kind = int(item["seed"]), item["kind"]
    if seed % 100 == 13:
        raise RuntimeError(f"boom on seed {seed}")
    noise = 1.0 + 0.1 * ((seed * 7919) % 11) / 11.0
    if item["arm"] in ("PID-HAND", "PID-CMA") and item["x"] is None:
        ret = -10.0 * noise
    else:
        x = np.asarray(item["x"] if item["x"] is not None else np.full(DIM, 0.5))
        ret = -(2.0 + 40.0 * float(np.sum((x - TARGET[: x.size]) ** 2))) * noise * (1.5 if kind == "approach" else 1.0)
    lesioned = bool(item.get("lesion"))
    if lesioned:
        ret *= 3.0
    band = 0.1 if lesioned else 0.5 + 0.01 * (seed % 7)
    metrics = {"frac_in_band": band, "min_dist": 1.2 + 0.01 * (seed % 5), "loss_events_per_min": 1.0, "rms_bearing_err_deg": 30.0 if lesioned else 8.0,
               "frac_in_view": 0.4 if lesioned else 0.95}
    if kind == "approach":
        metrics.update(success=float(seed % 2 == 0), time_to_standoff=9.0, overshoot=-0.1, floor_object=float(seed % 3 == 0))
    out = {
        "seed": seed,
        "kind": kind,
        "profile": item.get("profile", "train"),
        "ret": ret,
        "terms": {"standoff": 0.6 * ret, "bearing": 0.4 * ret},
        "metrics": metrics,
        "n_ticks": 200,
        "wall_s": 0.001,
        "brain_s": 0.0,
        "trace": None,
    }
    if item.get("audit"):
        out["channel_means"] = {"DNa02_L": 10.0, "DNa02_R": 12.0}
        out["bias_audit"] = {"yaw_drive_abs": 0.5, "b_yaw_abs": 0.1, "yaw_bias_ratio": 0.2, "fwd_drive_abs": 0.0, "b_fwd_abs": 0.0, "fwd_bias_ratio": 0.0}
    return out


@pytest.fixture
def fake_episodes(monkeypatch):
    monkeypatch.setattr(ev, "EPISODE_RUNNER", fake_runner)


class FakeBackend:
    """In-process backend with the same contract as LocalBackend/ModalBackend; fixed cost per episode."""

    name = "fake"

    def __init__(self, usd_per_item: float = 0.001):
        self.usd_per_item = usd_per_item
        self.calls = 0

    def evaluate(self, items):
        self.calls += 1
        res = ev.evaluate_items(items, processes=1)
        return res, {"container_s": 0.01 * len(items), "cost_usd": self.usd_per_item * len(items), "n_chunks": 1, "cost_basis": "fake"}


def small_cfg(**over) -> dict:
    o = {
        "cma": {"popsize": 6, "sigma0": 0.2},
        "episodes": {"k_follow": 2, "k_approach": 1, "episode_s": None},
        "selection": {"every": 2, "at_start": True, "at_end": True, "n_follow": 4, "n_approach": 2},
        "budget": {"per_run_usd": None, "total_usd": None},
    }
    return ct.load_config(overrides=ev.deep_merge(o, over))


def make_trainer(tmp_path, cfg=None, backend=None, arm="NOBRAIN", seed=1, name="run", **kw):
    return ct.Trainer(
        arm, seed, cfg or small_cfg(), backend or FakeBackend(), run_dir=tmp_path / name, x0=np.full(DIM, 0.5), log=lambda s: None, **kw
    )


def _strip_timing(rec: dict) -> dict:
    drop = {"time", "wall_s", "item_wall_s_sum", "episode_wall_s_sum"}
    return {k: v for k, v in rec.items() if k not in drop}


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------------------------
# Fitness normalization (plan 4.5)
# --------------------------------------------------------------------------------------------


def test_normalize_return_divides_by_abs_pid_with_floor():
    assert ct.normalize_return(-20.0, -10.0) == pytest.approx(-2.0)
    assert ct.normalize_return(-20.0, 10.0) == pytest.approx(-2.0)  # absolute value of the PID return
    assert ct.normalize_return(-3.0, -0.2) == pytest.approx(-3.0)  # floor 1
    assert ct.normalize_return(5.0, 0.0) == pytest.approx(5.0)
    assert ct.normalize_return(-3.0, None) == pytest.approx(-3.0)  # failed denominator: floor
    assert ct.normalize_return(None, -10.0, error_score=-20.0) == -20.0
    assert ct.normalize_return(float("nan"), -10.0, error_score=-7.0) == -7.0


def test_fitness_weights_follow_and_approach_equally():
    scores = [-1.0, -1.0, -1.0, -1.0, -1.0, -4.0, -4.0, -4.0]  # 5 follow, 3 approach
    kinds = ["follow"] * 5 + ["approach"] * 3
    assert ct.fitness(scores, kinds) == pytest.approx(0.5 * -1.0 + 0.5 * -4.0)
    assert ct.fitness([-2.0, -3.0], ["follow", "follow"]) == pytest.approx(-2.5)  # no approach: follow only
    assert math.isnan(ct.fitness([], []))


# --------------------------------------------------------------------------------------------
# Seeds (plan 4.4)
# --------------------------------------------------------------------------------------------


def test_train_seeds_deterministic_disjoint_and_split():
    a = ct.train_seeds(1, 7, 5, 3)
    assert a == ct.train_seeds(1, 7, 5, 3)
    assert [k for _, k in a] == ["follow"] * 5 + ["approach"] * 3
    seen = set()
    for run_seed in (1, 2, 3):
        for gen in range(1, 151):
            s = [x for x, _ in ct.train_seeds(run_seed, gen, 5, 3)]
            assert len(set(s)) == 8
            assert min(s) >= TRAIN_SEED_BASE
            assert not seen & set(s), "training seeds repeat across generations or run seeds"
            seen |= set(s)
    assert not seen & set(SELECTION_SEEDS)
    assert not seen & set(TEST_SEEDS)
    with pytest.raises(ValueError):
        ct.train_seeds(1, 0, 5, 3)


def test_selection_and_test_plans_fixed_kinds():
    cfg = ct.load_config()
    sel = ct.selection_plan(cfg)
    assert [s for s, _ in sel] == list(SELECTION_SEEDS)
    assert [k for _, k in sel] == ["follow"] * 40 + ["approach"] * 24
    test = ct.test_plan(cfg)
    assert [s for s, _ in test] == list(TEST_SEEDS)
    assert sum(k == "follow" for _, k in test) == 125
    assert set(SELECTION_SEEDS).isdisjoint(TEST_SEEDS)


def test_brain_for_arms():
    cfg = ct.load_config(overrides={"brain": {"core": 2}})
    assert ct.brain_for("FLY-CMA", 3, cfg) == "pursuit_core2.npz"
    assert ct.brain_for("FLY-HAND", 1, cfg) == "pursuit_core2.npz"
    assert ct.brain_for("FLY-SHUF", 3, cfg) == "pursuit_core2_shuf3.npz"
    assert ct.brain_for("NOBRAIN", 1, cfg) is None
    assert ct.brain_for("PID-CMA", 1, cfg) is None


# --------------------------------------------------------------------------------------------
# Chunking and the Modal backend contract
# --------------------------------------------------------------------------------------------


def test_make_chunks_preserves_order():
    items = list(range(37))
    chunks = ct.make_chunks(items, 16)
    assert [len(c) for c in chunks] == [16, 16, 5]
    assert [x for c in chunks for x in c] == items
    assert ct.make_chunks([], 16) == []
    assert ct.make_chunks([1, 2], 0) == [[1], [2]]


class FakeModalFn:
    def __init__(self, fail_chunk: int | None = None):
        self.fail_chunk = fail_chunk
        self.chunks = []

    def map(self, chunks, return_exceptions=False, order_outputs=True):
        for i, c in enumerate(chunks):
            self.chunks.append(c)
            if i == self.fail_chunk:
                yield RuntimeError("container died")
            else:
                yield ev.evaluate_chunk_local(c, workers=1)


def test_chunk_size_by_arm():
    cfg = small_cfg(chunking={"items_per_chunk": 16, "by_arm": {"NOBRAIN": 96}})
    assert ct.chunk_size(cfg, "NOBRAIN") == 96 and ct.chunk_size(cfg, "FLY-CMA") == 16 and ct.chunk_size(cfg) == 16


def test_modal_backend_chunks_and_failed_chunk(fake_episodes):
    cfg = small_cfg(chunking={"items_per_chunk": 3})
    items = [ev.make_item("NOBRAIN", 10_000 + i, "follow", x=np.full(DIM, 0.5), tag={"k": i}) for i in range(8)]
    fn = FakeModalFn(fail_chunk=1)
    res, stats = ct.ModalBackend(cfg, fn).evaluate(items)
    assert [len(c) for c in fn.chunks] == [3, 3, 2]
    assert [r["tag"]["k"] for r in res] == list(range(8))  # order preserved
    assert [r["ok"] for r in res] == [True] * 3 + [False] * 3 + [True] * 2
    assert "container died" in res[3]["error"]
    assert stats["n_chunks"] == 3 and stats["chunk_errors"] == 1
    assert stats["cost_usd"] >= 0


# --------------------------------------------------------------------------------------------
# evaluate_item error capture
# --------------------------------------------------------------------------------------------


def test_evaluate_item_ok_and_echo(fake_episodes):
    it = ev.make_item("NOBRAIN", 10_001, "follow", x=np.full(DIM, 0.5), tag={"role": "cand", "i": 3})
    r = ev.evaluate_item(it)
    assert r["ok"] and r["error"] is None
    assert r["tag"] == {"role": "cand", "i": 3} and r["arm"] == "NOBRAIN" and r["seed"] == 10_001
    assert r["ret"] < 0 and "standoff" in r["terms"]


def test_evaluate_item_captures_errors(fake_episodes, monkeypatch):
    r = ev.evaluate_item(ev.make_item("NOBRAIN", 10_013, "follow", x=np.full(DIM, 0.5), tag="t"))
    assert r["ok"] is False and r["ret"] is None and "boom" in r["error"] and r["tag"] == "t"
    assert "RuntimeError" in r["traceback"]

    monkeypatch.setattr(ev, "EPISODE_RUNNER", lambda item: {"ret": float("nan")})
    r = ev.evaluate_item(ev.make_item("NOBRAIN", 1, "follow"))
    assert r["ok"] is False and "non-finite" in r["error"]

    def exits(item):
        raise SystemExit(3)

    monkeypatch.setattr(ev, "EPISODE_RUNNER", exits)
    assert ev.evaluate_item(ev.make_item("NOBRAIN", 1, "follow"))["ok"] is False


def test_evaluate_items_pool_errors_do_not_raise():
    """Real worker processes: without A's env module every item fails, and each comes back as an error in order."""
    items = [ev.make_item("NOBRAIN", 10_000 + i, "follow", tag=i, brain=None) for i in range(4)]
    items.append(ev.make_item("NOBRAIN", 1, "follow", tag=4, x=[0.5] * 3))
    res = ev.evaluate_items(items, processes=2)
    try:
        assert [r["tag"] for r in res] == [0, 1, 2, 3, 4]
        assert all(isinstance(r["ok"], bool) for r in res)
    finally:
        ev.shutdown_pool()


def test_apply_overrides_episode_s():
    cfg = {"episode": {"follow_s": 60.0, "approach_s": 25.0, "preroll_s": 1.0}, "walker": {"follow_s": 99.0}, "dt": 0.05}
    out = ev.apply_overrides(cfg, {"episode_s": 10, "dt": 0.1})
    assert out["episode"] == {"follow_s": 10.0, "approach_s": 10.0, "preroll_s": 1.0}
    assert out["walker"]["follow_s"] == 99.0 and out["dt"] == 0.1 and "episode_s" not in out
    assert ev.apply_overrides(cfg, {"episode_s": 40})["episode"]["approach_s"] == 25.0  # only ever shortens
    assert cfg["episode"]["follow_s"] == 60.0  # input untouched


# --------------------------------------------------------------------------------------------
# The driver: learning, logs, checkpoint and resume
# --------------------------------------------------------------------------------------------


def test_trainer_runs_logs_and_improves(tmp_path, fake_episodes):
    t = make_trainer(tmp_path)
    summary = t.run(8)
    assert summary["gen"] == 8 and summary["stopped"] == "done"
    logs = _read_jsonl(t.log_path)
    assert [r["gen"] for r in logs] == list(range(1, 9))
    for r in logs:
        assert math.isfinite(r["fit_best"]) and r["fit_best"] >= r["fit_median"] >= r["fit_worst"]
        assert set(r["terms_mean"]) == {"standoff", "bearing"}
        assert r["n_items"] == 6 * 3 + r["n_pid"]
        assert r["cost_usd_total"] > 0
    assert logs[0]["n_pid"] == 3 and logs[1]["n_pid"] == 3  # fresh seeds each generation
    sel = _read_jsonl(t.sel_path)
    assert [r["gen"] for r in sel] == [0, 2, 4, 6, 8]
    best = json.loads(t.best_path.read_text())
    assert best["gen"] == max(sel, key=lambda r: r["sel_fitness"])["gen"]
    assert len(best["x"]) == DIM
    assert logs[-1]["fit_mean"] > logs[0]["fit_mean"]  # the fake problem is easy


def test_resume_is_identical_continuation(tmp_path, fake_episodes):
    a = make_trainer(tmp_path, name="a")
    a.run(5)
    b1 = make_trainer(tmp_path, name="b")
    b1.run(5, stop_after=1)  # "killed" after generation 1
    assert b1.state["gen"] == 1
    b2 = make_trainer(tmp_path, name="b")  # a fresh driver process
    assert b2.load_or_init() is True
    b2.run(5)
    la, lb = _read_jsonl(a.log_path), _read_jsonl(b2.log_path)
    assert [_strip_timing(r) for r in la] == [_strip_timing(r) for r in lb]
    assert [_strip_timing(r) for r in _read_jsonl(a.sel_path)] == [_strip_timing(r) for r in _read_jsonl(b2.sel_path)]
    np.testing.assert_array_equal(a.state["es"].mean, b2.state["es"].mean)
    assert a.state["es"].sigma == b2.state["es"].sigma


def test_resume_drops_log_lines_past_the_checkpoint(tmp_path, fake_episodes):
    t = make_trainer(tmp_path)
    t.run(2)
    with open(t.log_path, "a") as f:  # a crash after logging gen 3 but before its checkpoint
        f.write(json.dumps({"gen": 3, "fit_best": 0.0}) + "\n")
    t2 = make_trainer(tmp_path)
    t2.load_or_init()
    assert [r["gen"] for r in _read_jsonl(t2.log_path)] == [1, 2]
    t2.run(3)
    assert [r["gen"] for r in _read_jsonl(t2.log_path)] == [1, 2, 3]


def test_resume_refuses_changed_config(tmp_path, fake_episodes):
    make_trainer(tmp_path).run(1)
    with pytest.raises(RuntimeError, match="popsize"):
        make_trainer(tmp_path, cfg=small_cfg(cma={"popsize": 8})).load_or_init()


def test_errors_are_penalized_and_logged(tmp_path, fake_episodes, monkeypatch):
    # K = 20 so each generation's seeds include one ending in 13 (fake_runner raises there).
    cfg = small_cfg(episodes={"k_follow": 14, "k_approach": 6}, selection={"every": 0, "at_start": False, "at_end": False})
    t = make_trainer(tmp_path, cfg=cfg)
    t.run(1)
    rec = _read_jsonl(t.log_path)[0]
    assert rec["n_errors"] == 6 + 1  # one seed x 6 candidates, plus its PID-HAND denominator
    assert rec["n_pid_errors"] == 1
    assert any("boom" in e for e in rec["error_examples"])
    errs = _read_jsonl(t.err_path)
    assert len(errs) == 7 and all(e["seed"] % 100 == 13 for e in errs)
    # the failed episode scores error_score, so every candidate's fitness is below the no-error value
    assert rec["fit_best"] < 0


def test_budget_stop(tmp_path, fake_episodes):
    t = make_trainer(tmp_path, backend=FakeBackend(usd_per_item=0.01), budget_usd=0.5)
    summary = t.run(50)
    assert summary["stopped"] == "budget"
    assert summary["cost_usd_total"] <= 0.5
    assert summary["gen"] < 50
    assert _read_jsonl(t.log_path)[-1]["event"] == "budget stop"
    t2 = make_trainer(tmp_path, backend=FakeBackend(usd_per_item=0.01), budget_usd=0.5)
    assert t2.run(50)["gen"] == summary["gen"]  # stays stopped on restart


def test_state_pickle_is_self_contained(tmp_path, fake_episodes):
    t = make_trainer(tmp_path)
    t.run(1)
    st = pickle.loads(t.state_path.read_bytes())
    assert st["gen"] == 1 and st["dim"] == DIM and isinstance(st["es"].opts["randn"], ct.SeededRandn)
    assert all(isinstance(k, str) for k in st["pid_cache"])
    assert b"__main__" not in t.state_path.read_bytes()


def test_cli_checkpoint_loads_outside_main(tmp_path):
    """`python -m flyfollow.rl.cma_train` must write a state.pkl that other modules (Modal train) can load."""
    import subprocess
    import sys

    env = dict(os.environ, FLYFOLLOW_DATA=str(tmp_path))
    cmd = [sys.executable, "-m", "flyfollow.rl.cma_train", "--arm", "PID-CMA", "--seed", "1", "--gens", "1", "--pop", "4",
           "--k-follow", "1", "--k-approach", "0", "--episode-s", "2", "--select-every", "0", "--tag", "clitest", "--processes", "1"]
    subprocess.run(cmd, check=True, env=env, cwd=REPO_ROOT, capture_output=True, timeout=120)
    raw = (tmp_path / "runs" / "PID-CMA_s1_clitest" / "state.pkl").read_bytes()
    assert b"__main__" not in raw
    assert pickle.loads(raw)["gen"] == 1


# --------------------------------------------------------------------------------------------
# Budget, projection and the Modal app
# --------------------------------------------------------------------------------------------


def test_budget_shares_and_projection():
    cfg = ct.load_config()
    runs = ct.planned_runs(cfg)
    assert [r["seed"] for r in runs[:4]] == [1, 1, 1, 1]  # seed 1 of every arm first
    share = cfg["budget"]["share"]
    total = sum(ct.run_budget_usd(cfg, r["arm"]) for r in runs)
    planned_share = sum(share[a] for a in cfg["runs"]["arms"])
    assert total == pytest.approx(cfg["budget"]["total_usd"] * planned_share)
    assert planned_share + sum(v for a, v in share.items() if a not in cfg["runs"]["arms"]) <= 1.0 + 1e-9
    p = ct.project(cfg)
    assert p["total_usd"] > 0 and p["wall_h"] >= p["wall_h_container_cap"]
    measured = {"NOBRAIN-YAW": {"container_s_per_gen": 10.0, "chunk_wall_s": 2.0}}
    assert ct.project(cfg, measured)["per_arm"]["NOBRAIN-YAW"]["source"] == "smoke"
    assert ct.project(cfg, {"NOBRAIN": measured["NOBRAIN-YAW"]})["per_arm"]["NOBRAIN-YAW"]["source"] == "smoke of NOBRAIN"
    # Starter plan: eval containers + one driver per run + the orchestrator stay under 100.
    assert cfg["modal"]["eval"]["max_containers"] + len(runs) + 1 <= 100


class FakeVolume:
    def __init__(self):
        self.commits = 0
        self.reloads = 0

    def commit(self):
        self.commits += 1

    def reload(self):
        self.reloads += 1


def test_modal_train_body_commits_and_resumes(tmp_path, fake_episodes, monkeypatch):
    """Run the Modal `train` function body locally: fake Volume, fake evaluate_chunk, real Trainer."""
    from flyfollow.rl import modal_app

    monkeypatch.setenv("FLYFOLLOW_DATA", str(tmp_path))
    vol = FakeVolume()
    monkeypatch.setattr(modal_app, "volume", vol)
    monkeypatch.setattr(modal_app, "evaluate_chunk", FakeModalFn())
    monkeypatch.setattr(modal_app.ct.Trainer, "_initial_x", lambda self: np.full(DIM, 0.5))
    over = {"cma": {"popsize": 4}, "episodes": {"k_follow": 2, "k_approach": 1}, "selection": {"every": 0, "at_start": False, "at_end": False}}
    s1 = modal_app.train.local("NOBRAIN", 1, 2, "t", over)
    assert s1["gen"] == 2 and s1["stopped"] == "done" and s1["cost_usd_total"] > 0
    assert vol.reloads == 1 and vol.commits >= 2 + 2  # lease, one per generation, final
    run_dir = tmp_path / "runs" / "NOBRAIN_s1_t"
    assert not (run_dir / "lease.json").exists()
    s2 = modal_app.train.local("NOBRAIN", 1, 3, "t", over)  # a retry / preemption restart resumes
    assert s2["gen"] == 3
    assert [r["gen"] for r in _read_jsonl(run_dir / "log.jsonl")] == [1, 2, 3]
    # a second live driver on the same run is refused
    (run_dir / "lease.json").write_text(json.dumps({"call_id": "fc-other", "heartbeat": __import__("time").time()}))
    monkeypatch.setattr(modal_app.modal, "current_function_call_id", lambda: "fc-me")
    with pytest.raises(RuntimeError, match="being trained"):
        modal_app.train.local("NOBRAIN", 1, 4, "t", over)


def test_modal_app_imports_without_network():
    from flyfollow.rl import modal_app

    assert modal_app.app.name == "flyfollow"
    assert modal_app.EV["max_containers"] == ct.load_config()["modal"]["eval"]["max_containers"]
    assert modal_app.TR["timeout_s"] <= 86400


# --------------------------------------------------------------------------------------------
# Yaw-only arms (the fly steers, the PID sets forward): lead decision 2026-09-26
# --------------------------------------------------------------------------------------------

YAW_TRAINED = ("FLY-YAW", "FLY-SHUF-YAW", "NOBRAIN-YAW")


def test_yaw_arms_are_known_and_trainable():
    from flyfollow.interfaces import ARMS, TRAINED_ARMS, YAW_ONLY_ARMS

    assert set(YAW_ONLY_ARMS) <= set(ARMS)
    assert set(YAW_TRAINED) <= set(TRAINED_ARMS)
    assert {"FLY-YAW", "FLY-SHUF-YAW", "FLY-YAW-HAND"} <= set(ct.FLY_ARMS)
    assert "NOBRAIN-YAW" not in ct.FLY_ARMS


def test_brain_for_yaw_arms():
    cfg = ct.load_config(overrides={"brain": {"core": 1}})
    assert ct.brain_for("FLY-YAW", 2, cfg) == "pursuit_core1.npz"
    assert ct.brain_for("FLY-YAW-HAND", 1, cfg) == "pursuit_core1.npz"
    for seed in (1, 2, 3):
        assert ct.brain_for("FLY-SHUF-YAW", seed, cfg) == f"pursuit_core1_shuf{seed}.npz"
        assert ct.brain_for("FLY-SHUF-YAW", seed, cfg) == ct.brain_for("FLY-SHUF", seed, cfg)
    assert ct.brain_for("NOBRAIN-YAW", 1, cfg) is None


def test_default_run_list_is_the_yaw_comparison():
    cfg = ct.load_config()
    runs = ct.planned_runs(cfg)
    assert cfg["runs"]["arms"] == ["FLY-YAW", "FLY-SHUF-YAW", "NOBRAIN-YAW", "PID-CMA"]
    assert len(runs) == 12
    assert [(r["arm"], r["seed"]) for r in runs[:4]] == [("FLY-YAW", 1), ("FLY-SHUF-YAW", 1), ("NOBRAIN-YAW", 1), ("PID-CMA", 1)]
    assert ct.run_budget_usd(cfg, "FLY-YAW") == pytest.approx(800 * 0.4 / 3)
    assert ct.run_budget_usd(cfg, "FLY-SHUF-YAW") == pytest.approx(800 * 0.3 / 3)
    assert ct.run_budget_usd(cfg, "FLY-CMA") == pytest.approx(800 * 0.1)  # optional single full-fly run
    assert ct.chunk_size(cfg, "NOBRAIN-YAW") == ct.chunk_size(cfg, "NOBRAIN") > ct.chunk_size(cfg, "FLY-YAW")
    for arm in YAW_TRAINED:
        assert cfg["projection"]["episode_wall_s"][arm] > 0


def test_projection_uses_fly_cma_smoke_for_yaw_fly_arms():
    cfg = ct.load_config()
    fly = {"container_s_per_gen": 100.0, "chunk_wall_s": 20.0, "n_items": None}
    p = ct.project(cfg, {"FLY-CMA": fly})
    for arm in ("FLY-YAW", "FLY-SHUF-YAW"):
        assert p["per_arm"][arm]["source"] == "smoke of FLY-CMA"
        assert p["per_arm"][arm]["container_s_per_gen"] == pytest.approx(100.0)
    assert p["per_arm"]["NOBRAIN-YAW"]["source"] == "config guess"
    own = ct.project(cfg, {"FLY-CMA": fly, "FLY-YAW": {"container_s_per_gen": 50.0, "chunk_wall_s": 9.0}})
    assert own["per_arm"]["FLY-YAW"]["source"] == "smoke" and own["per_arm"]["FLY-SHUF-YAW"]["source"] == "smoke of FLY-CMA"


@pytest.mark.parametrize("arm", YAW_TRAINED)
def test_trainer_runs_yaw_arms_with_fakes(tmp_path, fake_episodes, arm):
    t = make_trainer(tmp_path, arm=arm, seed=2)
    assert t.brain == ct.brain_for(arm, 2, t.cfg)
    s = t.run(2)
    assert s["gen"] == 2 and s["errors_total"] == 0
    items_brains = {json.loads(line).get("gen") for line in t.log_path.read_text().splitlines()}
    assert items_brains == {1, 2}


def test_cma_stds_for_yaw_param_spaces():
    from flyfollow.rl import params

    cfg = ct.load_config()
    for arm in ("FLY-YAW", "FLY-SHUF-YAW", "NOBRAIN-YAW"):
        try:
            ps = params.param_space(arm)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"param_space({arm}) not ready: {e}")
        stds = ct.cma_stds(arm, cfg["cma"])
        assert stds is not None and stds.size == ps.dim
        assert ps.dim < params.param_space("FLY-CMA").dim  # no forward-only readout parameters
        assert (stds > 0).all()


def _write_best(root, arm, seed, tag, x, brain):
    d = root / "runs" / ct.run_name(arm, seed, tag)
    d.mkdir(parents=True)
    (d / "best.json").write_text(json.dumps({"arm": arm, "x": list(x), "brain": brain, "gen": 10, "sel_fitness": -0.8,
                                             "params": {"dec_b_yaw": 0.1}}))


def test_eval_discovers_yaw_runs_and_references(tmp_path, monkeypatch):
    from flyfollow.interfaces import TRAINED_ARMS
    from flyfollow.rl import eval as fe

    monkeypatch.setenv("FLYFOLLOW_DATA", str(tmp_path))
    cfg = ct.load_config()
    for arm in ("FLY-YAW", "FLY-SHUF-YAW", "NOBRAIN-YAW", "PID-CMA"):
        for seed in (1, 2):
            _write_best(tmp_path, arm, seed, "t", [0.5] * DIM, ct.brain_for(arm, seed, cfg))
    arms = list(TRAINED_ARMS) + list(cfg["eval"]["references"])
    entries = fe.discover(cfg, "t", arms)
    names = [e["name"] for e in entries]
    assert names[0] == "FLY-YAW-HAND" and "FLY-HAND" in names and "PID-HAND" in names
    assert "FLY-SHUF-YAW_s2_t" in names and "FLY-CMA_s1_t" not in names
    shuf2 = next(e for e in entries if e["name"] == "FLY-SHUF-YAW_s2_t")
    assert shuf2["brain"].endswith("_shuf2.npz")
    hand = next(e for e in entries if e["name"] == "FLY-YAW-HAND")
    assert hand["x"] is None and hand["brain"] == ct.brain_for("FLY-YAW-HAND", 1, cfg)


def test_eval_and_chart_end_to_end_with_fakes(tmp_path, fake_episodes, monkeypatch):
    """discover -> evaluate sets -> lesion and bias audit -> summaries -> markdown -> chart PNG, on fake episodes."""
    from flyfollow.rl import chart as fc
    from flyfollow.rl import eval as fe

    monkeypatch.setenv("FLYFOLLOW_DATA", str(tmp_path))
    cfg = small_cfg(eval={"sets": {"test": "train", "stress": "stress"}})
    for arm in ("FLY-YAW", "FLY-SHUF-YAW", "NOBRAIN-YAW", "PID-CMA"):
        for seed in (1, 2, 3):
            x = np.clip(TARGET + 0.05 * seed * (1 if arm != "NOBRAIN-YAW" else -1), 0, 1)
            _write_best(tmp_path, arm, seed, "t", x, ct.brain_for(arm, seed, cfg))
    entries = fe.discover(cfg, "t", ["FLY-YAW", "FLY-SHUF-YAW", "NOBRAIN-YAW", "PID-CMA", "FLY-YAW-HAND", "PID-HAND"])
    plan = fe.plan_for(6, 4)
    backend = FakeBackend()
    recs = fe.evaluate_sets(entries, cfg["eval"]["sets"], plan, backend, cfg, None)
    lesion = fe.lesion_and_bias([e for e in entries if e["arm"] == "FLY-YAW"], plan, backend, cfg, None)
    assert set(lesion) == {"FLY-YAW_s1_t", "FLY-YAW_s2_t", "FLY-YAW_s3_t"}
    L = lesion["FLY-YAW_s1_t"]
    assert L["n_pairs"] == 10 and L["lesioned"]["follow_rms_bearing_err_deg"] > L["intact"]["follow_rms_bearing_err_deg"]
    assert L["bias_audit"]["yaw_bias_ratio"] == pytest.approx(0.2)
    report = {"name": "t", "time": "now", "tag": "t", "n_follow": 6, "n_approach": 4, "episode_s": None, "sets": cfg["eval"]["sets"],
              "entries": entries, "summaries": {}, "arms": {}, "lesion": lesion}
    for set_name in cfg["eval"]["sets"]:
        pid = {(r["seed"], r["kind"]): r["ret"] for r in recs[set_name]["PID-HAND"]}
        report["summaries"][set_name] = {e["name"]: fe.summarize(recs[set_name][e["name"]], pid, cfg) for e in entries}
        report["arms"][set_name] = {}
        for arm in fe.ARM_ORDER:
            ss = [report["summaries"][set_name][e["name"]] for e in entries if e["arm"] == arm]
            if ss:
                report["arms"][set_name][arm] = fe.aggregate_arm(ss)
    agg = report["arms"]["test"]["FLY-YAW"]["follow.frac_in_band"]
    assert agg["n_seeds"] == 3 and agg["min"] <= agg["mean"] <= agg["max"]
    assert report["arms"]["test"]["PID-HAND"]["fitness"]["mean"] == pytest.approx(-10.0 / 20.0, rel=0.2)  # floor 20
    md = tmp_path / "t.md"
    fe.write_markdown(report, md)
    text = md.read_text()
    assert "n/a (PID forward)" in text and "FLY-SHUF-YAW" in text
    out = fc.render(json.loads(json.dumps(report, default=ct._json_default)), {}, cfg, "test", tmp_path / "t.png",
                    list(fc.DEFAULT_ARMS), list(fc.DEFAULT_REFS))
    assert out.exists() and out.stat().st_size > 20_000
    assert fc.label("FLY-SHUF-YAW") == "Shuffled fly steers" and fc.label("PID-HAND") == "PID (hand)"


# --------------------------------------------------------------------------------------------
# Fine-tuning: env overrides (reward weights) and --init-from
# --------------------------------------------------------------------------------------------


class RecordingBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.items: list[dict] = []

    def evaluate(self, items):
        self.items.extend(items)
        return super().evaluate(items)


def test_parse_env_overrides():
    assert ct.parse_env_overrides(["reward.w_j=26.4", "episode.follow_s=30"]) == {"reward": {"w_j": 26.4}, "episode": {"follow_s": 30}}
    assert ct.parse_env_overrides("reward.w_j=26.4,reward.w_x=7") == {"reward": {"w_j": 26.4, "w_x": 7}}
    with pytest.raises(ValueError):
        ct.parse_env_overrides(["reward.w_j"])


def test_env_overrides_reach_every_item_including_pid_denominators(tmp_path, fake_episodes):
    cfg = small_cfg(env_overrides={"reward": {"w_j": 26.4}}, episodes={"episode_s": 10})
    be = RecordingBackend()
    t = make_trainer(tmp_path, cfg=cfg, backend=be)
    t.run(2)
    roles = {it["tag"]["role"] for it in be.items}
    assert roles == {"cand", "pid", "sel"}
    assert all(it["overrides"] == {"reward": {"w_j": 26.4}, "episode_s": 10.0} for it in be.items)
    assert all('"w_j": 26.4' in k for k in t.state["pid_cache"])  # denominators cached under the same env
    conf = json.loads((t.run_dir / "config.json").read_text())
    assert conf["env_overrides"] == {"reward": {"w_j": 26.4}, "episode_s": 10.0}
    assert ev.apply_overrides({"reward": {"w_j": 6.6, "w_x": 7.6}}, be.items[0]["overrides"])["reward"] == {"w_j": 26.4, "w_x": 7.6}


def test_resume_refuses_changed_env_overrides(tmp_path, fake_episodes):
    make_trainer(tmp_path, cfg=small_cfg(env_overrides={"reward": {"w_j": 26.4}})).run(1)
    with pytest.raises(RuntimeError, match="env_overrides"):
        make_trainer(tmp_path, cfg=small_cfg()).load_or_init()


def _fake_best(path, x, arm="FLY-YAW", brain="pursuit_core1.npz", calibration=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"arm": arm, "run_seed": 3, "tag": "v1", "gen": 140, "sel_fitness": -0.75, "x": list(x),
                                "brain": brain, "calibration": calibration}))
    return path


def test_load_init_resolves_repo_style_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYFOLLOW_DATA", str(tmp_path))
    _fake_best(tmp_path / "runs" / "FLY-YAW_s3_v1" / "best.json", [0.5] * 34)
    rec = ct.load_init("data/runs/FLY-YAW_s3_v1/best.json")  # as typed in the repo; found under FLYFOLLOW_DATA
    assert len(rec["x"]) == 34 and rec["arm"] == "FLY-YAW" and rec["gen"] == 140 and len(rec["sha256"]) == 64
    with pytest.raises(FileNotFoundError):
        ct.load_init("data/runs/nope/best.json")


def test_init_from_sets_x0_and_is_recorded(tmp_path, fake_episodes, monkeypatch):
    from flyfollow.rl.params import param_space

    dim = param_space("FLY-YAW").dim
    x = np.linspace(0.3, 0.7, dim)
    rec = ct.load_init(_fake_best(tmp_path / "b" / "best.json", x))
    monkeypatch.setattr(ct, "calibration_fingerprint", lambda arm, brain: None)
    cfg = small_cfg(selection={"every": 0, "at_start": True, "at_end": False})
    t = ct.Trainer("FLY-YAW", 3, cfg, FakeBackend(), run_dir=tmp_path / "run", init=rec, log=lambda m: None)
    t.run(1)
    np.testing.assert_allclose(t.state["x0"], x)
    np.testing.assert_allclose(_read_jsonl(t.sel_path)[0]["x"], x)  # gen 0 selection scores the init itself
    conf = json.loads((t.run_dir / "config.json").read_text())
    assert conf["init_from"]["sha256"] == rec["sha256"] and "x" not in conf["init_from"]
    assert json.loads(t.best_path.read_text())["init_from"]["gen"] == 140
    # a different init cannot resume this run
    other = dict(rec, sha256="0" * 64)
    with pytest.raises(RuntimeError, match="started from"):
        ct.Trainer("FLY-YAW", 3, cfg, FakeBackend(), run_dir=tmp_path / "run", init=other, log=lambda m: None).load_or_init()


def test_init_from_refuses_wrong_dim_brain_or_calibration(tmp_path, monkeypatch):
    from flyfollow.rl.params import param_space

    dim = param_space("FLY-YAW").dim
    cfg = small_cfg()

    def trainer(rec, name):
        return ct.Trainer("FLY-YAW", 3, cfg, FakeBackend(), run_dir=tmp_path / name, init=rec, log=lambda m: None)

    with pytest.raises(ValueError, match="parameters"):
        trainer(ct.load_init(_fake_best(tmp_path / "a" / "best.json", [0.5] * (dim + 1))), "a").load_or_init()
    with pytest.raises(ValueError, match="brain"):
        trainer(ct.load_init(_fake_best(tmp_path / "b" / "best.json", [0.5] * dim, brain="pursuit_core2.npz")), "b").load_or_init()
    stale = {"file": "FLY-CMA__pursuit_core1.json", "sha256": "f" * 64}
    monkeypatch.setattr(ct, "calibration_fingerprint", lambda arm, brain: {"file": "FLY-CMA__pursuit_core1.json", "sha256": "e" * 64})
    with pytest.raises(ValueError, match="calibration"):
        trainer(ct.load_init(_fake_best(tmp_path / "c" / "best.json", [0.5] * dim, calibration=stale)), "c").load_or_init()


def test_modal_train_body_with_init_and_env_overrides(tmp_path, fake_episodes, monkeypatch):
    from flyfollow.rl import modal_app
    from flyfollow.rl.params import param_space

    monkeypatch.setenv("FLYFOLLOW_DATA", str(tmp_path))
    monkeypatch.setattr(modal_app, "volume", FakeVolume())
    fn = FakeModalFn()
    monkeypatch.setattr(modal_app, "evaluate_chunk", fn)
    monkeypatch.setattr(ct, "calibration_fingerprint", lambda arm, brain: None)
    x = np.full(param_space("FLY-YAW").dim, 0.4)
    rec = ct.load_init(_fake_best(tmp_path / "src" / "best.json", x))
    over = {"cma": {"popsize": 4, "sigma0": 0.02}, "episodes": {"k_follow": 2, "k_approach": 1}, "selection": {"every": 5},
            "budget": {"per_run_usd": 15}, "env_overrides": {"reward": {"w_j": 26.4}}}
    s = modal_app.train.local("FLY-YAW", 3, 1, "smooth", over, None, rec)
    assert s["gen"] == 1
    run_dir = tmp_path / "runs" / "FLY-YAW_s3_smooth"
    conf = json.loads((run_dir / "config.json").read_text())
    assert conf["init_from"]["sha256"] == rec["sha256"] and conf["cfg"]["cma"]["sigma0"] == 0.02
    assert conf["cfg"]["budget"]["per_run_usd"] == 15
    sent = [it for chunk in fn.chunks for it in chunk]
    assert sent and all(it["overrides"] == {"reward": {"w_j": 26.4}} for it in sent)
    assert any(it["arm"] == "PID-HAND" for it in sent)


def test_launch_preview_with_fine_tune_options(tmp_path, monkeypatch, capsys):
    """The launch entrypoint without --confirm: prints the plan, touches nothing on Modal."""
    from flyfollow.rl import modal_app
    from flyfollow.rl.params import param_space

    monkeypatch.setenv("FLYFOLLOW_DATA", str(tmp_path))
    _fake_best(tmp_path / "runs" / "FLY-YAW_s3_v1" / "best.json", [0.5] * param_space("FLY-YAW").dim)
    raw = modal_app.launch.info.raw_f
    raw(confirm=False, arms="FLY-YAW", seeds="3", gens=40, tag="smooth", init_from="data/runs/FLY-YAW_s3_v1/best.json",
        env_override="reward.w_j=26.4", sigma0=0.02, pop=32, k_follow=20, k_approach=12, select_every=5, budget=15.0)
    out = capsys.readouterr().out
    assert "FLY-YAW       seed 3 gens 40" in out and "budget $15" in out
    assert '"w_j": 26.4' in out and "init from data/runs/FLY-YAW_s3_v1/best.json" in out
    assert "not launching" in out
    with pytest.raises(SystemExit, match="parameters"):
        raw(confirm=False, arms="PID-CMA", seeds="3", init_from="data/runs/FLY-YAW_s3_v1/best.json")


# --------------------------------------------------------------------------------------------
# Integration with the real env and controllers (skipped until they exist)
# --------------------------------------------------------------------------------------------


def _have(mod: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(mod) is not None
    except ModuleNotFoundError:
        return False


real = pytest.mark.skipif(
    not (_have("flyfollow.rl.env") and _have("flyfollow.rl.rollout") and _have("flyfollow.rl.controllers")),
    reason="env, rollout or controllers not landed yet",
)


@real
@pytest.mark.parametrize("arm", ["PID-HAND", "NOBRAIN", "NOBRAIN-YAW", "FLY-YAW-HAND"])
def test_real_episode_deterministic_with_controller_cache(arm):
    ev.clear_caches()
    brain = ct.brain_for(arm, 1, ct.load_config())
    it = ev.make_item(arm, 10_001, "follow", brain=brain, overrides={"episode_s": 5.0})
    r1 = ev.evaluate_item(it)
    if not r1["ok"] and arm in ("NOBRAIN-YAW", "FLY-YAW-HAND") and any(
        m in (r1["error"] or "") for m in ("unknown arm", "no calibration", "FileNotFoundError", "KeyError")
    ):
        pytest.skip(f"{arm} controller not landed yet: {r1['error']}")
    assert r1["ok"], r1.get("traceback")
    ev.evaluate_item(ev.make_item(arm, 10_002, "approach", brain=brain, overrides={"episode_s": 5.0}))  # dirty the cached controller
    r2 = ev.evaluate_item(it)
    assert r1["ret"] == r2["ret"], "controller reuse changed the result: Controller.reset does not reset all state"
    ev.clear_caches()
    assert ev.evaluate_item(it)["ret"] == r1["ret"]
