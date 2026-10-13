"""Smoke tests for the two paper sampling protocols (S6, IV-A / IV-B)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from trajectory_prediction.config import Config
from trajectory_prediction.data import Windows, load_tracks, split_tracks, validate_split
from trajectory_prediction.engagement_scenarios import simulate_engagements
from trajectory_prediction.engine import evaluate, train
from trajectory_prediction.learner import TrajectoryLearner
from trajectory_prediction.model import build_forecaster

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_sample_stride_decimation_matches_fine_simulation(tmp_path):
    """IV-A data (dt=0.5 s, stride 5) must be the 5:1 decimation of the same
    dt=0.1 s simulation: identical states, labels, and engagement descriptors."""
    fine_path, coarse_path = tmp_path / "fine.npz", tmp_path / "coarse.npz"
    fine = simulate_engagements(fine_path, scenes=4, steps=200, dt=.1, seed=11,
                                sample_stride=1, overload_window_steps=20)
    coarse = simulate_engagements(coarse_path, scenes=4, steps=40, dt=.5, seed=11,
                                  sample_stride=5, overload_window_steps=4)
    assert coarse["internal_dt"] == pytest.approx(.1) and coarse["sample_stride"] == 5
    with np.load(fine_path) as a, np.load(coarse_path) as b:
        assert b["time"].shape == (160,)                       # 4 scenes x 40 rows
        assert np.allclose(b["time"][:2], [0., .5])
        for key in ("states", "context_states", "labels", "overload_labels", "engagement_states"):
            assert np.allclose(a[key][::5], b[key]), f"{key} is not the 5:1 decimation"
    # The decimated file is loadable under the IV-A protocol (dt=0.5 s).
    tracks = load_tracks(coarse_path, .5, 32)
    assert len(tracks) == 4
    assert np.allclose(np.diff(tracks[0].time), .5)


def test_holdout_test_false_gives_ivb_eighty_twenty_split(tmp_path):
    data = tmp_path / "e.npz"
    simulate_engagements(data, scenes=6, steps=60, dt=.1, seed=13)
    tracks = load_tracks(data, .1, 20)
    splits = split_tracks(tracks, .8, .2, 42, holdout_test=False)
    assert splits["test"] == []
    assert len(splits["train"]) == 4 and len(splits["validation"]) == 2
    validate_split(tracks, splits, holdout_test=False)
    with pytest.raises(ValueError, match="Empty data partition"):
        validate_split(tracks, splits, holdout_test=True)
    # Default behaviour (civilian / IV-A) keeps a test partition.
    legacy = split_tracks(tracks, .7, .15, 42)
    assert legacy["test"] and len(legacy["train"]) == 4
    validate_split(tracks, legacy)


def test_paper_protocol_configs_validate():
    intention = Config.read(CONFIGS / "engagement_intention.json")
    overload = Config.read(CONFIGS / "engagement_overload.json")
    forecast = Config.read(CONFIGS / "engagement_forecast.json")
    smoke = Config.read(CONFIGS / "cpu_engagement.json")
    assert (intention.dt, intention.observation_steps, intention.stride) == (.5, 20, 5)
    assert intention.intention_task == "semantic9" and intention.intention_classes == 9
    assert overload.intention_task == "overload11" and overload.intention_classes == 11
    assert (forecast.dt, forecast.observation_steps, forecast.prediction_steps) == (.1, 80, 60)
    assert forecast.holdout_test is False and forecast.military_teachers is True
    assert forecast.heads == 6 and forecast.graph_top_k == 8 and forecast.attention_temperature == .1
    assert (forecast.ade_weight, forecast.fde_weight) == (.4, .6)
    assert smoke.epochs <= 3 and smoke.device == "cpu"


def test_end_to_end_engagement_training_and_evaluation(tmp_path):
    cfg = Config.read(CONFIGS / "cpu_engagement.json")
    cfg.epochs, cfg.batch_size, cfg.observation_steps, cfg.prediction_steps = 1, 8, 10, 5
    cfg.device, cfg.workers, cfg.cpu_threads = "cpu", 0, 1
    cfg.validate()
    data = tmp_path / "e.npz"
    simulate_engagements(data, scenes=6, steps=60, dt=.1, seed=17)
    output = tmp_path / "run"
    summary = train(data, output, cfg)
    assert summary["completed_epoch"] == 1
    report = evaluate(data, output / "last.pt", split="test", device_name="cpu")
    assert np.isfinite(report["ade_m"]) and np.isfinite(report["fde_m"])
    assert (output / "manifest.json").exists() and (output / "history.csv").exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["window_counts"]["test"] > 0
