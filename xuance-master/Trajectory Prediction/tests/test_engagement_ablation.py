"""Smoke tests for the military I/E ablation and evasion-phase reporting (S7)."""
from pathlib import Path

import numpy as np
import pytest

from trajectory_prediction.config import Config
from trajectory_prediction.engine import evaluate, train
from trajectory_prediction.engagement_scenarios import simulate_engagements
from trajectory_prediction.study import study
from trajectory_prediction.workflows import compare

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def ablation_cfg(tmp_path=None, **updates):
    cfg = Config.read(CONFIGS / "cpu_engagement.json")
    cfg.epochs, cfg.batch_size, cfg.observation_steps, cfg.prediction_steps = 1, 8, 10, 5
    cfg.device, cfg.workers, cfg.cpu_threads = "cpu", 0, 1
    for key, value in updates.items():
        setattr(cfg, key, value)
    if not cfg.engagement_edges:
        # A7 hard blocking rides on the engagement-edge pathway; the E-off
        # ablation quadrant and context-free configurations drop it too.
        cfg.overload_reachability = False
    return cfg.validate()


def test_engagement_ablation_quadrants(tmp_path):
    cfg = ablation_cfg()
    data = tmp_path / "e.npz"
    simulate_engagements(data, scenes=6, steps=60, dt=.1, seed=31)
    result = compare(data, tmp_path / "study", cfg,
                     ["graph", "graph_no_intention", "graph_no_engagement", "graph_no_intention_engagement"],
                     [42], backend="standalone", graph_training="joint")
    rows = {row["model"]: row for row in result["runs"]}
    assert set(rows) == {"graph", "graph_no_intention", "graph_no_engagement", "graph_no_intention_engagement"}
    for row in rows.values():
        assert np.isfinite(row["ade_m"]) and np.isfinite(row["fde_m"])
    # I axis
    assert rows["graph"]["intention_enabled"] and rows["graph_no_engagement"]["intention_enabled"]
    assert not rows["graph_no_intention"]["intention_enabled"]
    assert not rows["graph_no_intention_engagement"]["intention_enabled"]
    # E axis: engagement edges, candidate cropper and E/R/T teachers together
    assert rows["graph"]["engagement_edges_enabled"] and rows["graph"]["military_teachers_enabled"]
    assert not rows["graph_no_engagement"]["engagement_edges_enabled"]
    assert not rows["graph_no_engagement"]["candidate_pruning_enabled"]
    assert not rows["graph_no_engagement"]["military_teachers_enabled"]
    assert not rows["graph_no_intention_engagement"]["engagement_edges_enabled"]
    assert (tmp_path / "study" / "comparison.json").exists()


def test_engagement_ablation_two_stage_checkpoint_import(tmp_path):
    """The server study path (two_stage): per-E-axis intention classifiers.

    Regression guard for the 2026-10-12 server crash: importing the
    engagement-trained 15-dim static branch into the E-off quadrants raised
    "size mismatch for static.0.weight" inside torch load_state_dict.
    """
    cfg = ablation_cfg()
    data = tmp_path / "e.npz"
    simulate_engagements(data, scenes=6, steps=60, dt=.1, seed=31)
    result = compare(data, tmp_path / "study2s", cfg,
                     ["graph", "graph_no_engagement", "graph_no_intention_engagement"],
                     [42], backend="standalone", graph_training="two_stage", intention_epochs=1)
    rows = {row["model"]: row for row in result["runs"]}
    assert np.isfinite(rows["graph"]["ade_m"]) and np.isfinite(rows["graph_no_engagement"]["ade_m"])
    # Two per-E-axis classifiers were trained and both exist on disk.
    assert (tmp_path / "study2s" / "_intention" / "seed_42" / "best.pt").exists()
    assert (tmp_path / "study2s" / "_intention" / "plain_seed_42" / "best.pt").exists()


def test_evasion_phase_errors_are_reported(tmp_path):
    cfg = ablation_cfg()
    data = tmp_path / "e.npz"
    simulate_engagements(data, scenes=6, steps=120, dt=.1, seed=37)
    output = tmp_path / "run"
    train(data, output, cfg)
    metrics = evaluate(data, output / "last.pt", split="test", device_name="cpu")
    assert "evasion_phases" in metrics
    phases = metrics["evasion_phases"]
    assert phases["pre_evasion"]["samples"] + phases["evasion"]["samples"] == metrics["samples"]
    for group in phases.values():
        assert np.isfinite(group["ade_m"]) and np.isfinite(group["fde_m"])
    assert phases["evasion"]["samples"] > 0      # the trigger fires inside the library


def test_study_engagement_mode_requires_context():
    cfg = ablation_cfg(context_entities=0, engagement_edges=False, military_teachers=False,
                       candidate_pruning=False, attention_supervision=False, heads=1)
    with pytest.raises(ValueError, match="context_entities=1"):
        study("unused", cfg, ["graph"], [1], [2], tracks=3, steps=20, data_mode="engagement",
              backend="standalone")
