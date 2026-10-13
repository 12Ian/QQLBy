"""Regression tests for the P1 paper-alignment items.

Covers B5 (Eqs. 38-41 teacher regularisers), F2 (IV-A windows without a
prediction horizon), A6 (FOV attention mask), G4 (Table VI batch simulation),
G7 (macro F1 / majority-class baseline) and the class-balanced window report.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from trajectory_prediction.config import Config
from trajectory_prediction.data import Windows, load_tracks
from trajectory_prediction.engagement_graph import engagement_edges
from trajectory_prediction.engagement_scenarios import (FIGURE14_SCENARIOS,
                                                        simulate_engagement_batch,
                                                        simulate_engagements)
from trajectory_prediction.metrics import EvaluationDetails
from trajectory_prediction.model import build_forecaster
from trajectory_prediction.objective import attention_terms

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _config(name, **overrides):
    base = json.loads((CONFIGS / name).read_text(encoding="utf-8"))
    return Config(**{**base, **overrides})


# ---- B5: paper teacher regularisers ----------------------------------------
def test_paper_teacher_regularisers_differ_from_legacy():
    """"paper" averages the heads before the entropy/jitter terms (Eqs. 38-41);
    "legacy" keeps the earlier per-head approximation."""
    base = _config("engagement_forecast.json", observation_steps=6, prediction_steps=4,
                   graph_dim=48, projection_dim=8, lstm_hidden=8, heads=3, graph_head_dim=16,
                   temporal_heads=2, context_entities=1, candidate_anchors=4,
                   candidate_pruning=False, device="cpu")
    values = {}
    for mode in ("legacy", "paper"):
        cfg = Config(**{**vars(base), "teacher_regularizers": mode})
        model = build_forecaster(cfg)
        observed = torch.randn(2, cfg.observation_steps, 6)
        observed[..., 3:] *= 40.
        context = torch.randn(2, 1, cfg.observation_steps, 6)
        context[..., 3:] *= 40.
        with torch.no_grad():
            output = model(observed, context=context,
                           context_mask=torch.ones(2, 1, cfg.observation_steps, dtype=torch.bool),
                           engagement=torch.randn(2, cfg.observation_steps, 9))
            groups, separation = attention_terms(output["attention"], output["graph_observed"], cfg,
                                                 output.get("node_valid"), output.get("entity_steps"))
        assert all(torch.isfinite(value) for value in groups)
        values[mode] = torch.stack(groups)
    assert not torch.allclose(values["legacy"], values["paper"])
    with pytest.raises(ValueError):
        Config(**{**vars(base), "teacher_regularizers": "guess"}).validate()


# ---- F2: IV-A windows need no future horizon -------------------------------
def test_intention_windows_accept_zero_prediction_steps(tmp_path):
    cfg = _config("engagement_intention.json")
    assert cfg.prediction_steps == 0 and cfg.stage == "intention"
    data = tmp_path / "iva.npz"
    simulate_engagements(data, scenes=20, steps=200, dt=.5, sample_stride=5, initial_fov_coverage=.5)
    tracks = load_tracks(data, cfg.dt, cfg.observation_steps + cfg.prediction_steps)
    dataset = Windows(tracks, [t.track_id for t in tracks], cfg)
    item = dataset[0]
    assert item["observed"].shape == (cfg.observation_steps, 6)
    assert item["future"].shape == (0, 3)
    assert int(item["label"]) >= 0
    model = build_forecaster(cfg)
    output = model(item["observed"][None], context=item["context"][None],
                   context_mask=item["context_mask"][None],
                   engagement=item["engagement"][None])
    assert output["trajectory"].shape == (1, 0, 3)
    assert output["logits"].shape[-1] == cfg.intention_classes
    # A non-intention stage must still demand a real horizon.
    with pytest.raises(ValueError):
        Config(**{**vars(cfg), "stage": "joint"}).validate()


# ---- A6: FOV attention mask ------------------------------------------------
def _two_entity_features(defender_offset):
    cfg = Config(context_entities=1, observation_steps=4, dt=.1, engagement_edges=True,
                 fov_half_angle_deg=60., fov_attention_mask=True, max_speed_mps=450.)
    features = torch.zeros(1, 8, 6)
    features[0, 4:, :3] = torch.tensor(defender_offset, dtype=torch.float32)
    features[:, :, 3] = 300.                     # both fly along +x
    return cfg, features


def test_fov_attention_mask_drops_invisible_engagement_links():
    cfg, features = _two_entity_features([0., 1000., 0.])       # defender 90 deg off-axis
    valid = torch.ones(1, 8, dtype=torch.bool)
    _, allowed = engagement_edges(features, valid, cfg)
    assert not bool(allowed[0, 0, 4])            # target -> defender suppressed
    assert bool(allowed[0, 0, 1])                # in-entity temporal edge survives
    cfg_visible, features_visible = _two_entity_features([1000., 0., 0.])
    _, allowed_visible = engagement_edges(features_visible, valid, cfg_visible)
    assert bool(allowed_visible[0, 0, 4])        # defender dead ahead: linked
    cfg_off = Config(**{**vars(cfg), "fov_attention_mask": False})
    _, allowed_off = engagement_edges(features, valid, cfg_off)
    assert bool(allowed_off[0, 0, 4])            # mask disabled keeps the edge


# ---- G4: Table VI / Fig. 14 batch simulation -------------------------------
def test_table_vi_scenarios_cover_the_four_quadrants(tmp_path):
    conditions = [dict(scenario, r0_m=scenario["r0_km"] * 1e3, vT=300., vM=360.,
                       a_yaw=scenario["yaw_sign"] * abs(scenario["a_phi_0"]),
                       a_pitch=scenario["pitch_sign"] * abs(scenario["a_theta_0"]))
                  for scenario in FIGURE14_SCENARIOS]
    data = tmp_path / "table_vi.npz"
    metadata = simulate_engagement_batch(data, conditions, steps=140, dt=.1, seed=0)
    assert metadata["scenes"] == 4
    assert [row["scenario"] for row in metadata["diagnostics"]] == [c["name"] for c in FIGURE14_SCENARIOS]
    assert metadata["table_vi"][0]["r0_km"] == 5.3
    tracks = load_tracks(data, .1, 140)
    assert len(tracks) == 4
    # The four scenarios must actually differ: distinct terminal states.
    ends = np.asarray([track.states[-1, :3] for track in tracks])
    assert len(np.unique(ends.round(1), axis=0)) == 4
    for track in tracks:
        assert track.engagement is not None and track.overload_labels is not None
        assert float(track.states[:, 3:6].max()) <= 360. + 1e-3      # VM bound


# ---- G7: macro F1 and majority baseline ------------------------------------
def test_classification_metrics_report_macro_f1_and_baseline():
    result = {"confusion_matrix": np.diag([10, 10, 20, 10]).tolist()}
    EvaluationDetails.classification_metrics(result)
    assert result["intention_macro_f1"] == pytest.approx(1.0)
    assert result["intention_balanced_accuracy"] == pytest.approx(1.0)
    assert result["intention_majority_baseline"] == pytest.approx(0.4)
    assert result["intention_classes_present"] == 4
    assert len(result["intention_per_class"]) == 4
    skewed = {"confusion_matrix": np.array([[300, 0], [300, 0]]).tolist()}
    EvaluationDetails.classification_metrics(skewed)
    assert skewed["intention_majority_baseline"] == pytest.approx(0.5)
    assert skewed["intention_macro_f1"] == pytest.approx(1 / 3, abs=1e-6)  # mean(2/3, 0)
