"""Regression tests for the P0 paper-alignment fixes (PAPER_ALIGNMENT_GAP_AUDIT).

Covers: C1 nine-class coverage + window-level balance, C2 signed eleven-level
overload quantization, B8 1 m early-stopping threshold, A2 engagement
descriptors in the intention/temporal queries, and the Eq. (54)/(55)
uncertainty weighting of the attention teachers and the geometry term.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from trajectory_prediction.baselines import SequenceForecaster
from trajectory_prediction.config import Config
from trajectory_prediction.data import Windows, load_tracks
from trajectory_prediction.engagement_scenarios import (OVERLOAD_LEVELS, SEMANTIC_CLASSES,
                                                        overload_gap_report, overload_label,
                                                        sample_semantic_class, signed_normal_overload,
                                                        simulate_engagements)
from trajectory_prediction.model import build_forecaster
from trajectory_prediction.objective import TrajectoryObjective

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
G0 = 9.81


def _config(name):
    return Config(**json.loads((CONFIGS / name).read_text(encoding="utf-8")))


def _tracks(tmp_path, path="eng.npz", dt=.5, **kwargs):
    data = tmp_path / path
    simulate_engagements(data, scenes=30, steps=200, dt=dt, sample_stride=5, **kwargs)
    return data


# ---- C1: nine semantic classes are all reachable ---------------------------
def test_nine_semantic_classes_are_all_sampled(tmp_path):
    """Primitives are drawn per class, so 'no maneuver' (class 0) exists."""
    rng = np.random.default_rng(7)
    counts = np.bincount([sample_semantic_class(rng) for _ in range(4000)], minlength=SEMANTIC_CLASSES)
    assert counts.min() > 3000 // SEMANTIC_CLASSES          # roughly uniform
    data = _tracks(tmp_path)
    with np.load(data) as archive:
        assert np.count_nonzero(np.bincount(archive["labels"], minlength=SEMANTIC_CLASSES)) == SEMANTIC_CLASSES


def test_class_balance_weights_skew_the_mixture():
    rng = np.random.default_rng(3)
    weights = [6., 1., 1., 1., 1., 1., 1., 1., 1.]
    draws = np.bincount([sample_semantic_class(rng, weights) for _ in range(8000)], minlength=9)
    assert draws[0] > 2 * draws[1]
    with pytest.raises(ValueError):
        sample_semantic_class(rng, [1.] * 4)


def test_windows_are_class_balanced_when_requested(tmp_path):
    """IV-A: balanced coverage. Raw windows are dominated by the post-trigger
    evasion maneuver, so the dataset resamples to equal class counts."""
    cfg = _config("engagement_intention.json")
    assert cfg.balance_intention_classes
    data = _tracks(tmp_path)
    tracks = load_tracks(data, cfg.dt, cfg.observation_steps + cfg.prediction_steps)
    balanced = Windows(tracks, [t.track_id for t in tracks], cfg)
    assert len(balanced.class_counts) == SEMANTIC_CLASSES
    assert min(balanced.class_counts) > 0
    assert len(set(balanced.class_counts)) == 1            # every class equal
    unbalanced = Config(**{**vars(cfg), "balance_intention_classes": False})
    raw = Windows(tracks, [t.track_id for t in tracks], unbalanced)
    assert max(raw.class_counts) > max(balanced.class_counts)   # imbalance was real


# ---- C2: signed eleven-level overload --------------------------------------
def test_signed_overload_covers_negative_levels(tmp_path):
    """Eq. (4) is a magnitude; the 11-level task needs a sign. Left/climb is
    positive, right/dive negative, straight flight is zero."""
    vT = np.array([100., 0., 0.])                    # heading +x, z up, left = +y
    assert signed_normal_overload(np.array([0., 40., 0.]), vT) > 0
    assert signed_normal_overload(np.array([0., -40., 0.]), vT) < 0
    assert signed_normal_overload(np.array([0., 0., 40.]), vT) > 0    # climb
    assert signed_normal_overload(np.array([0., 0., -40.]), vT) < 0   # dive
    assert signed_normal_overload(np.zeros(3), vT) == pytest.approx(0.)
    assert abs(signed_normal_overload(np.array([0., 40., 0.]), vT)) == pytest.approx(40.)
    data = _tracks(tmp_path, path="signed.npz")
    with np.load(data) as archive:
        levels = np.bincount(archive["overload_labels"], minlength=len(OVERLOAD_LEVELS))
    # Signed quantization reaches nine rows: the two extreme bins need
    # |a_n| > 9.5 g, which aT_max = 9 g makes unreachable (paper-side caveat).
    assert np.count_nonzero(levels) >= 9
    assert levels[:5].sum() > 0                                  # negative bins populated
    assert levels[5] > 0                                         # straight/no-load bin
    assert levels[:4].sum() > 0 and levels[6:9].sum() > 0         # both signs used


def test_overload_bins_follow_the_paper_rule():
    """The paper locates a_bar_n in (Im - 0.5, Im + 0.5] g, ties to the larger
    label; 1 g-wide gaps between bins fall back to the nearest level."""
    assert overload_label([0.]) == 5                     # 0 g
    assert overload_label([2.1 * G0]) == 6               # +2 g
    assert overload_label([-2.1 * G0]) == 4              # -2 g
    assert overload_label([2.0 * G0]) == 6               # level centre
    assert overload_label([2.5 * G0]) == 6               # tie -> larger bin
    assert overload_label([1.2 * G0]) == 6               # gap -> nearest level
    assert overload_label([-1.2 * G0]) == 4
    assert overload_label([30. * G0]) == 10              # beyond grid -> end level
    assert overload_label([-30. * G0]) == 0
    # The paper rule leaves the two extreme bins unreachable for aT_max = 9 g,
    # which is why the censored variant exists.
    assert overload_label([9.5 * G0], censored_ends=True) == 10
    assert overload_label([8.6 * G0], censored_ends=True) == 10
    report = overload_gap_report([1.2 * G0, 2.1 * G0])
    assert report["means"] == 2 and report["in_gap"] == 1
    assert report["gap_fraction"] == pytest.approx(.5)
    with pytest.raises(ValueError):
        overload_label([])


# ---- B8: 1 m early-stopping threshold --------------------------------------
def test_early_stopping_threshold_is_one_metre():
    """Paper IV-B stops when 20 epochs improve ADE/FDE by less than 1 m."""
    assert Config().early_stopping_min_delta_m == 1.0
    for name in ("engagement_forecast.json", "paper_forecast.json", "server.json"):
        assert _config(name).early_stopping_min_delta_m == 1.0


# ---- A2: engagement descriptors enter Q_int and Q_tra ----------------------
def test_engagement_descriptors_reach_intention_and_temporal_queries(tmp_path):
    cfg = _config("engagement_forecast.json")
    assert cfg.engagement_edges
    model = build_forecaster(cfg)
    # Q_int / Q_tra width: 6 + [h_t; h_s] + classes + 9 descriptors (Eq. 16),
    # and the static branch of the intention encoder takes state + descriptors.
    assert model.node_embedding.in_features == (6 + 2 * cfg.projection_dim
                                                + cfg.intention_classes + 9)
    assert model.intention.static[0].in_features == 6 + 9
    observed = torch.randn(2, cfg.observation_steps, 6)
    observed[..., 3:] *= 40.
    context = torch.randn(2, cfg.context_entities, cfg.observation_steps, 6)
    context[..., 3:] *= 40.
    mask = torch.ones(2, cfg.context_entities, cfg.observation_steps, dtype=torch.bool)
    engagement = torch.randn(2, cfg.observation_steps, 9)
    torch.manual_seed(0)
    with torch.no_grad():
        first = model(observed, context=context, context_mask=mask, engagement=engagement)
        second = model(observed, context=context, context_mask=mask, engagement=engagement + 1.0)
        neutral = model(observed, context=context, context_mask=mask)
    # Q_int: the intention posterior reacts to the engagement geometry.
    assert not torch.allclose(first["logits"], second["logits"])
    assert torch.isfinite(neutral["trajectory"]).all()
    # Q_tra: the decoder starts at the constant-velocity baseline (zero output
    # layer), so give it nonzero weights to observe the descriptor pathway.
    torch.nn.init.normal_(model.decoder[-1].weight, std=.1)
    torch.nn.init.normal_(model.decoder[-1].bias, std=.1)
    with torch.no_grad():
        left = model(observed, context=context, context_mask=mask, engagement=engagement)
        right = model(observed, context=context, context_mask=mask, engagement=engagement + 1.0)
    assert not torch.allclose(left["trajectory"], right["trajectory"])
    with pytest.raises(ValueError):
        model(observed, context=context, context_mask=mask, engagement=torch.randn(2, cfg.observation_steps, 4))


def test_windows_emit_engagement_descriptors(tmp_path):
    cfg = _config("engagement_forecast.json")
    data = _tracks(tmp_path, path="desc.npz", dt=cfg.dt)
    tracks = load_tracks(data, cfg.dt, cfg.observation_steps + cfg.prediction_steps)
    assert all(t.engagement is not None for t in tracks)
    item = Windows(tracks, [t.track_id for t in tracks], cfg)[0]
    assert item["engagement"].shape == (cfg.observation_steps, 9)


@pytest.mark.parametrize("name", ["lstm", "gru", "tcn", "transformer"])
def test_baselines_accept_engagement_argument(name, tmp_path):
    """Baselines ignore the descriptors but share the call signature."""
    cfg = Config(**{**vars(_config("engagement_forecast.json")), "model_type": name})
    model = SequenceForecaster(cfg)
    observed = torch.randn(2, cfg.observation_steps, 6)
    output = model(observed, engagement=torch.zeros(2, cfg.observation_steps, 9))
    assert output["trajectory"].shape == (2, cfg.prediction_steps, 3)


# ---- A-01: Eq. (55) weights the geometry term, Eq. (54) keeps the log term --
def test_uncertainty_weighting_covers_geometry_without_log_regulariser(tmp_path):
    """Eq. (55): L_tra = L_attn + 1/(2 sigma_sup^2) (w1 ADE + w2 FDE).

    The supervised geometric term carries its own homoscedastic weight
    1/(2 sigma_sup^2) but *no* log-sigma regulariser: that penalty belongs to
    Eq. (54) and is applied to the three attention groups only.
    """
    cfg = Config(**{**vars(_config("engagement_forecast.json")),
                    "loss_balance": "uncertainty", "attention_supervision": False})
    objective = TrajectoryObjective(cfg)
    model = build_forecaster(cfg)
    observed = torch.randn(2, cfg.observation_steps, 6)
    observed[..., 3:] *= 40.
    batch = {"observed": observed, "future": torch.randn(2, cfg.prediction_steps, 3),
             "label": torch.full((2,), -1),
             "context": torch.randn(2, cfg.context_entities, cfg.observation_steps, 6),
             "context_mask": torch.ones(2, cfg.context_entities, cfg.observation_steps, dtype=torch.bool),
             "engagement": torch.randn(2, cfg.observation_steps, 9)}
    with torch.no_grad():
        output = model(batch["observed"], context=batch["context"], context_mask=batch["context_mask"],
                       engagement=batch["engagement"])
        objective.log_variance.fill_(0.0)
        base = objective(output, batch)                     # 0.5 * geometry
        # sigma_sup^2 = 4 -> the geometry term is quartered, and nothing else
        # is added, so the total loss is exactly base / 4.
        objective.log_variance[3].fill_(math.log(4.0))
        quartered = objective(output, batch)
        objective.log_variance[3].fill_(6.0)
        inflated = objective(output, batch)
    assert base["ade_m"].item() > 0
    assert quartered["loss"] == pytest.approx(base["loss"].item() / 4., rel=1e-4)
    # No +0.5 * log sigma_sup term: only exp(-6) shrinks the weighted geometry.
    assert inflated["loss"] == pytest.approx(base["loss"].item() * math.exp(-6.0), rel=1e-4)
    assert inflated["ade_m"] == pytest.approx(base["ade_m"].item())
