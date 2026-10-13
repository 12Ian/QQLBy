"""Gradient-accumulation guard tests (audit round 4, small-VRAM deployment).

The paper pins ``batch_size = 256`` for both IV-A and IV-B, so the effective
optimization batch must stay 256 even on GPUs (e.g. RTX 5080, 16 GB) where a
full 256-sample forward does not fit.  ``micro_batch_size`` splits each
optimizer step into equal sub-batches whose accumulated gradients equal the
full-batch gradient -- the model carries no BatchNorm-style batch statistics,
so the equivalence is exact up to floating-point summation order.

Defaults keep the historically verified path bit-identical:
``micro_batch_size = 0`` disables accumulation entirely.
"""

import json
from pathlib import Path

import pytest
import torch

from trajectory_prediction.config import Config
from trajectory_prediction.data import Windows, fit_normalizer, load_tracks, split_tracks
from trajectory_prediction.engagement_scenarios import simulate_engagements
from trajectory_prediction.learner import TrajectoryLearner
from trajectory_prediction.model import Forecaster

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _config(**overrides):
    values = dict(observation_steps=6, prediction_steps=4, stride=5, lstm_hidden=8,
                  projection_dim=8, graph_dim=12, heads=3, graph_layers=1,
                  batch_size=8, epochs=1, device="cpu", cpu_threads=1,
                  context_entities=1, static_state_branch=True,
                  decoder_type="autoregressive_gru", engagement_edges=True,
                  max_speed_mps=450., temporal_radius=2, interaction_time_radius=1,
                  interaction_radius_m=50., attention_supervision=True,
                  military_teachers=True, teacher_corridor_radial_m=600.,
                  teacher_corridor_normal_m=300., dropout=0., jitter_weight=0.)
    values.update(overrides)
    return Config(**values).validate()


def test_micro_batch_defaults_off():
    assert Config().micro_batch_size == 0
    cfg = _config()

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)

    learner = TrajectoryLearner(cfg, TinyPolicy(), torch.device("cpu"))
    assert learner.accum_steps == 1


def test_micro_batch_validation():
    with pytest.raises(ValueError):
        _config(batch_size=256, micro_batch_size=100)  # not a divisor
    with pytest.raises(ValueError):
        _config(batch_size=256, micro_batch_size=512)  # larger than the batch
    assert _config(batch_size=256, micro_batch_size=128).micro_batch_size == 128


def test_accumulated_update_matches_full_batch(tmp_path):
    data = tmp_path / "e.npz"
    simulate_engagements(data, scenes=12, steps=60, dt=.1, seed=3)
    tracks = load_tracks(data, .1, 10)
    cfg = _config()  # batch_size=8, accumulation off
    windows = Windows(tracks, split_tracks(tracks, .7, .15, 42)["train"], cfg)
    normalizer = fit_normalizer(windows)
    samples = [windows[index] for index in range(8)]
    keys = ("observed", "context", "context_mask", "future", "label")
    batch8 = {key: torch.stack([sample[key] for sample in samples]) for key in keys}

    # One full-batch step (effective 8 = one step).
    torch.manual_seed(7)
    full = TrajectoryLearner(cfg, Forecaster(cfg, normalizer), torch.device("cpu"))
    full.update(batch8)

    # Two micro steps of 4 with accumulation 2: same effective batch.
    cfg_micro = _config(batch_size=8, micro_batch_size=4)
    torch.manual_seed(7)
    micro = TrajectoryLearner(cfg_micro, Forecaster(cfg_micro, normalizer), torch.device("cpu"))
    assert micro.accum_steps == 2
    first_output, _ = micro.update({key: value[:4] for key, value in batch8.items()})
    assert micro.iterations == 0  # mid-accumulation: no optimizer step yet
    second_output, _ = micro.update({key: value[4:] for key, value in batch8.items()})
    assert micro.iterations == 1
    assert torch.isfinite(first_output["trajectory"]).all()
    assert torch.isfinite(second_output["trajectory"]).all()

    for (name_a, param_a), (name_b, param_b) in zip(full.policy.named_parameters(),
                                                    micro.policy.named_parameters()):
        assert name_a == name_b
        assert torch.allclose(param_a, param_b, rtol=1e-4, atol=1e-6), name_a


def test_trailing_partial_group_matches_plain_loader(tmp_path):
    data = tmp_path / "e.npz"
    simulate_engagements(data, scenes=12, steps=60, dt=.1, seed=3)
    tracks = load_tracks(data, .1, 10)
    cfg = _config()
    windows = Windows(tracks, split_tracks(tracks, .7, .15, 42)["train"], cfg)
    normalizer = fit_normalizer(windows)
    samples = [windows[index] for index in range(6)]
    keys = ("observed", "context", "context_mask", "future", "label")
    batch6 = {key: torch.stack([sample[key] for sample in samples]) for key in keys}

    # A plain batch_size loader would take the trailing 6-sample batch as-is.
    torch.manual_seed(7)
    plain = TrajectoryLearner(cfg, Forecaster(cfg, normalizer), torch.device("cpu"))
    plain.update(batch6)

    # Accumulated: a full 4-sample micro leaves a pending partial group of
    # 2; the epoch-end flush must reproduce the plain 6-sample step.
    cfg_micro = _config(batch_size=8, micro_batch_size=4)
    torch.manual_seed(7)
    micro = TrajectoryLearner(cfg_micro, Forecaster(cfg_micro, normalizer), torch.device("cpu"))
    micro.update({key: value[:4] for key, value in batch6.items()})
    micro.update({key: value[4:] for key, value in batch6.items()})
    assert micro._group_pending == 1
    micro.flush()
    assert micro._group_pending == 0

    for (name_a, param_a), (name_b, param_b) in zip(plain.policy.named_parameters(),
                                                    micro.policy.named_parameters()):
        assert name_a == name_b
        assert torch.allclose(param_a, param_b, rtol=1e-4, atol=1e-6), name_a


def test_engagement_configs_pin_paper_batch():
    for name in ("engagement_forecast.json", "engagement_forecast_literal.json"):
        payload = json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))
        # Paper IV-B: "The batch size is set to 256."
        assert payload["batch_size"] == 256, name
        assert payload["micro_batch_size"] == 128, name
        Config(**payload).validate()
