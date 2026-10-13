"""Smoke tests for the military E/R/T attention teachers (S3, Eqs. 42-53)."""
import numpy as np
import pytest
import torch

from trajectory_prediction.config import Config
from trajectory_prediction.data import (Windows, fit_normalizer, load_tracks,
                                        split_tracks)
from trajectory_prediction.engagement_scenarios import simulate_engagements
from trajectory_prediction.learner import TrajectoryLearner
from trajectory_prediction.military_teachers import military_teacher_scores
from trajectory_prediction.model import Forecaster
from trajectory_prediction.objective import attention_terms


def military(**updates):
    values = dict(observation_steps=6, prediction_steps=4, stride=5, lstm_hidden=8,
                  projection_dim=8, graph_dim=12, heads=3, graph_layers=1,
                  batch_size=8, epochs=1, device="cpu", cpu_threads=1,
                  context_entities=1, static_state_branch=True,
                  decoder_type="autoregressive_gru", engagement_edges=True,
                  max_speed_mps=450., temporal_radius=2, interaction_time_radius=1,
                  interaction_radius_m=50., attention_supervision=True,
                  military_teachers=True, teacher_corridor_radial_m=600.,
                  teacher_corridor_normal_m=300.)
    values.update(updates)
    return Config(**values).validate()


def test_energy_teacher_penalizes_hard_maneuvers():
    cfg = military()
    straight = torch.zeros(1, 12, 6)
    straight[..., 3] = 300.
    turn = straight.clone()
    turn[:, :6, 4] = 300.  # target velocities perpendicular -> large a_req
    energy = military_teacher_scores(straight, cfg, 6)[0]
    energy_turn = military_teacher_scores(turn, cfg, 6)[0]
    assert energy_turn.mean() < energy.mean()
    assert torch.isfinite(energy).all()


def test_risk_teacher_rewards_break_through_geometry():
    cfg = military()
    straight = torch.zeros(1, 12, 6)
    straight[..., 3] = 300.
    far = straight.clone(); far[:, 6:, 0] = -8000.
    near = straight.clone(); near[:, 6:, 0] = -1000.
    escaping = near.clone(); escaping[:, :6, 3] = 420.  # separation grows
    risk_far = military_teacher_scores(far, cfg, 6)[1]
    risk_near = military_teacher_scores(near, cfg, 6)[1]
    risk_escape = military_teacher_scores(escaping, cfg, 6)[1]
    # Eq. 48 break-through risk grows with separation, escape rate, reachability
    assert risk_far.mean() > risk_near.mean()
    assert risk_escape.mean() > risk_near.mean()


def test_teachers_detached_and_student_trains():
    cfg = military()
    observed = torch.randn(1, 12, 6).requires_grad_(True)
    attention = [torch.randn(1, 3, 12, 12, requires_grad=True)]
    groups, separation = attention_terms(attention, observed, cfg,
                                         torch.ones(1, 12, dtype=torch.bool), 6)
    (sum(groups) + separation).backward()
    assert observed.grad.abs().sum() == 0          # Remark 4: no leakage
    assert attention[0].grad.abs().sum() > 0
    assert all(torch.isfinite(group).all() for group in groups)


def test_full_training_step_with_military_teachers(tmp_path):
    cfg = military()
    data = tmp_path / "e.npz"
    simulate_engagements(data, scenes=6, steps=60, dt=.1, seed=3)
    tracks = load_tracks(data, .1, 10)
    windows = Windows(tracks, split_tracks(tracks, .7, .15, 42)["train"], cfg)
    sample = windows[0]
    batch = {k: sample[k][None] for k in ("observed", "context", "context_mask", "future", "label")}
    model = Forecaster(cfg, fit_normalizer(windows))
    learner = TrajectoryLearner(cfg, model, torch.device("cpu"))
    output, loss = learner.update(batch)
    assert output["trajectory"].shape == (1, cfg.prediction_steps, 3)
    assert torch.isfinite(loss["loss"])


def test_civilian_teachers_unchanged_when_flag_off():
    cfg = military(military_teachers=False)
    observed = torch.randn(1, 12, 6)
    distance = (torch.arange(12)[:, None] - torch.arange(12)[None]).abs().float()
    # Two student distributions: one concentrated on the temporal diagonal,
    # one spreading mass to distant steps. The civilian locality teacher
    # (third group) should prefer the former.
    near = torch.nn.functional.softmax(-distance, dim=-1).repeat(1, 3, 1, 1).requires_grad_(True)
    far = torch.nn.functional.softmax(0.5 * distance, dim=-1).repeat(1, 3, 1, 1)
    near_groups, _ = attention_terms([near], observed, cfg, None, None)
    far_groups, _ = attention_terms([far], observed, cfg, None, None)
    assert float(near_groups[2]) < float(far_groups[2])
    near_groups[2].backward()
    assert near.grad.abs().sum() > 0
