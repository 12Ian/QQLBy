"""Regression tests for the A-class fidelity fixes of PAPER_FIDELITY_REVIEW.

Each test pins one formula-level deviation from the paper that a correct-looking
implementation can silently reintroduce:

  A-01  Eq. (55) weights the supervised geometry term by 1/(2 sigma_sup^2).
  A-02  Eq. (36) divides the teacher scores by tau exactly once.
  A-03  Eq. (49) aggregates risk with a windowed log-sum-exp.
  A-04  Eqs. (50)-(51) measure the corridor deviation as (dist - b) / b.
  A-05  Eq. (24) bounds the time-averaged acceleration increment v theta^2 / T^2.
  A-06  Eq. (14) feeds z = [h_t; h_s] straight into the classification head.
  A-11  Eq. (41) sums (not averages) the cross-group cosine penalties.
  A-12  Eq. (37) sums (not averages) over the temporal axis.
  A-13  IV-B: "The linear regression is employed for the output head."
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from trajectory_prediction.candidate_pruning import anchor_directions, candidate_branch
from trajectory_prediction.config import Config
from trajectory_prediction.military_teachers import (_risk_score, corridor_deviation,
                                                     military_teacher_scores)
from trajectory_prediction.model import IntentionEncoder, build_forecaster
from trajectory_prediction.objective import _edge_subsets, attention_terms

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
G0 = 9.81


def _config(name):
    return Config(**json.loads((CONFIGS / name).read_text(encoding="utf-8")))


# ---- A-02 + A-12: one temperature division, sum over the temporal axis -----
def test_military_teachers_are_tempered_once_and_summed_over_time():
    """Eq. (36) tau appears once; Eq. (37) sums over t and averages over |H_k|."""
    torch.manual_seed(0)
    cfg = Config(observation_steps=8, prediction_steps=4, dt=.1, heads=6, graph_dim=48,
                 military_teachers=True, attention_supervision=True,
                 entropy_weight=0., jitter_weight=0., teacher_temperature=.1).validate()
    nodes = 3 * cfg.observation_steps                       # target + defender + candidates
    observed = torch.randn(2, nodes, 6)
    observed[..., 3:] *= 40.
    node_valid = torch.ones(2, nodes, dtype=torch.bool)
    attention = [torch.rand(2, cfg.heads, nodes, nodes).softmax(-1)]
    groups, _ = attention_terms(attention, observed, cfg, node_valid, cfg.observation_steps)

    teachers = military_teacher_scores(observed, cfg, cfg.observation_steps)
    width = cfg.heads // 3
    student = attention[0][:, :width]
    scores = teachers[0][:, None].expand_as(student) / cfg.teacher_temperature
    teacher = scores.masked_fill(~(student.detach() > 0), -torch.inf).softmax(-1)
    kl = (teacher * (teacher.clamp_min(1e-12).log() - student.clamp_min(1e-12).log())).sum(-1)
    expected = kl.sum(-1).mean()                            # sum over t, /|H_k|, /batch
    assert groups[0].item() == pytest.approx(expected.item(), rel=1e-5)
    # Doubling tau must change the term: had the code divided twice, the effective
    # temperature would be tau^2 and this scaling would not match Eq. (36).
    scaled = Config(**{**vars(cfg), "teacher_temperature": .2}).validate()
    scaled_groups, _ = attention_terms(attention, observed, scaled, node_valid, scaled.observation_steps)
    assert scaled_groups[0].item() != pytest.approx(groups[0].item(), rel=1e-3)


# ---- Eq. (36): 1(i->j in E_k(i)) restricts each teacher to its edge subset --
def test_teacher_distributions_use_their_own_edge_subset():
    """Eq. (36) + Definition 1: pi^(k) is normalised over E_k(i), not over every
    admissible edge, and each head group uses a *different* subset."""
    torch.manual_seed(11)
    steps = 8
    cfg = Config(observation_steps=steps, prediction_steps=4, dt=.1, heads=6, graph_dim=48,
                 military_teachers=True, attention_supervision=True,
                 entropy_weight=0., jitter_weight=0., teacher_temperature=.1,
                 temporal_radius=2, interaction_time_radius=3,
                 teacher_edge_subsets=True).validate()
    nodes = 2 * steps
    observed = torch.randn(2, nodes, 6)
    observed[..., 3:] *= 40.
    node_valid = torch.ones(2, nodes, dtype=torch.bool)
    attention = [torch.rand(2, cfg.heads, nodes, nodes).softmax(-1)]
    groups, separation = attention_terms(attention, observed, cfg, node_valid, steps)

    teachers = military_teacher_scores(observed, cfg, steps)
    width = cfg.heads // 3
    subsets = _edge_subsets(nodes, steps, cfg, observed.device)
    # The three subsets are pairwise disjoint (Section III-B edge types).
    assert not (subsets[0] & subsets[1]).any()
    assert not (subsets[0] & subsets[2]).any()
    assert not (subsets[1] & subsets[2]).any()
    # Self loops belong to no subset: temporal needs lag > 0, spatial and
    # countermeasure need a cross-entity pair.
    assert not subsets[0].diagonal().any() and not subsets[1].diagonal().any()
    for group in range(3):
        student = attention[0][:, group * width:(group + 1) * width]
        valid = student.detach() > 0
        restricted = valid & subsets[group][None, None]
        mask = torch.where(~restricted.any(-1, keepdim=True), valid, restricted)
        scores = teachers[group][:, None].expand_as(student) / cfg.teacher_temperature
        teacher = scores.masked_fill(~mask, -torch.inf).softmax(-1).detach()
        kl = (teacher * (teacher.clamp_min(1e-12).log()
                         - student.clamp_min(1e-12).log())).sum(-1)
        assert groups[group].item() == pytest.approx(kl.sum(-1).mean().item(), rel=1e-5)
        assert torch.isfinite(teacher).all()
    # Every group's support is strictly smaller than the unrestricted one.
    row = attention[0][0, 0] > 0
    assert row.sum() > 1 and (subsets[0] & row).sum() < row.sum()
    assert torch.isfinite(separation)


def test_edge_subsets_are_opt_in_and_preserve_the_unrestricted_mask():
    """With the switch off (civilian / single-entity graphs) Eq. (36) keeps the
    unrestricted student support, so the earlier behaviour is unchanged."""
    torch.manual_seed(12)
    steps = 8
    base = dict(observation_steps=steps, prediction_steps=4, dt=.1, heads=6, graph_dim=48,
                military_teachers=True, attention_supervision=True,
                entropy_weight=0., jitter_weight=0., teacher_temperature=.1,
                temporal_radius=2, interaction_time_radius=3)
    observed = torch.randn(2, 2 * steps, 6)
    observed[..., 3:] *= 40.
    node_valid = torch.ones(2, 2 * steps, dtype=torch.bool)
    attention = [torch.rand(2, 6, 2 * steps, 2 * steps).softmax(-1)]
    off, _ = attention_terms(attention, observed,
                             Config(**base, teacher_edge_subsets=False).validate(),
                             node_valid, steps)
    on, _ = attention_terms(attention, observed,
                            Config(**base, teacher_edge_subsets=True).validate(),
                            node_valid, steps)
    assert off[0].item() != pytest.approx(on[0].item(), rel=1e-3)
    # Single-entity callers (no entity_steps) cannot build cross-entity subsets,
    # so the switch must be a no-op there instead of emptying the support.
    civilian = {**base, "military_teachers": False}
    single_attention = torch.rand(2, 6, steps, steps).softmax(-1)
    single_observed = torch.randn(2, steps, 6)
    single, _ = attention_terms([single_attention], single_observed,
                                Config(**civilian, teacher_edge_subsets=True).validate(),
                                None, None)
    plain, _ = attention_terms([single_attention], single_observed,
                               Config(**civilian, teacher_edge_subsets=False).validate(),
                               None, None)
    assert torch.isfinite(torch.stack(single)).all()
    assert plain[0].item() == pytest.approx(single[0].item(), rel=1e-6)


# ---- A-11: Eq. (41) sums the cross-group cosine penalties -------------------
def test_separation_sums_cross_group_cosines():
    torch.manual_seed(3)
    cfg = Config(observation_steps=8, prediction_steps=4, dt=.1, heads=6, graph_dim=48,
                 attention_supervision=True, separation_weight=.1).validate()
    nodes = 2 * cfg.observation_steps
    observed = torch.randn(2, nodes, 6)
    node_valid = torch.ones(2, nodes, dtype=torch.bool)
    attention = [torch.rand(2, cfg.heads, nodes, nodes).softmax(-1)]
    _, separation = attention_terms(attention, observed, cfg, node_valid, cfg.observation_steps)

    width = cfg.heads // 3
    flat = F.normalize(attention[0].flatten(2), dim=-1, eps=1e-12)
    cosine = flat @ flat.transpose(1, 2)
    group = torch.arange(cfg.heads, device=cosine.device) // width
    cross = group[:, None] != group[None]
    expected = cosine[:, cross].square().sum(-1).mean()
    assert separation.item() == pytest.approx(expected.item(), rel=1e-5)


# ---- A-03: Eq. (49) windowed log-sum-exp over the prediction horizon -------
def test_risk_teacher_aggregates_the_future_window(monkeypatch):
    """s^(R)_ij = (1/lambda_R) log sum_{tau=t}^{t+H} w_j,tau exp(lambda_R r_j,tau)."""
    import trajectory_prediction.military_teachers as module
    cfg = Config(observation_steps=6, prediction_steps=3, dt=.1, teacher_risk_lambda=1.,
                 military_teachers=True, attention_supervision=True, heads=3,
                 context_entities=1, engagement_edges=True, static_state_branch=True,
                 decoder_type="autoregressive_gru").validate()
    length, entities = cfg.observation_steps, 2
    observed = torch.zeros(1, entities * length, 6)
    observed[..., 3:] = torch.tensor([0., 0., 300.])
    threat = torch.zeros(1, entities * length)
    threat[0, 4] = 3.0                                       # a spike at tau = 4
    monkeypatch.setattr(module, "_threat_levels", lambda *args, **kwargs: threat)

    score = _risk_score(observed, cfg, length)[0]            # [N, N]
    horizon = cfg.prediction_steps
    seconds = max(cfg.prediction_steps, 1) * cfg.dt
    for i in range(entities * length):
        t = i % length
        for j in range(entities * length):
            entity = j // length
            total = 0.
            for tau in range(t, min(t + horizon, length - 1) + 1):
                weight = math.exp(-((tau - t) * cfg.dt) / seconds)
                total += weight * math.exp(threat[0, entity * length + tau].item())
            assert score[i, j].item() == pytest.approx(math.log(total), rel=1e-4, abs=1e-5)
    # The spike only enters the aggregation once the window reaches tau = 4.
    assert score[0, 0].item() < score[3, 0].item()


# ---- A-04: Eqs. (50)-(51) delta = (dist - b) / b ---------------------------
def test_corridor_deviation_is_measured_against_the_half_width():
    assert corridor_deviation(torch.tensor(0.), torch.tensor(100.)).item() == pytest.approx(-1.)
    assert corridor_deviation(torch.tensor(100.), torch.tensor(100.)).item() == pytest.approx(0.)
    assert corridor_deviation(torch.tensor(200.), torch.tensor(100.)).item() == pytest.approx(1.)


# ---- A-05: Eq. (24) v * theta^2 / T^2 <= a_max -----------------------------
@pytest.mark.parametrize("speed,kept", [(300., True), (450., False)])
def test_candidate_energy_gate_uses_the_mean_acceleration_increment(speed, kept):
    """Eq. (24) bounds v theta^2 / T^2, not the plain turn load v theta / T."""
    cfg = Config(observation_steps=8, prediction_steps=60, dt=.1, engagement_overload_g=9.,
                 context_entities=1, candidate_pruning=True, candidate_anchors=2,
                 static_state_branch=True, decoder_type="autoregressive_gru",
                 relative_frame="world").validate()
    observed = torch.zeros(2, cfg.observation_steps, 6)
    observed[:, :, 5] = speed                                # heading along +z
    anchors = torch.tensor([[0., 0., 1.], [0., 0., -1.]])    # straight / reversal
    result = candidate_branch(observed, None, None, cfg, anchors)
    horizon = cfg.prediction_steps * cfg.dt
    a_max = cfg.engagement_overload_g * G0
    turn_load = speed * math.pi / horizon                    # the *old* criterion
    mean_increment = turn_load * math.pi / horizon           # Eq. (24)
    assert (mean_increment <= a_max) is kept
    assert bool(result["keep"][0, 0])                        # straight branch survives
    assert bool(result["keep"][0, 1]) is kept
    if speed == 300.:
        # Discriminating case: the reversal is admissible under Eq. (24) even
        # though its turn load alone exceeds the overload limit.
        assert turn_load > a_max > mean_increment


# ---- A-06: Eq. (14) z = [h_t; h_s] ----------------------------------------
def test_intention_encoder_splices_both_branches():
    cfg = Config(static_state_branch=True, projection_dim=8, lstm_hidden=8,
                 intention_classes=9).validate()
    encoder = IntentionEncoder(cfg, static_dim=6)
    assert not hasattr(encoder, "fusion")
    assert encoder.classifier[0].in_features == 2 * cfg.projection_dim
    logits, context = encoder(torch.randn(2, 5, 6), torch.randn(2, 6))
    assert context.shape == (2, 2 * cfg.projection_dim)
    assert logits.shape == (2, cfg.intention_classes)
    # Without the static branch the splice degenerates to h_t alone.
    plain = IntentionEncoder(Config(static_state_branch=False, projection_dim=8,
                                    lstm_hidden=8, intention_classes=9).validate())
    assert plain.classifier[0].in_features == 8


# ---- A-13: IV-B linear-regression output head ------------------------------
def test_output_head_is_linear_regression():
    cfg = _config("engagement_forecast.json")
    model = build_forecaster(cfg)
    assert isinstance(model.decoder, nn.Sequential)
    assert len(model.decoder) == 1
    assert isinstance(model.decoder[0], nn.Linear)
    assert (model.decoder[0].in_features, model.decoder[0].out_features) == (cfg.graph_dim, 3)
    # Eq. (32) keeps the generic MLP reading available.
    mlp = build_forecaster(Config(**{**vars(cfg), "decoder_head": "mlp"}))
    assert len(mlp.decoder) == 3


# ---- C6: Section IV-B per-segment bounded sampling of a_bar_eva ------------
def test_evasion_overload_is_resampled_per_segment(tmp_path):
    """IV-B: "Segment duration and the nominal overload level a_bar_eva are
    sampled within predefined bounds ... holding a_eva(t) within each short
    primitive segment and updating it at segment transitions."

    With evasion_overload_range=(lo, hi) the commanded evasion overload must
    vary across segments (bounded sampling) instead of sitting at saturation
    for the entire post-trigger tail.
    """
    from trajectory_prediction.engagement_scenarios import simulate_engagements
    path = tmp_path / "eva.npz"
    meta = simulate_engagements(path, scenes=16, steps=260, dt=.1, seed=11,
                                evasion_overload_range=(0.3, 0.6))
    assert 0.3 * 9 * G0 <= meta["max_overload_g"] * G0
    with np.load(path) as z:
        states = z["states"].reshape(16, 260, 6)
        evade = z["evasion_mask"].reshape(16, 260)
    # The commanded normal overload during evasion segments must stay within
    # [0.3, 0.6] * aT_max (the sampler bounds), never at full saturation.
    for i in range(16):
        rows = evade[i]
        if rows.sum() < 40:
            continue
        v = states[i, rows, 3:]
        a = np.linalg.norm(np.diff(v, axis=0) / .1, axis=-1)
        assert a.max() <= 0.6 * 9 * G0 + 1e-3
        assert a.max() >= 0.25 * 9 * G0  # bounded sampling, not ~0 either
    assert meta["evasion_overload_range"] == [0.3, 0.6]


# ---- C9: intercept termination freezes the tail -----------------------------
def test_intercept_termination_fills_frozen_tail(tmp_path):
    """A complete engagement segment ends at interception (r <= radius) or a
    certain miss; post-termination rows must hold the frozen terminal state
    (no post-intercept crossing artifacts) and the diagnostics must record
    the termination reason."""
    from trajectory_prediction.engagement_scenarios import simulate_engagements
    path = tmp_path / "term.npz"
    meta = simulate_engagements(path, scenes=24, steps=300, dt=.1, seed=5,
                                intercept_radius_m=200., miss_grow_steps=25)
    reasons = {d["termination_reason"] for d in meta["diagnostics"]}
    assert reasons <= {"intercepted", "missed", "ran_to_horizon"}
    assert any(d["termination_reason"] != "ran_to_horizon" for d in meta["diagnostics"])
    assert meta["intercepts"] + meta["misses"] + meta["horizon_completes"] == 24
    with np.load(path) as z:
        states = z["states"].reshape(24, 300, 6)
        ctx = z["context_states"].reshape(24, 300, 1, 6)
    # diagnostics are appended in scene order, so scene i maps to row i
    for idx, d in enumerate(meta["diagnostics"]):
        step = d["terminated_step"]
        if step < 0 or step >= 299:
            continue
        assert np.allclose(states[idx, step + 1], states[idx, step])
        assert np.allclose(ctx[idx, step + 1], ctx[idx, step])
        assert np.allclose(states[idx, -1], states[idx, step])
