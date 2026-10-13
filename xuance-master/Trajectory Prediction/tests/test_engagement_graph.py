"""Smoke tests for engagement-descriptor edges and countermeasure edges (S2).

Verifies Eq. (15) phi_edg: hand-checked r/rdot/eL/eLdot/visibility against
the pairwise computation, mutual exclusivity of the three edge types, model
forward/backward with engagement edges, edge_context gradients under
attention supervision, and masking invariance for unavailable defenders.
"""
import numpy as np
import pytest
import torch

from trajectory_prediction.config import Config
from trajectory_prediction.data import (Windows, fit_normalizer, load_tracks,
                                        split_tracks)
from trajectory_prediction.engagement_graph import (ENGAGEMENT_EDGE_DIM,
                                                    engagement_edges)
from trajectory_prediction.engagement_scenarios import simulate_engagements
from trajectory_prediction.learner import TrajectoryLearner
from trajectory_prediction.model import Forecaster


def military(**updates):
    values = dict(observation_steps=6, prediction_steps=4, stride=5, lstm_hidden=8,
                  projection_dim=8, graph_dim=12, heads=3, graph_layers=1,
                  batch_size=8, epochs=1, device="cpu", cpu_threads=1,
                  context_entities=1, static_state_branch=True,
                  decoder_type="autoregressive_gru", engagement_edges=True,
                  max_speed_mps=450., temporal_radius=2, interaction_time_radius=1,
                  interaction_radius_m=50., attention_supervision=True)
    values.update(updates)
    return Config(**values).validate()


def test_engagement_descriptors_match_pairwise_computation():
    cfg = military()
    torch.manual_seed(0)
    B, O = 2, 6
    target = torch.randn(B, O, 6) * torch.tensor([3000., 3000., 500., 50., 50., 10.])
    defender = torch.randn(B, O, 6) * torch.tensor([3000., 3000., 500., 50., 50., 10.])
    features = torch.cat((target, defender), 1)
    valid = torch.ones(B, 2 * O, dtype=torch.bool)
    edges, allowed = engagement_edges(features, valid, cfg, 1.0)
    assert edges.shape == (B, 2 * O, 2 * O, ENGAGEMENT_EDGE_DIM)
    i, j = 0, O  # target node 0 -> defender node 0 (same time index)
    pos = features[0, i, :3] - features[0, j, :3]
    vel = features[0, i, 3:] - features[0, j, 3:]
    r, eL = pos.norm(), pos / pos.norm()
    rdot = float((vel * eL).sum())
    edge = edges[0, i, j]
    assert abs(edge[7].item() * cfg.engagement_range_ref_m - r.item()) < 1e-2
    assert abs(edge[8].item() * cfg.max_speed_mps - rdot) < 1e-2
    assert torch.allclose(edge[11:14], eL, atol=1e-4)
    cross = torch.linalg.cross(eL, vel)
    eLdot = torch.linalg.cross(cross, eL) / r
    omega = cfg.max_speed_mps / cfg.engagement_range_ref_m
    assert torch.allclose(edge[14:17], eLdot / omega, rtol=1e-3, atol=1e-5)
    v_i = features[0, i, 3:]
    cosine = float((v_i * (-eL)).sum() / v_i.norm())
    assert edge[17].item() == float(cosine >= np.cos(np.radians(60.)))


def test_edge_descriptor_carries_los_relative_speed_and_its_rate():
    """Section III-B phi_edg: "LOS-relative speed and its temporal variation".

    V_L is a property of the target-defender pairing (Eq. (2)), so the edge
    descriptor of every source node carries the V_L of its own time index,
    together with the backward difference dV_L/dt inside the entity block.
    """
    cfg = military()
    torch.manual_seed(7)
    B, O = 2, cfg.observation_steps
    torch.manual_seed(7)
    features = torch.randn(B, 2 * O, 6) * torch.tensor([3000., 3000., 500., 50., 50., 10.])
    valid = torch.ones(B, 2 * O, dtype=torch.bool)
    edges, _ = engagement_edges(features, valid, cfg, 1.0)
    pair = features[:, :O] - features[:, O:2 * O]
    sep = pair[..., :3].norm(dim=-1).clamp_min(1e-3)
    eL = pair[..., :3] / sep[..., None]
    # Eq. (2): V_L = V_M e_M . e_L - V_T e_T . e_L = -(v_T - v_M) . e_L
    v_l = -(pair[..., 3:] * eL).sum(-1)
    v_ref, a_ref = cfg.max_speed_mps, cfg.engagement_overload_g * 9.81
    # Source node i -> any destination: slots 9 (V_L) and 10 (dV_L/dt).
    for i in (0, 1, O - 1, O, O + 1):
        time = i % O
        assert edges[0, i, :, 9].unique().numel() == 1      # broadcast over j
        got = edges[0, i, 0, 9].item() * v_ref
        assert got == pytest.approx(v_l[0, time].item(), rel=1e-4, abs=1e-3)
        expected_rate = 0. if time == 0 else (v_l[0, time] - v_l[0, time - 1]).item() / cfg.dt
        assert edges[0, i, 0, 10].item() * a_ref == pytest.approx(expected_rate, abs=1e-3)
        # Both entities share the same pairing value at the same time index.
        assert edges[0, i, 0, 9].item() == pytest.approx(edges[0, i % O, 0, 9].item(), abs=1e-6)


def test_edge_types_mutually_exclusive_and_self_allowed():
    cfg = military()
    torch.manual_seed(1)
    B, O = 2, 6
    features = torch.randn(B, 2 * O, 6) * torch.tensor([3000., 3000., 500., 50., 50., 10.])
    valid = torch.ones(B, 2 * O, dtype=torch.bool)
    edges, allowed = engagement_edges(features, valid, cfg, 1.0)
    temporal = edges[..., 18] > .5
    spatial = edges[..., 19] > .5
    counter = edges[..., 20] > .5
    assert not (temporal & spatial).any()
    assert not (temporal & counter).any()
    assert not (spatial & counter).any()
    assert allowed.diagonal(dim1=1, dim2=2).all()
    assert counter.any()  # adversarial edges must appear at engagement ranges


def test_model_trains_with_engagement_edges(tmp_path):
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
    assert model.graph[0].edge_context is not None
    assert model.graph[0].edge_context.weight.grad.abs().sum() > 0
    for weights in output["attention"]:
        torch.testing.assert_close(weights.sum(-1), torch.ones(1, cfg.heads, 2 * cfg.observation_steps))
    # unavailable defender rows cannot influence the forecast
    model.eval()
    masked = batch["context_mask"].clone()
    masked[:, :, 3] = False
    reference = model(batch["observed"], batch["context"], masked)
    corrupt = batch["context"].clone()
    corrupt[:, :, 3] = float("nan")
    changed = model(batch["observed"], corrupt, masked)
    torch.testing.assert_close(reference["trajectory"], changed["trajectory"], rtol=0, atol=0)
    torch.testing.assert_close(reference["logits"], changed["logits"], rtol=0, atol=0)


def test_civilian_edges_still_default_when_flag_off(tmp_path):
    cfg = military(engagement_edges=False, historical_edge_features=True,
                   attention_supervision=False)
    data = tmp_path / "e.npz"
    simulate_engagements(data, scenes=6, steps=60, dt=.1, seed=3)
    tracks = load_tracks(data, .1, 10)
    windows = Windows(tracks, split_tracks(tracks, .7, .15, 42)["train"], cfg)
    model = Forecaster(cfg, fit_normalizer(windows))
    output = model(windows[0]["observed"][None], windows[0]["context"][None],
                   windows[0]["context_mask"][None])
    assert model.graph[0].edge_context.weight.shape == (cfg.heads, 7)
    assert torch.isfinite(output["trajectory"]).all()


def test_node_gate_consumes_the_engagement_descriptor(tmp_path):
    """Eq. (20): m_j = sigmoid(f_n(h_j, u)) - the node gate must read u.

    The military graph layers therefore carry a node_context projection that
    receives the per-node LOS engagement descriptor, and attention supervision
    drives gradients through it; the civilian graph (no engagement descriptor)
    has no such projection.
    """
    cfg = military()
    data = tmp_path / "e.npz"
    simulate_engagements(data, scenes=6, steps=60, dt=.1, seed=11)
    tracks = load_tracks(data, .1, 10)
    windows = Windows(tracks, split_tracks(tracks, .7, .15, 42)["train"], cfg)
    sample = windows[0]
    keys = ("observed", "context", "context_mask", "future", "label", "engagement")
    batch = {k: sample[k][None] for k in keys if k in sample}
    assert "engagement" in batch, "military windows must carry the LOS descriptor"
    model = Forecaster(cfg, fit_normalizer(windows))
    assert model.graph[0].node_context is not None
    TrajectoryLearner(cfg, model, torch.device("cpu")).update(batch)
    assert model.graph[0].node_context.weight.grad is not None
    assert model.graph[0].node_context.weight.grad.abs().sum() > 0
    civilian = Forecaster(military(engagement_edges=False, attention_supervision=False,
                                   context_entities=0, static_state_branch=False,
                                   decoder_type="parallel"))
    assert civilian.graph[0].node_context is None


def test_candidate_branches_converge_on_terminal_nodes():
    """Decision D1 (G=1): candidate futures branch from the observation END.

    The trailing candidate block indexes alternative futures (not time), so an
    alive candidate row must connect only to the terminal target and terminal
    defender nodes, never to the same-index history row or to sibling branches;
    cropped/padding candidate rows keep only their self edge.
    """
    cfg = military()
    O = cfg.observation_steps
    target = torch.zeros(1, O, 6)
    target[..., 0] = torch.arange(O) * 300. * cfg.dt
    target[..., 3] = 300.
    defender = target.clone()
    defender[..., 1] += 2000.
    alive = 3
    candidate = torch.zeros(1, O, 6)
    candidate[0, :alive, 0] = 3000. + torch.arange(alive) * 50.
    candidate[0, :alive, 3] = 300.
    features = torch.cat((target, defender, candidate), 1)
    valid = torch.zeros(1, 3 * O, dtype=torch.bool)
    valid[:, :2 * O] = True
    valid[:, 2 * O:2 * O + alive] = True
    edges, allowed = engagement_edges(features, valid, cfg, 1.0)
    term_target, term_defender = O - 1, 2 * O - 1
    # An alive candidate sees only the two terminal history nodes (plus itself).
    history_neighbors = allowed[0, 2 * O, :2 * O].nonzero().flatten().tolist()
    assert sorted(history_neighbors) == sorted([term_target, term_defender])
    # No candidate-to-candidate edges except the self loop.
    cand_block = allowed[0, 2 * O:3 * O, 2 * O:3 * O]
    assert cand_block.sum() == O                              # only diagonal self edges
    # The terminal target aggregates every alive candidate and no dead padding.
    assert allowed[0, term_target, 2 * O:2 * O + alive].all()
    assert not allowed[0, term_target, 2 * O + alive]
    # A non-terminal history node does not reach into the candidate block.
    assert not allowed[0, 0, 2 * O:3 * O].any()
    # Candidate-terminal links are typed as countermeasure (adversarial) edges.
    assert edges[0, term_target, 2 * O, 20] > .5


def test_overload_reachability_blocks_unreachable_cross_entity_links():
    """A7: IV-B "imposes hard blocking on links deemed unreachable".

    A cross-entity link whose relative velocity would need a normal overload
    beyond aT_max to align with the pair LOS within the time lag must be
    removed from the adjacency; disabling the switch restores it, and
    same-entity links plus self edges are never blocked by this rule.
    """
    O = 6
    cfg = military(overload_reachability=True, interaction_time_radius=3,
                   interaction_radius_m=8000.)
    cfg_off = military(overload_reachability=False, interaction_time_radius=3,
                       interaction_radius_m=8000.)
    # Crossing geometry: the target flies +x at 300 m/s while the defender
    # crosses from 2 km to the left at 400 m/s.  The relative velocity is far
    # from the pair LOS, so lag>0 cross-entity links need a lateral overload
    # |v_rel| * theta / lag >> 9 g and must be hard-blocked.
    target = torch.zeros(1, O, 6)
    target[..., 0] = torch.arange(O) * 300. * cfg.dt
    target[..., 3] = 300.
    defender = target.clone()
    defender[..., 1] += 2000.
    defender[..., 3] = 0.
    defender[..., 4] = -400.
    features = torch.cat((target, defender), 1)
    valid = torch.ones(1, 2 * O, dtype=torch.bool)
    _, allowed_on = engagement_edges(features, valid, cfg, 1.0)
    _, allowed_off = engagement_edges(features, valid, cfg_off, 1.0)
    cross = torch.zeros(1, 2 * O, 2 * O, dtype=torch.bool)
    cross[:, O:, :O] = True
    cross[:, :O, O:] = True
    on_count = (allowed_on & cross).sum()
    off_count = (allowed_off & cross).sum()
    assert off_count > on_count > 0, "unreachable cross-entity links must be blocked"
    # the surviving cross links are exactly the zero-lag spatial edges
    lag = (torch.arange(2 * O) % O)[None, :] - (torch.arange(2 * O) % O)[:, None]
    survivors = (allowed_on & cross)[0]
    assert (lag[survivors] == 0).all()
    # same-entity links are untouched by the overload rule
    same = ~cross
    assert ((allowed_off & same) & ~(allowed_on & same)).sum() == 0
    # every node keeps its self edge
    assert allowed_on[0].diagonal().all()
