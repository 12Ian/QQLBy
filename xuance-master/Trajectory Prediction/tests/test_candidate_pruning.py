"""Smoke tests for candidate branch expansion and the Table I cropper (S4)."""
import numpy as np
import pytest
import torch

from trajectory_prediction.candidate_pruning import anchor_directions, candidate_branch
from trajectory_prediction.config import Config
from trajectory_prediction.data import Windows, fit_normalizer, load_tracks, split_tracks
from trajectory_prediction.engagement_scenarios import simulate_engagements
from trajectory_prediction.learner import TrajectoryLearner
from trajectory_prediction.model import Forecaster


def cropping(**updates):
    values = dict(observation_steps=10, prediction_steps=5, dt=.1, stride=5,
                  lstm_hidden=8, projection_dim=8, graph_dim=12, heads=3,
                  graph_layers=1, batch_size=8, epochs=1, device="cpu", cpu_threads=1,
                  context_entities=1, static_state_branch=True,
                  decoder_type="autoregressive_gru", engagement_edges=True,
                  max_speed_mps=450., temporal_radius=2, interaction_time_radius=1,
                  interaction_radius_m=50., attention_supervision=True,
                  military_teachers=True, teacher_corridor_radial_m=600.,
                  teacher_corridor_normal_m=300., candidate_pruning=True,
                  candidate_anchors=8, crop_warmup_epochs=2)
    values.update(updates)
    return Config(**values).validate()


def _scene(speed=300., heading=(1., 0., 0.), steps=10, dt=.1):
    """Straight, level flight history ending at the origin."""
    velocity = torch.tensor(heading, dtype=torch.float32) * speed
    positions = torch.arange(-steps + 1, 1, dtype=torch.float32)[:, None] * velocity * dt
    observed = torch.cat((positions, velocity.expand(steps, 3)), -1)[None]
    return observed


def _defender(position, velocity, steps=10):
    state = torch.cat((torch.tensor(position, dtype=torch.float32),
                       torch.tensor(velocity, dtype=torch.float32)))
    return state.expand(1, 1, steps, 6).clone(), torch.ones(1, 1, steps, dtype=torch.bool)


def test_anchor_directions_are_unit_and_deterministic():
    anchors = anchor_directions(16)
    assert anchors.shape == (16, 3)
    assert torch.allclose(anchors.norm(dim=-1), torch.ones(16), atol=1e-5)
    assert torch.allclose(anchors, anchor_directions(16))
    assert anchors.unique(dim=0).shape[0] == 16


def test_cene_prunes_turns_beyond_the_overload_limit():
    cfg = cropping(prediction_steps=5, terrain_enabled=False)
    observed = _scene()
    anchors = anchor_directions(8)
    short = candidate_branch(observed, None, None, cfg, anchors)
    long = candidate_branch(observed, None, None, cropping(prediction_steps=60), anchors)
    # 0.5 s horizon: only near-straight turns fit inside 9 g; 6 s admits ~100 deg.
    assert short["keep"].sum() < long["keep"].sum() <= 8
    assert short["keep"].sum() >= 1                       # the straight branch survives
    # Eq. (24) gates the TIME-AVERAGED acceleration increment, not the
    # instantaneous turn load: for a constant-rate coordinated turn it equals
    # v*theta^2/T^2 (the instantaneous a_req = v*theta/T is the soft cost /
    # Eq. (45) teacher, deliberately NOT a hard gate - see fidelity A-05).
    speed, horizon = 300., cfg.prediction_steps * cfg.dt
    # Anchors live in the heading frame: +z is straight ahead, so the turn angle
    # of anchor i is arccos(a_z).
    theta = anchors[:, 2].clamp(-1., 1.).acos()
    mean_increment = speed * theta.pow(2) / horizon ** 2
    expected = mean_increment <= cfg.engagement_overload_g * 9.81
    assert torch.equal(short["keep"][0, :8], expected)


def test_cesc_prunes_closing_candidates_and_bypasses_when_blind():
    cfg = cropping(prediction_steps=60)          # wide horizon: Cene keeps many
    observed = _scene()
    anchors = anchor_directions(8)
    ahead, mask = _defender((2000., 0., 0.), (-400., 0., 0.))
    behind, behind_mask = _defender((-2000., 0., 0.), (400., 0., 0.))
    visible = candidate_branch(observed, ahead, mask, cfg, anchors)
    blind = candidate_branch(observed, behind, behind_mask, cfg, anchors)
    energy = candidate_branch(observed, None, None, cfg, anchors)["keep"]
    # Defender in front and closing: Cesc removes closing branches.
    assert visible["keep"].sum() < energy.sum()
    # ... but the cropper never empties the branch set (D1 safety net), and the
    # branch it keeps is the cheapest one, i.e. the constant-velocity candidate.
    assert visible["keep"].sum() >= 1
    # With an unreachable escape threshold every branch violates Cesc, and the
    # safety net keeps exactly the cheapest (constant-velocity) branch.
    starved = candidate_branch(observed, ahead, mask, cropping(prediction_steps=60,
                                                              escape_rate_min_mps=1e6), anchors)
    assert int(starved["keep"].sum()) == 1
    assert starved["keep"][0, starved["cost"][0, :8].argmin()]
    # Defender behind, outside the FOV: Cesc collapses to 1 (Table I step 4).
    assert torch.equal(blind["keep"], energy)


def test_cenv_altitude_band_prunes_when_terrain_enabled():
    anchors = anchor_directions(8)
    free = cropping(prediction_steps=60, terrain_enabled=False)
    band = cropping(prediction_steps=60, terrain_enabled=True,
                    terrain_altitude_min_m=-50., terrain_altitude_max_m=50.)
    observed = _scene()
    assert candidate_branch(observed, None, None, band, anchors)["keep"].sum() < \
        candidate_branch(observed, None, None, free, anchors)["keep"].sum()
    # Everything above/below the band is dropped, nothing inside it is.
    branch = candidate_branch(observed, None, None, band, anchors)
    altitude = branch["absolute"][0, :8, 2]
    assert torch.equal(branch["keep"][0, :8],
                       (altitude >= -50.) & (altitude <= 50.))


def test_model_attaches_candidate_block_and_masks_attention():
    cfg = cropping()
    anchors = anchor_directions(cfg.candidate_anchors)
    observed, context, context_mask = _scene(steps=10), *_defender((2000., 0., 0.), (-400., 0., 0.))
    model = Forecaster(cfg, {"mean": torch.zeros(6), "std": torch.ones(6)})
    model.set_crop_progress(1.0)
    output = model(observed, context, context_mask)
    block = cfg.observation_steps
    assert output["attention"][-1].shape[-1] == 3 * block
    assert torch.isfinite(output["trajectory"]).all()
    attention = output["attention"][-1]
    assert torch.allclose(attention.sum(-1), torch.ones_like(attention.sum(-1)), atol=1e-5)
    branch = candidate_branch(observed, context, context_mask, cfg, anchors)
    cropped = (~branch["keep"][0]).nonzero().flatten() + 2 * block
    alive = branch["keep"][0].nonzero().flatten() + 2 * block
    # Cropped candidates receive no message from the historical nodes (only the
    # preserved self edge of the candidate row itself remains).
    assert attention[..., :2 * block, cropped].abs().sum() == 0
    # Surviving candidates do receive attention, so the branch is not inert.
    assert alive.numel() and attention[..., :2 * block, alive].abs().sum() > 0


def test_constant_turn_arc_matches_chord_endpoint_and_keeps_speed():
    """Guard for _constant_turn_arc: the closed-form integral of the rotating
    velocity must land exactly on the analytic chord displacement

        d = (v T / theta) * [e3 sin(theta) + (n x e3)(1 - cos(theta))]
          = 2 (v T / theta) sin(theta/2) * bisector

    and every sampled arc point must keep the constant speed v (a sampling
    discretisation of a coordinated turn, not a re-interpolation)."""
    from trajectory_prediction.candidate_pruning import _constant_turn_arc

    torch.manual_seed(7)
    velocity = torch.randn(2, 3) * 40.
    velocity = velocity / velocity.norm(dim=-1, keepdim=True) * 300.
    position = torch.randn(2, 3) * 1000.
    direction = torch.nn.functional.normalize(torch.randn(2, 4, 3), dim=-1)   # [B,A,3]
    horizon, samples = 6.0, 12
    arc_pos, arc_vel, theta = _constant_turn_arc(position, velocity, direction, horizon, samples)

    speed = velocity.norm(dim=-1)
    assert torch.allclose(arc_vel.norm(dim=-1), speed[:, None, None].expand_as(arc_vel.norm(dim=-1)),
                          rtol=1e-4)
    # Independent chord reference: rotate e3 about n by theta (Rodrigues),
    # evaluated per (batch, anchor).
    e3 = velocity / speed[:, None]
    axis = torch.linalg.cross(e3[:, None].expand_as(direction), direction, dim=-1)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    perp = torch.linalg.cross(axis, e3[:, None].expand_as(axis), dim=-1)
    chord = ((speed[:, None] * horizon) / theta.clamp_min(1e-8))[..., None] * (
        e3[:, None] * torch.sin(theta)[..., None] + perp * (1. - torch.cos(theta))[..., None])
    assert torch.allclose(arc_pos[:, :, -1] - position[:, None], chord, rtol=1e-4, atol=1e-3)
    # The FIRST arc sample (s = 1/S) also satisfies the closed form at
    # phi = theta / S, guarding the interior points against re-interpolation.
    phi_first = (theta / samples)[..., None]
    first = ((speed[:, None] * horizon) / theta.clamp_min(1e-8))[..., None] * (
        e3[:, None] * torch.sin(phi_first) + perp * (1. - torch.cos(phi_first)))
    assert torch.allclose(arc_pos[:, :, 0] - position[:, None], first, rtol=1e-4, atol=1e-3)


def test_cesc_uses_window_average_range_rate_not_endpoint():
    """Eq. (27): cesc = 1[(1/T) sum_k rdot_k >= rdot_min], a window mean.

    A faster head-on defender that the straight branch passes LATE in the
    window is closing for most of the arc (negative mean) even though the
    endpoint range rate is already positive; the window rule rejects it.  An
    early pass opens for most of the window and is accepted.  An endpoint-only
    rule would accept both.  The assertion reads the pre-safety-net Cesc factor
    (the never-empty safety net otherwise revives the cheapest straight branch
    when a faster defender makes every branch close over the horizon).
    """
    cfg = cropping(prediction_steps=60)
    observed = _scene()
    anchors = anchor_directions(8)
    relative_sum = 300. + 400.
    horizon = cfg.prediction_steps * cfg.dt
    late_pass = relative_sum * 0.75 * horizon             # pass at t = 0.75 T
    early_pass = relative_sum * 0.25 * horizon            # pass at t = 0.25 T
    late, late_mask = _defender((late_pass, 100., 0.), (-400., 0., 0.))
    early, early_mask = _defender((early_pass, 100., 0.), (-400., 0., 0.))
    # Anchor 0 is the straight branch (canonical +z = current heading).
    late_escape = candidate_branch(observed, late, late_mask, cfg, anchors)["escape"]
    early_escape = candidate_branch(observed, early, early_mask, cfg, anchors)["escape"]
    assert not bool(late_escape[0, 0])                    # window mean rdot < 0
    assert bool(early_escape[0, 0])                       # window mean rdot > 0
    # Sanity: the late-pass endpoint itself is already opening, proving the
    # rejection comes from time-averaging rather than the endpoint sign.
    t_end = horizon
    endpoint_los_x = 300. * t_end - (late_pass - 400. * t_end)
    assert endpoint_los_x > 0


def test_cenv_checks_altitude_at_every_arc_point():
    """Eq. (28): cenv = 1[g(y_k) not in F, forall k] - every arc point, not the
    endpoint alone.  A 90 deg pull-up climbs monotonically; the hard mask must
    match an independent all-arc-point altitude-band evaluation."""
    from trajectory_prediction.candidate_pruning import _constant_turn_arc
    cfg = cropping(prediction_steps=60, terrain_enabled=True,
                   terrain_altitude_min_m=-5000., terrain_altitude_max_m=2000.)
    observed = _scene()
    anchors = torch.tensor([[0., 0., 1.], [1., 0., 0.]])     # straight / 90 deg pull-up
    horizon = cfg.prediction_steps * cfg.dt
    position, velocity = observed[:, -1, :3], observed[:, -1, 3:]
    # Anchor [1,0,0] maps to world +z for a horizontal heading (a vertical
    # pull-up); independently sample its arc to evaluate the altitude band.
    pull_up = torch.tensor([[[0., 0., 1.]]])
    arc_pos, _, _ = _constant_turn_arc(position, velocity, pull_up, horizon,
                                       cfg.candidate_arc_samples)
    for high in (100., 400., 900., 1500.):
        tuned = cropping(prediction_steps=60, terrain_enabled=True,
                         terrain_altitude_min_m=-5000., terrain_altitude_max_m=high)
        keep = candidate_branch(observed, None, None, tuned, anchors)["keep"][0]
        expected_climb = bool(((arc_pos[0, 0, :, 2] >= -5000.) & (arc_pos[0, 0, :, 2] <= high)).all())
        assert bool(keep[1]) == expected_climb
    assert bool(keep[0])                                    # straight/level survives


def test_soft_crop_ramps_in_with_crop_progress():
    cfg = cropping()
    observed, context, context_mask = _scene(steps=10), *_defender((-2000., 0., 0.), (400., 0., 0.))
    model = Forecaster(cfg, {"mean": torch.zeros(6), "std": torch.ones(6)})
    model.eval()
    model.set_crop_progress(0.0)
    early = model(observed, context, context_mask)["attention"][-1].clone()
    model.set_crop_progress(1.0)
    late = model(observed, context, context_mask)["attention"][-1].clone()
    assert not torch.allclose(early, late)
    assert torch.isfinite(late).all()


def test_training_step_with_candidates_is_finite(tmp_path):
    cfg = cropping()
    data = tmp_path / "e.npz"
    simulate_engagements(data, scenes=6, steps=60, dt=.1, seed=23)
    tracks = load_tracks(data, cfg.dt, cfg.observation_steps + cfg.prediction_steps)
    windows = Windows(tracks, split_tracks(tracks, .7, .15, 42)["train"], cfg)
    sample = windows[0]
    batch = {key: sample[key][None] for key in ("observed", "context", "context_mask", "future", "label")}
    model = Forecaster(cfg, fit_normalizer(windows))
    model.set_crop_progress(1.0)
    output, loss = TrajectoryLearner(cfg, model, torch.device("cpu")).update(batch)
    assert torch.isfinite(loss["loss"])
    assert output["trajectory"].shape == (1, cfg.prediction_steps, 3)
    assert np.isfinite(float(loss["loss"]))
