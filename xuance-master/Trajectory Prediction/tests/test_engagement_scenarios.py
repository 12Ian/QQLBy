"""Smoke tests for the military intercept-engagement scenario generator.

Covers the S1 milestones from MILITARY_PORT_PLAN.md: Eq.(1)-(2) kinematics,
Table IV sampling bounds, 9g/15g overload saturation, the r <= (2/3) r0
evasion trigger, FOV visibility descriptors, label ranges, reproducibility,
and schema compatibility with the civilian data pipeline.
"""
import numpy as np
import pytest

from trajectory_prediction.data import load_tracks
from trajectory_prediction.engagement_scenarios import (
    G0, simulate_engagements, los_state, fov_visible, overload_label,
    OVERLOAD_LEVELS)


def test_los_state_matches_hand_computation():
    rT = np.array([1000., 0., 0.])
    vT = np.array([0., 250., 0.])
    rM = np.zeros(3)
    vM = np.array([300., 0., 0.])
    r, rdot, eL, eLdot, V_L = los_state(rT, vT, rM, vM)
    assert r == pytest.approx(1000.)
    assert eL[0] == pytest.approx(1.) and abs(eL[1]) < 1e-9
    # rdot = (vT - vM) . eL = (0 - 300) * 1 = -300 (closing)
    assert rdot == pytest.approx(-300.)
    # V_L = vM.eL - vT.eL = 300 - 0 = 300
    assert V_L == pytest.approx(300.)
    # eLdot: v_rel perpendicular part divided by r
    v_rel = vT - vM
    expected = np.cross(np.cross(eL, v_rel), eL) / r
    np.testing.assert_allclose(eLdot, expected, atol=1e-12)


def test_fov_visibility_cone():
    vT = np.array([300., 0., 0.])
    # eL points from defender M to target T; the defender sits at -eL from
    # the target, so "defender ahead-left 30 deg" means eL = -(cos30, sin30, 0).
    ahead_30 = -np.array([np.cos(np.radians(30.)), np.sin(np.radians(30.)), 0.])
    ahead_80 = -np.array([np.cos(np.radians(80.)), np.sin(np.radians(80.)), 0.])
    behind = np.array([np.cos(np.radians(30.)), np.sin(np.radians(30.)), 0.])
    assert fov_visible(vT, ahead_30) is True     # 30 deg < 60 deg half-angle
    assert fov_visible(vT, ahead_80) is False    # 80 deg > 60 deg half-angle
    assert fov_visible(vT, behind) is False      # defender behind the target


def test_overload_label_boundaries():
    # Round-to-nearest even-level bins (levels spaced 2g, half-width 0.5g).
    assert overload_label(np.array([0.4 * G0])) == 5    # nearest level 0 g
    assert overload_label(np.array([2.4 * G0])) == 6    # nearest level 2 g
    assert overload_label(np.array([2.6 * G0])) == 6    # still nearest 2 g
    assert overload_label(np.array([-10.4 * G0])) == 0  # clamps to -10 g


def test_overload_label_censored_ends_reaches_extreme_levels():
    # Round-2 fix (audit item 3): with aT_max = 9 g the strict (9.5, 10.5] bin
    # is physically unreachable, so the +/-10 g levels stay empty.  The censored
    # ends widen them to (-inf, -8.5] / [8.5, +inf) and 9 g saturations map to
    # the extreme bins instead of falling through to nearest-level.
    assert overload_label(np.array([9. * G0]), censored_ends=True) == 10
    assert overload_label(np.array([8.6 * G0]), censored_ends=True) == 10
    assert overload_label(np.array([8.4 * G0]), censored_ends=True) == 9   # 8 g bin
    assert overload_label(np.array([-9. * G0]), censored_ends=True) == 0
    assert overload_label(np.array([0.4 * G0]), censored_ends=True) == 5   # interior unchanged
    assert overload_label(np.array([2.4 * G0]), censored_ends=True) == 6   # interior unchanged


def test_simulate_engagements_censored_flag_populates_extreme_bins(tmp_path):
    # End-to-end guard for the CLI --censored-overload-ends switch: with
    # saturated evasion (1,1) the window means sit near 9 g, which the strict
    # (9.5, 10.5] bin can never reach and nearest-level rounds down to +8 g;
    # the censored ends map them to the +10 g level instead.
    strict = tmp_path / "strict.npz"
    censored = tmp_path / "censored.npz"
    kwargs = dict(scenes=12, steps=200, dt=.1, seed=7, evasion_overload_range=(1., 1.))
    m1 = simulate_engagements(strict, **kwargs)
    m2 = simulate_engagements(censored, censored_overload_ends=True, **kwargs)
    assert m1["censored_overload_ends"] is False
    assert m2["censored_overload_ends"] is True
    assert m1["sha256"] != m2["sha256"]  # the label series must actually change
    strict_ov = np.load(strict)["overload_labels"].reshape(-1)
    ov = np.load(censored)["overload_labels"].reshape(-1)
    assert (strict_ov == 10).sum() == 0   # strict bin unreachable below 9.5 g
    assert (ov == 10).sum() > 0           # censored: saturated windows -> +10 g


def test_simulate_engagements_evasion_overload_range_dial(tmp_path):
    # Round-2 calibration (audit item 1): the evasion-overload range must be a
    # real dial - the saturation point is echoed in the metadata so runs are
    # auditable, and (1,1) differs from the default random draw.
    saturated = tmp_path / "sat.npz"
    m = simulate_engagements(saturated, scenes=4, steps=120, dt=.1, seed=11,
                             evasion_overload_range=(1., 1.))
    assert m["evasion_overload_range"] == [1., 1.]
    default = simulate_engagements(tmp_path / "def.npz", scenes=4, steps=120, dt=.1, seed=11)
    assert default["evasion_overload_range"] == [0.5, 1.0]
    assert m["sha256"] != default["sha256"]


def test_engagement_simulation_physical_limits_and_schema(tmp_path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    meta = simulate_engagements(a, scenes=12, steps=200, dt=.1, seed=7)
    assert meta["sha256"] == simulate_engagements(b, scenes=12, steps=200, dt=.1, seed=7)["sha256"]
    assert meta["domain"] == "military_intercept_engagement"
    assert meta["table_iv"]["r0_range_km"] == [4., 8.]
    with np.load(a) as z:
        states = z["states"].reshape(12, 200, 6)
        ctx = z["context_states"].reshape(12, 200, 1, 6)
        eng = z["engagement_states"].reshape(12, 200, 9)
        ev = z["evasion_mask"].reshape(12, 200)
        sem = z["labels"].reshape(12, 200)
        ov = z["overload_labels"].reshape(12, 200)
        assert np.linalg.norm(states[..., 3:], axis=-1).max() <= 350. + 1e-3
        assert np.linalg.norm(ctx[..., 3:], axis=-1).max() <= 450. + 1e-3
        acc = np.linalg.norm(np.diff(states[..., 3:], axis=1) / .1, axis=-1)
        assert acc.max() <= 9 * G0 + 1e-3
        accM = np.linalg.norm(np.diff(ctx[..., 3:], axis=1) / .1, axis=-1)
        assert accM.max() <= 15 * G0 + 1e-3
        r = eng[..., 0] * 1e3
        for i in range(12):
            if ev[i].any():
                first = int(np.flatnonzero(ev[i])[0])
                assert r[i, first] <= (2. / 3.) * r[i, 0] + 1e-6
        assert sem.min() >= 0 and sem.max() <= 8
        assert ov.min() >= 0 and ov.max() <= 10
        sep = np.linalg.norm(states[:, :, :3] - ctx[:, :, 0, :3], axis=-1)
        np.testing.assert_allclose(sep, r, atol=1.)
        assert ev.any()  # evasion must trigger in this horizon
    tracks = load_tracks(a, .1, 20)
    assert len(tracks) == 12
    assert tracks[0].states.shape == (200, 6)
    assert tracks[0].context.shape == (200, 1, 6)


def test_engagement_scene_split_has_no_leakage(tmp_path):
    from trajectory_prediction.data import Windows, split_tracks, validate_split
    a = tmp_path / "a.npz"
    simulate_engagements(a, scenes=9, steps=120, dt=.1, seed=3)
    tracks = load_tracks(a, .1, 20)
    splits = split_tracks(tracks, .7, .15, 11)
    validate_split(tracks, splits)
    assert not set(splits["train"]) & set(splits["test"])


def test_npz_stores_overload_series_and_censored_flag(tmp_path):
    # F1: the raw signed normal-overload series and the bin convention must be
    # persisted so data.window_intention_label can quantize the WINDOW MEAN
    # (paper IV-A) with the same extremes the generator used.
    strict = tmp_path / "strict.npz"
    simulate_engagements(strict, scenes=4, steps=120, dt=.5, sample_stride=5, seed=5)
    with np.load(strict) as z:
        assert "overload_series" in z.files
        assert "overload_censored" in z.files
        assert not bool(z["overload_censored"])
        series = z["overload_series"]
        assert series.shape == z["labels"].shape
        assert np.isfinite(series).all()
    censored = tmp_path / "cens.npz"
    meta = simulate_engagements(censored, scenes=4, steps=120, dt=.5, sample_stride=5, seed=5,
                                censored_overload_ends=True)
    assert meta["censored_overload_ends"] is True
    with np.load(censored) as z:
        assert bool(z["overload_censored"])


def test_overload11_window_label_is_window_mean_quantization(tmp_path):
    # F1 (paper IV-A): the eleven-level window label must equal the
    # quantization of the observation-window MEAN of the raw overload series,
    # not a majority vote over per-step trailing-window bins.
    from trajectory_prediction.data import load_tracks, window_intention_label
    path = tmp_path / "f1.npz"
    simulate_engagements(path, scenes=6, steps=120, dt=.5, sample_stride=5, seed=9)
    tracks = load_tracks(path, .5, 20)
    for track in tracks:
        assert track.overload_series is not None
        for start in range(0, 40, 10):
            label = window_intention_label(track, start, start + 20, 11, "overload11")
            mean_g = float(np.mean(track.overload_series[start:start + 20])) / G0
            levels = np.array([-10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10.])
            offset = mean_g - levels
            inside = (offset > -.5) & (offset <= .5)
            expected = int(np.flatnonzero(inside)[-1]) if inside.any() else int(np.argmin(np.abs(offset)))
            assert label == expected


def test_primitive_duration_range_dial(tmp_path):
    # F2: segment duration is a real dial (paper leaves the bounds open) and
    # the default (2, 6) s reproduces the previous hard-coded sampler exactly.
    base = tmp_path / "base.npz"
    short = tmp_path / "short.npz"
    meta_base = simulate_engagements(base, scenes=6, steps=120, dt=.5, sample_stride=5, seed=11)
    meta_short = simulate_engagements(short, scenes=6, steps=120, dt=.5, sample_stride=5, seed=11,
                                      primitive_duration_range=(1.0, 3.0))
    assert meta_base["primitive_duration_range_s"] == [2.0, 6.0]
    assert meta_short["primitive_duration_range_s"] == [1.0, 3.0]
    with np.load(base) as a, np.load(short) as b:
        assert not np.array_equal(a["states"], b["states"])
    with pytest.raises(ValueError):
        simulate_engagements(tmp_path / "bad.npz", scenes=6, steps=120, dt=.5,
                             primitive_duration_range=(6.0, 2.0))


def test_evasion_segment_seconds_dial(tmp_path):
    # F2: evasion segment length is parameterized (default 4 s = old value).
    # The evasion trigger fires early with a short r0, but C9 termination
    # (intercept/certain-miss) often ends the engagement within 4-10 s of the
    # trigger - i.e. within one or two segments either way - so the RNG draws
    # can still coincide.  Disable miss termination and keep the horizon long
    # so the post-trigger evasion phase spans many segments and the resample
    # cadence (4 s vs 2 s) provably changes the commanded overloads.
    kwargs = dict(scenes=6, steps=900, dt=.1, seed=13, r0_range=(900., 1100.),
                  miss_grow_steps=10 ** 9)
    base = tmp_path / "b.npz"
    fast = tmp_path / "f.npz"
    simulate_engagements(base, **kwargs)
    meta = simulate_engagements(fast, evasion_segment_seconds=2.0, **kwargs)
    assert meta["evasion_segment_seconds"] == 2.0
    with np.load(base) as a, np.load(fast) as b:
        assert a["evasion_mask"].any()          # evasion really fired here
        assert not np.array_equal(a["overload_labels"], b["overload_labels"])
