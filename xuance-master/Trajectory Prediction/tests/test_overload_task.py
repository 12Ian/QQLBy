"""Smoke tests for the 11-level overload-intensity intention task (S5, IV-A)."""
import numpy as np
import pytest
import torch

from trajectory_prediction.config import Config
from trajectory_prediction.data import (OVERLOAD_NAMES, Track, Windows, fit_normalizer,
                                        load_tracks, split_tracks, window_intention_label)
from trajectory_prediction.engagement_scenarios import simulate_engagements
from trajectory_prediction.learner import TrajectoryLearner
from trajectory_prediction.model import Forecaster


def engagement(**updates):
    values = dict(observation_steps=6, prediction_steps=4, stride=5, dt=.1, lstm_hidden=8,
                  projection_dim=8, graph_dim=12, heads=3, graph_layers=1,
                  batch_size=8, epochs=1, device="cpu", cpu_threads=1,
                  context_entities=1, static_state_branch=True,
                  decoder_type="autoregressive_gru", engagement_edges=True,
                  max_speed_mps=450., temporal_radius=2, interaction_time_radius=1,
                  interaction_radius_m=50., attention_supervision=True)
    values.update(updates)
    return Config(**values).validate()


def _write_dataset(path, steps=30, dt=.1, semantic=0, overload=7, with_overload=True):
    """Three scenes with constant semantic and overload labels."""
    ids, scenes, times, states, context, mask, labels, olabels = [], [], [], [], [], [], [], []
    for scene in range(3):
        state = np.zeros((steps, 6), dtype=np.float32)
        state[:, 3] = 300.
        ids += [f"scene_{scene}_agent_0"] * steps
        scenes += [f"scene_{scene}"] * steps
        times.append(np.arange(steps) * dt)
        states.append(state)
        context.append(state[:, None, :])
        mask.append(np.ones((steps, 1), dtype=bool))
        labels.append(np.full(steps, semantic, dtype=np.int64))
        olabels.append(np.full(steps, overload, dtype=np.int64))
    arrays = dict(track_id=np.asarray(ids), scene_id=np.asarray(scenes),
                  scenario=np.asarray(["engagement"] * len(ids)),
                  time=np.concatenate(times), states=np.concatenate(states),
                  context_states=np.concatenate(context), context_mask=np.concatenate(mask),
                  labels=np.concatenate(labels))
    if with_overload:
        arrays["overload_labels"] = np.concatenate(olabels)
    np.savez_compressed(path, **arrays)


def test_load_tracks_exposes_overload_labels(tmp_path):
    data = tmp_path / "e.npz"
    simulate_engagements(data, scenes=4, steps=40, dt=.1, seed=5)
    tracks = load_tracks(data, .1, 10)
    assert tracks and all(t.overload_labels is not None for t in tracks)
    for track in tracks:
        assert track.overload_labels.shape == track.labels.shape
        assert track.overload_labels.min() >= 0 and track.overload_labels.max() <= 10
        assert len(OVERLOAD_NAMES) == 11


def test_window_label_follows_intention_task(tmp_path):
    data = tmp_path / "label.npz"
    _write_dataset(data, semantic=3, overload=9)
    tracks = load_tracks(data, .1, 10)
    ids = [t.track_id for t in tracks]
    semantic = Windows(tracks, ids, engagement(intention_task="semantic9"))
    overload = Windows(tracks, ids, engagement(intention_task="overload11", intention_classes=11))
    assert int(semantic[0]["label"]) == 3
    assert int(overload[0]["label"]) == 9


def test_intention_task_requires_matching_class_count():
    with pytest.raises(ValueError, match="intention_classes=11"):
        Config(intention_task="overload11").validate()
    with pytest.raises(ValueError, match="intention_classes=9"):
        Config(intention_task="semantic9", intention_classes=11).validate()
    assert Config(intention_task="overload11", intention_classes=11).validate().intention_classes == 11


def test_overload11_requires_overload_labels_in_data(tmp_path):
    data = tmp_path / "missing.npz"
    _write_dataset(data, with_overload=False)
    tracks = load_tracks(data, .1, 10)
    ids = [t.track_id for t in tracks]
    with pytest.raises(ValueError, match="overload_labels"):
        Windows(tracks, ids, engagement(intention_task="overload11", intention_classes=11))
    assert len(Windows(tracks, ids, engagement(intention_task="semantic9"))) > 0


def test_window_intention_label_edges():
    unlabeled = Track("a", np.arange(30) * .1, np.zeros((30, 6), dtype=np.float32),
                      labels=np.full(30, -1), overload_labels=np.full(30, -1))
    assert window_intention_label(unlabeled, 0, 6, 9, "semantic9") == -1
    assert window_intention_label(unlabeled, 0, 6, 11, "overload11") == -1
    votes = Track("b", np.arange(30) * .1, np.zeros((30, 6), dtype=np.float32),
                  labels=np.array([2, 2, 2, 2, 0, 0] + [0] * 24),
                  overload_labels=np.array([5] * 6 + [0] * 24))
    assert window_intention_label(votes, 0, 6, 9, "semantic9") == 2      # majority vote
    assert window_intention_label(votes, 0, 6, 11, "overload11") == 5    # overload source
    with pytest.raises(ValueError, match="overload11"):
        window_intention_label(Track("c", np.arange(30) * .1, np.zeros((30, 6), dtype=np.float32),
                                     labels=np.zeros(30, dtype=np.int64)), 0, 6, 11, "overload11")


def test_overload11_trains_eleven_way_classifier(tmp_path):
    cfg = engagement(intention_task="overload11", intention_classes=11)
    data = tmp_path / "e.npz"
    simulate_engagements(data, scenes=6, steps=60, dt=.1, seed=7)
    tracks = load_tracks(data, cfg.dt, 10)
    windows = Windows(tracks, split_tracks(tracks, .7, .15, 42)["train"], cfg)
    sample = windows[0]
    batch = {key: sample[key][None] for key in ("observed", "context", "context_mask", "future", "label")}
    assert 0 <= int(sample["label"]) <= 10
    model = Forecaster(cfg, fit_normalizer(windows))
    output, loss = TrajectoryLearner(cfg, model, torch.device("cpu")).update(batch)
    assert output["logits"].shape[-1] == 11
    assert output["trajectory"].shape == (1, cfg.prediction_steps, 3)
    assert torch.isfinite(loss["loss"])


def test_window_mean_quantization_with_series(tmp_path):
    # F1 (paper IV-A): with the raw overload series present, the eleven-level
    # window label is ONE quantization of the window mean a_bar_n; a window
    # mixing +3 g (30 steps) and +9 g (10 steps) has mean 4.5 g, which the
    # (I_m - 0.5, I_m + 0.5] rule with tie->larger assigns to the +4 g row.
    from trajectory_prediction.engagement_scenarios import G0
    series = np.concatenate([np.full(30, 3.0 * G0), np.full(10, 9.0 * G0)])
    track = Track("t", np.arange(40) * .5, np.zeros((40, 6), dtype=np.float32),
                  np.zeros(40, dtype=np.int64), overload_labels=np.full(40, 6),
                  overload_series=series)
    assert window_intention_label(track, 0, 40, 11, "overload11") == 7
    # censored ends must reach the +/-10 g rows for saturated window means
    sat = Track("s", np.arange(40) * .5, np.zeros((40, 6), dtype=np.float32),
                np.zeros(40, dtype=np.int64), overload_labels=np.full(40, 10),
                overload_series=np.full(40, 9.4 * G0), overload_censored=True)
    assert window_intention_label(sat, 0, 40, 11, "overload11") == 10
    # strict convention keeps 9.4 g in the 9->10 g gap -> nearest level 10 g
    strict = Track("x", np.arange(40) * .5, np.zeros((40, 6), dtype=np.float32),
                   np.zeros(40, dtype=np.int64), overload_labels=np.full(40, 10),
                   overload_series=np.full(40, 9.4 * G0), overload_censored=False)
    assert window_intention_label(strict, 0, 40, 11, "overload11") == 10


def test_overload11_engagement_dataset_end_to_end_label_fidelity(tmp_path):
    # F1 end-to-end: simulate -> load -> window label must equal a direct
    # window-mean quantization computed from the npz arrays, for every window.
    from trajectory_prediction.engagement_scenarios import G0, simulate_engagements
    path = tmp_path / "e2e.npz"
    simulate_engagements(path, scenes=5, steps=120, dt=.5, sample_stride=5, seed=21)
    tracks = load_tracks(path, .5, 20)
    levels = np.array([-10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10.])
    for track in tracks:
        for start in range(0, 60, 5):
            label = window_intention_label(track, start, start + 20, 11, "overload11")
            mean_g = float(np.mean(track.overload_series[start:start + 20])) / G0
            offset = mean_g - levels
            inside = (offset > -.5) & (offset <= .5)
            expected = int(np.flatnonzero(inside)[-1]) if inside.any() else int(np.argmin(np.abs(offset)))
            assert label == expected


def test_legacy_overload_dataset_without_series_still_loads(tmp_path):
    # F1 backward compatibility: an old npz (no overload_series / no
    # overload_censored) loads fine and window labels fall back to majority
    # voting over the stored per-step bins.
    from trajectory_prediction.engagement_scenarios import simulate_engagements
    path = tmp_path / "legacy.npz"
    simulate_engagements(path, scenes=4, steps=120, dt=.5, sample_stride=5, seed=33)
    legacy = tmp_path / "legacy2.npz"
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files if k not in ("overload_series", "overload_censored")}
    np.savez_compressed(legacy, **arrays)
    tracks = load_tracks(legacy, .5, 20)
    assert all(t.overload_series is None and t.overload_censored is False for t in tracks)
    votes = Track("v", np.arange(30) * .5, np.zeros((30, 6), dtype=np.float32),
                  np.zeros(30, dtype=np.int64), overload_labels=np.array([5] * 6 + [0] * 24))
    assert window_intention_label(votes, 0, 30, 11, "overload11") == 0
