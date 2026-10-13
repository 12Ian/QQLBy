import json
import csv
from argparse import Namespace
from dataclasses import replace

import numpy as np
import pytest
import torch

from trajectory_prediction.civilian_scenarios import simulate_scenes
from trajectory_prediction.data import Windows, load_tracks, split_tracks, validate_split, fit_normalizer, rotate_xy
from trajectory_prediction.engine import train, evaluate, load_checkpoint
from trajectory_prediction.inference import Predictor
from trajectory_prediction.interaction_graph import interaction_edges
from trajectory_prediction.learner import TrajectoryLearner
from trajectory_prediction.model import Forecaster
from trajectory_prediction.rolling import RollingPredictor, rolling_evaluate
from trajectory_prediction.workflows import pipeline, compare
from trajectory_prediction.reporting import report
from trajectory_prediction.study import study
from trajectory_prediction.xuance_bridge import framework
from test_native_workflows import settings


def interaction(**updates):
    values = dict(context_entities=1, historical_edge_features=True, historical_reachability=True,
                  crop_warmup_epochs=2, early_stopping_mode="both", intention_dropout=.2)
    values.update(updates)
    return settings(**values)


def dataset(tmp_path, cfg=None, scenes=12):
    cfg = cfg or interaction()
    path = tmp_path / "scenes.npz"
    simulate_scenes(path, scenes, 48, cfg.dt, seed=11, measurement_dropout=.15)
    tracks = load_tracks(path, cfg.dt, cfg.observation_steps + cfg.prediction_steps)
    splits = split_tracks(tracks, cfg.train_fraction, cfg.validation_fraction, cfg.split_seed)
    return path, tracks, splits, Windows(tracks, splits["train"], cfg)


def test_civilian_simulation_reproducibility_and_physical_limits(tmp_path):
    first = simulate_scenes(tmp_path / "a.npz", 12, 80)
    second = simulate_scenes(tmp_path / "b.npz", 12, 80)
    assert first["sha256"] == second["sha256"]
    tracks = load_tracks(tmp_path / "a.npz", .5, 10)
    assert len(tracks) == 24 and len({track.scene_id for track in tracks}) == 12
    for track in tracks:
        assert np.linalg.norm(track.states[:, 3:], axis=1).max() <= 1.5 + 1e-6
        assert np.linalg.norm(np.diff(track.states[:, 3:], axis=0) / .5, axis=1).max() <= 1 + 1e-6
        other = next(t for t in tracks if t.scene_id == track.scene_id and t.track_id != track.track_id)
        np.testing.assert_array_equal(track.context[:, 0], other.states)
        expected = .5 * (track.states[:-1, 3:] + track.states[1:, 3:]) / 2
        np.testing.assert_allclose(np.diff(track.states[:, :3], axis=0), expected, atol=3e-6)
    assert any(sum(row["waiting_steps"]) for row in first["diagnostics"])


def test_scene_split_and_observation_only_normalizer(tmp_path):
    _, tracks, splits, windows = dataset(tmp_path)
    validate_split(tracks, splits)
    groups = [{t.scene_id for t in tracks if t.track_id in ids} for ids in splits.values()]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    broken = {key: list(value) for key, value in splits.items()}
    moved = broken["train"].pop()
    broken["test"].append(moved)
    with pytest.raises(ValueError, match="Scene members"):
        validate_split(tracks, broken)
    normalizer = fit_normalizer(windows)
    for track in tracks:
        if track.track_id in splits["test"]:
            track.states += 10000
            track.context += 10000
    after = fit_normalizer(windows)
    for key in normalizer:
        torch.testing.assert_close(normalizer[key], after[key], rtol=0, atol=0)
    sample = windows[0]
    original = sample["context"].clone()
    track_index, start = windows.index[0]
    windows.tracks[track_index].context[start + windows.cfg.observation_steps:] += 300
    torch.testing.assert_close(windows[0]["context"], original, rtol=0, atol=0)


def test_temporal_spatial_edges_and_missing_measurement_mask():
    cfg = interaction()
    features = torch.zeros(1, 12, 6)
    features[:, :6, 0] = torch.arange(6)
    features[:, 6:, 0] = torch.arange(6)
    features[:, 6:, 1] = 2
    features[..., 3] = 2
    valid = torch.ones(1, 12, dtype=torch.bool)
    valid[:, 8] = False
    edges, allowed = interaction_edges(features, valid, cfg)
    assert edges.shape == (1, 12, 12, 7)
    assert allowed[0, 0, 6] and not allowed[0, 0, 7]
    assert not allowed[0, :6, 8].any() and allowed[0, 8, 8]
    features[:, 6:, 1] = 100
    _, distant = interaction_edges(features, valid, cfg)
    assert not distant[:, :6, 6:].any()


def test_peer_branch_graph_gradients_masking_and_rotation(tmp_path):
    cfg = interaction(dropout=0, intention_dropout=0)
    _, _, _, windows = dataset(tmp_path, cfg)
    sample = windows[0]
    batch = {key: sample[key][None] for key in ("observed", "context", "context_mask", "future", "label")}
    # Make a stationary visible peer close to the focal for spatial links.
    batch["context"] = batch["observed"][:, None].clone()
    batch["context"][..., 1] += 1
    batch["context_mask"].fill_(True)
    model = Forecaster(cfg).eval()
    learner = TrajectoryLearner(cfg, model, torch.device("cpu"))
    output, _ = learner.update(batch)
    assert output["attention"][0].shape == (1, 6, 12, 12)
    assert model.entity_embedding.grad.abs().sum() > 0
    assert model.graph[0].edge_context.weight.grad.abs().sum() > 0
    model.eval()
    masked = batch["context_mask"].clone(); masked[:, :, 2] = False
    reference = model(batch["observed"], batch["context"], masked)
    corrupt = batch["context"].clone(); corrupt[:, :, 2] = float("nan")
    changed = model(batch["observed"], corrupt, masked)
    torch.testing.assert_close(reference["trajectory"], changed["trajectory"], rtol=0, atol=0)
    torch.testing.assert_close(reference["logits"], changed["logits"], rtol=0, atol=0)
    angle = torch.tensor([.6])
    observed = batch["observed"].clone()
    observed[..., :3] = rotate_xy(observed[..., :3], angle, inverse=True)
    observed[..., 3:] = rotate_xy(observed[..., 3:], angle, inverse=True)
    peers = batch["context"].clone()
    peers[:, 0, :, :3] = rotate_xy(peers[:, 0, :, :3], angle, inverse=True)
    peers[:, 0, :, 3:] = rotate_xy(peers[:, 0, :, 3:], angle, inverse=True)
    rotated = model(observed, peers, masked)
    torch.testing.assert_close(rotated["trajectory"], rotate_xy(reference["trajectory"], angle, inverse=True), atol=2e-5, rtol=2e-5)


def test_no_context_ablation_is_invariant_to_peer_values(tmp_path):
    cfg = interaction(use_interaction_context=False, dropout=0, intention_dropout=0)
    _, _, _, windows = dataset(tmp_path, cfg)
    sample = windows[0]
    model = Forecaster(cfg).eval()
    first = model(sample["observed"][None], sample["context"][None], sample["context_mask"][None])
    second = model(sample["observed"][None], sample["context"][None] + 1000, sample["context_mask"][None])
    assert first["attention"][0].shape[-1] == cfg.observation_steps
    for key in ("trajectory", "logits"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_native_pair_pipeline_rolling_and_scene_cluster_intervals(tmp_path):
    cfg = interaction(epochs=2)
    path, tracks, splits, _ = dataset(tmp_path, cfg)
    result = pipeline(path, tmp_path / "pipeline", cfg, intention_epochs=2)
    assert result["forecast_test"]["bootstrap"]["unit"] == "independent scene"
    assert "scene_cluster_bootstrap_95ci" in result["forecast_test"]
    checkpoint = tmp_path / "pipeline/forecast/best.pt"
    predictor = Predictor(checkpoint, "cpu")
    stream = RollingPredictor(predictor)
    track = tracks[0]
    for i in range(6):
        forecast = stream.push(track.time[i], track.states[i], track.context[i], track.context_mask[i])
    batch = predictor.predict(track.states[:6], track.context[:6].transpose(1, 0, 2), track.context_mask[:6].T)
    np.testing.assert_array_equal(forecast["trajectory"], batch["trajectory"])
    with pytest.raises(ValueError, match="synchronized"):
        stream.push(track.time[6], track.states[6])
    assert stream.received == 6
    rolling = rolling_evaluate(path, checkpoint, tmp_path / "rolling", device="cpu", refresh_steps=3)
    assert rolling["protocol"]["context_entities"] == 1
    assert rolling["bootstrap"]["unit"] == "independent scene"


def test_interaction_four_quadrants_and_report(tmp_path):
    cfg = interaction()
    path, _, _, _ = dataset(tmp_path, cfg)
    models = ["graph", "graph_no_intention", "graph_no_context", "graph_no_intention_context"]
    result = compare(path, tmp_path / "compare", cfg, models, [3], intention_epochs=1)
    assert {(r["intention_enabled"], r["interaction_context_enabled"]) for r in result["runs"]} == {(True,True),(False,True),(True,False),(False,False)}
    assert all(row["civilian_teachers_enabled"] for row in result["runs"])
    figures = report(tmp_path / "compare", tmp_path / "figures", path)
    assert "civilian_interaction_cases" in figures["figures"]
    recorded = json.loads((tmp_path / "figures/report_values.json").read_text(encoding="utf-8"))
    assert recorded["evidence"]["cases"][0]["peer_observed"]


def test_pair_resume_and_independent_scene_library_study(tmp_path):
    cfg = interaction(epochs=2)
    path, _, _, _ = dataset(tmp_path, cfg, scenes=6)
    train(path, tmp_path / "continuous", cfg)
    train(path, tmp_path / "resumed", replace(cfg, epochs=1))
    train(path, tmp_path / "resumed", cfg, resume=tmp_path / "resumed/last.pt")
    left, right = [load_checkpoint(tmp_path / name / "last.pt") for name in ("continuous", "resumed")]
    for key in left["model"]:
        torch.testing.assert_close(left["model"][key], right["model"][key], rtol=0, atol=0)
    for key in left["criterion"]:
        torch.testing.assert_close(left["criterion"][key], right["criterion"][key], rtol=0, atol=0)
    result = study(tmp_path / "study", interaction(), ["gru"], [101,102], [3], tracks=6, steps=24,
                   intention_epochs=1, data_mode="civilian_interaction")
    assert result["scenes_per_library"] == 6 and result["tracks_per_library"] == 12


def test_native_paired_scene_config_and_agent_prediction(tmp_path):
    cfg = interaction()
    path, _, _, windows = dataset(tmp_path, cfg, scenes=6)
    framework()
    from xuance import get_runner
    native = get_runner(algo="trajectory_prediction", env="civilian_trajectory", env_id="paired_scene",
                        parser_args=Namespace(data_path=str(path), output_dir=str(tmp_path / "native"),
                                              trajectory_overrides=cfg.to_dict(), device="cpu"))
    result = native.run(mode="benchmark")
    assert result["test"]["protocol"]["context_entities"] == 1
    sample = windows[0]
    prediction = native.agent.predict(sample["observed"].numpy(), sample["context"].numpy(), sample["context_mask"].numpy())
    assert prediction["trajectory"].shape == (4, 3)


def test_csv_peer_schema_and_missing_measurements(tmp_path):
    original = tmp_path / "source.npz"
    simulate_scenes(original, 3, 12, measurement_dropout=.5)
    with np.load(original, allow_pickle=False) as saved:
        fields = ["track_id", "scene_id", "scenario", "time", "label", "x", "y", "z", "vx", "vy", "vz"]
        components = ("x", "y", "z", "vx", "vy", "vz")
        fields += [f"context_0_{key}" for key in components] + ["context_0_mask"]
        path = tmp_path / "data.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fields); writer.writeheader()
            for index in range(len(saved["time"])):
                available = bool(saved["context_mask"][index, 0])
                row = {key: saved[key][index] for key in ("track_id", "scene_id", "scenario", "time")}
                row.update(label=int(saved["labels"][index]), context_0_mask=int(available))
                row.update(zip(components, saved["states"][index]))
                row.update({f"context_0_{key}": value if available else "" for key, value in zip(components, saved["context_states"][index, 0])})
                writer.writerow(row)
    tracks = load_tracks(path, .5, 10)
    assert len(tracks) == 6
    assert any(not track.context_mask.all() for track in tracks)
    for track in tracks:
        assert np.isfinite(track.context).all()


def test_context_requires_declared_scene_groups_and_consistent_category(tmp_path):
    source = tmp_path / "source.npz"
    simulate_scenes(source, 3, 12)
    with np.load(source, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    missing = tmp_path / "missing_scene.npz"
    np.savez_compressed(missing, **{key: value for key, value in values.items() if key != "scene_id"})
    with pytest.raises(ValueError, match="explicit nonempty scene_id"):
        load_tracks(missing, .5, 10)
    mixed = tmp_path / "mixed_category.npz"
    values["scenario"] = values["scenario"].copy()
    values["scenario"][values["track_id"] == "scene_000000_agent_1"] = "merge"
    np.savez_compressed(mixed, **values)
    with pytest.raises(ValueError, match="one scenario category"):
        load_tracks(mixed, .5, 10)
    csv_path = tmp_path / "missing_scene.csv"
    components = ("x", "y", "z", "vx", "vy", "vz")
    fields = ["track_id", "time", *components, *[f"context_0_{key}" for key in components]]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fields); writer.writeheader()
        for i in range(len(values["time"])):
            row = dict(track_id=values["track_id"][i], time=values["time"][i])
            row.update(zip(components, values["states"][i]))
            row.update({f"context_0_{key}": val for key, val in zip(components, values["context_states"][i, 0])})
            writer.writerow(row)
    with pytest.raises(ValueError, match="explicit nonempty scene_id"):
        load_tracks(csv_path, .5, 10)
