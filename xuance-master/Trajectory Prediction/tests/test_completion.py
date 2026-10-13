import csv
import json

import numpy as np
import pytest
import torch

from trajectory_prediction.data import generate_tracks, load_tracks, relative_features
from trajectory_prediction.engine import train, load_checkpoint
from trajectory_prediction.history_graph import history_edges
from trajectory_prediction.inference import Predictor
from trajectory_prediction.rolling import RollingPredictor, rolling_evaluate
from trajectory_prediction.workflows import compare
from trajectory_prediction.reporting import report
from trajectory_prediction.study import study
from test_native_workflows import settings


def advanced(**updates):
    values = dict(historical_edge_features=True, historical_reachability=True, crop_warmup_epochs=2,
                  early_stopping_mode="both", intention_dropout=.2, dropout=.1)
    values.update(updates)
    return settings(**values)


def test_history_reachability_and_edge_feature_gradients():
    cfg = advanced()
    observed = torch.zeros(2, 6, 6)
    observed[..., 3] = 2
    observed[..., 0] = torch.arange(6)
    observed[:, 3, 0] = 100
    edges, reachable = history_edges(relative_features(observed), cfg)
    assert edges.shape == (2, 6, 6, 7)
    assert not reachable[:, 3, :3].any()
    assert reachable[:, 3, 3].all()
    from trajectory_prediction.model import Forecaster
    from trajectory_prediction.learner import TrajectoryLearner
    policy = Forecaster(cfg)
    learner = TrajectoryLearner(cfg, policy, torch.device("cpu"), steps_per_epoch=2)
    output, loss = learner.update(dict(observed=observed, future=torch.zeros(2, 4, 3), label=torch.tensor([0, 1])))
    assert float(policy.crop_progress) == .25
    assert torch.isfinite(loss["loss"])
    assert policy.graph[0].edge_context.weight.grad.abs().sum() > 0
    for weights in output["attention"]:
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 6, 6))
        assert weights[:, :, 3, :3].eq(0).all()


def test_progressive_crop_and_both_metric_resume_is_exact(tmp_path):
    data = tmp_path / "data.npz"
    generate_tracks(data, 12, 24)
    train(data, tmp_path / "full", advanced(epochs=3))
    train(data, tmp_path / "resume", advanced(epochs=1))
    train(data, tmp_path / "resume", advanced(epochs=3), resume=tmp_path / "resume/last.pt")
    left, right = (load_checkpoint(tmp_path / path / "last.pt") for path in ("full", "resume"))
    for key in left["model"]:
        torch.testing.assert_close(left["model"][key], right["model"][key], atol=0, rtol=0)
    for key in left["criterion"]:
        torch.testing.assert_close(left["criterion"][key], right["criterion"][key], atol=0, rtol=0)
    assert left["early_stop_anchor"] == right["early_stop_anchor"]
    assert float(right["model"]["crop_progress"]) == 1


def test_rolling_uses_only_received_measurements_and_matches_batch(tmp_path):
    data = tmp_path / "data.npz"
    generate_tracks(data, 12, 24)
    train(data, tmp_path / "run", advanced(epochs=1))
    checkpoint = tmp_path / "run/best.pt"
    predictor = Predictor(checkpoint, "cpu")
    stream = RollingPredictor(predictor, refresh_steps=2)
    track = load_tracks(data, .5, 10)[0]
    for i in range(5):
        assert stream.push(track.time[i], track.states[i]) is None
    first = stream.push(track.time[5], track.states[5])
    np.testing.assert_allclose(first["trajectory"], predictor.predict(track.states[:6])["trajectory"], rtol=0, atol=0)
    saved_history = np.stack(stream.history)
    track.states[6:] += 1000  # Future edits cannot change already produced forecasts.
    np.testing.assert_array_equal(np.stack(stream.history), saved_history)
    with pytest.raises(ValueError, match="timestamps"):
        stream.push(track.time[5], track.states[5])
    assert stream.received == 6
    stream.reset()
    assert len(stream.history) == 0 and stream.received == 0
    result = rolling_evaluate(data, checkpoint, tmp_path / "rolling", device="cpu", refresh_steps=2)
    assert result["protocol"]["future_inputs_used"] is False
    assert result["samples"] > 0 and result["latency_seconds"]["p95"] > 0
    with (tmp_path / "rolling/rolling_predictions.csv").open(newline="", encoding="utf-8") as file:
        assert sum(1 for _ in csv.DictReader(file)) == result["samples"] * 4


def test_shared_classifier_full_factorial_ablation_and_figures(tmp_path):
    data = tmp_path / "data.npz"
    generate_tracks(data, 12, 24)
    cfg = advanced(epochs=1)
    models = ["graph", "graph_no_intention", "graph_no_attention_supervision", "graph_no_both"]
    result = compare(data, tmp_path / "compare", cfg, models, [3], intention_epochs=1)
    assert result["graph_training"] == "two_stage"
    pairs = {(r["intention_enabled"], r["civilian_teachers_enabled"]) for r in result["runs"]}
    assert pairs == {(True, True), (False, True), (True, False), (False, False)}
    shared = load_checkpoint(tmp_path / "compare/_intention/seed_3/best.pt")
    for model in models:
        saved = load_checkpoint(tmp_path / "compare" / model / "seed_3/best.pt")
        assert saved["config"]["freeze_intention"]
        for key in shared["model"]:
            if key.startswith("intention."):
                torch.testing.assert_close(saved["model"][key], shared["model"][key], atol=0, rtol=0)
    figures = report(tmp_path / "compare", tmp_path / "figures", data)
    assert "civilian_3d_cases" in figures["figures"]
    evidence = json.loads((tmp_path / "figures/report_values.json").read_text(encoding="utf-8"))
    for summary in result["summary"]:
        row = next(r for r in evidence["evidence"]["ade_fde"] if r["model"] == summary["model"])
        assert row["ade_m_mean"] == summary["ade_m_mean"]
    assert (tmp_path / "figures/horizon_error.svg").stat().st_size > 1000


def test_independent_library_study_and_resume(tmp_path):
    result = study(tmp_path / "study", advanced(epochs=1), ["graph", "gru"], [101, 102], [3, 4],
                   tracks=12, steps=24, intention_epochs=1)
    assert len(result["comparisons"]) == 2
    assert result["summary"][0]["independent_libraries"] == 2
    first = json.loads((tmp_path / "study/data/seed_101.metadata.json").read_text())
    second = json.loads((tmp_path / "study/data/seed_102.metadata.json").read_text())
    assert first["sha256"] != second["sha256"]
    resumed = study(tmp_path / "study", advanced(epochs=1), ["graph", "gru"], [101, 102], [3, 4],
                    tracks=12, steps=24, intention_epochs=1, resume=True)
    assert resumed["summary"] == result["summary"]
