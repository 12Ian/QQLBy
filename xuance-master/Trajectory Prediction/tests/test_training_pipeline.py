import csv
import numpy as np
import pytest
import torch

from trajectory_prediction.config import Config
from trajectory_prediction.data import generate_tracks, load_tracks
from trajectory_prediction.engine import evaluate, load_checkpoint, train
from trajectory_prediction.inference import Predictor


def cfg(epochs, **updates):
    values = dict(observation_steps=6, prediction_steps=4, stride=5, lstm_hidden=8,
                  projection_dim=8, graph_dim=12, heads=3, graph_layers=1,
                  batch_size=8, dropout=0.1, epochs=epochs, device="cpu", cpu_threads=1)
    values.update(updates)
    return Config(**values).validate()


def test_training_resume_matches_uninterrupted_and_predicts(tmp_path):
    path = tmp_path / "data.npz"
    generate_tracks(path, count=12, steps=24, seed=7)
    continuous, interrupted = tmp_path / "continuous", tmp_path / "interrupted"
    train(path, continuous, cfg(3))
    train(path, interrupted, cfg(2))
    train(path, interrupted, cfg(3), resume=interrupted / "last.pt")
    reference, resumed = (load_checkpoint(p / "last.pt") for p in (continuous, interrupted))
    for key in reference["model"]:
        torch.testing.assert_close(reference["model"][key], resumed["model"][key], rtol=0, atol=0)
    assert reference["validation"] == resumed["validation"]
    with (interrupted / "history.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["epoch"]) for r in rows] == [1, 1, 2, 2, 3, 3]
    result = evaluate(path, interrupted / "best.pt", device_name="cpu")
    assert np.isfinite(result["ade_m"]) and result["split"] == "test"
    predictor = Predictor(interrupted / "best.pt", "cpu")
    history = load_tracks(path, 0.5, 10)[0].states
    forecast = predictor.predict(history)
    assert forecast["trajectory"].shape == (4, 3)
    assert np.isfinite(forecast["trajectory"]).all()
    assert np.isclose(forecast["intention_probabilities"].sum(), 1)
    with pytest.raises(ValueError, match="dataset differs"):
        other = tmp_path / "other.npz"
        generate_tracks(other, count=12, steps=24, seed=9)
        train(other, interrupted, cfg(4), resume=interrupted / "last.pt")


def test_two_stage_intention_transfer(tmp_path):
    path = tmp_path / "data.npz"
    generate_tracks(path, count=12, steps=24)
    intention, forecast = tmp_path / "intention", tmp_path / "forecast"
    train(path, intention, cfg(1, stage="intention", dropout=0.0))
    before = load_checkpoint(intention / "best.pt")
    train(path, forecast, cfg(1, stage="forecast", freeze_intention=True, dropout=0.0),
          intention_checkpoint=intention / "best.pt")
    after = load_checkpoint(forecast / "last.pt")
    for key in before["model"]:
        if key.startswith("intention."):
            torch.testing.assert_close(before["model"][key], after["model"][key], rtol=0, atol=0)
    assert torch.equal(before["normalizer"]["mean"], after["normalizer"]["mean"])


def test_unlabeled_forecast_and_actionable_errors(tmp_path):
    path = tmp_path / "data.npz"
    generate_tracks(path, count=12, steps=24)
    with np.load(path) as archive:
        values = {key: archive[key] for key in archive.files if key != "labels"}
    unlabeled = tmp_path / "unlabeled.npz"
    np.savez(unlabeled, **values)
    with pytest.raises(ValueError, match="needs observed-window labels"):
        train(unlabeled, tmp_path / "invalid", cfg(1))
    train(unlabeled, tmp_path / "forecast", cfg(1, stage="forecast"))
    result = evaluate(unlabeled, tmp_path / "forecast" / "last.pt", device_name="cpu")
    assert result["intention_accuracy"] is None
    assert result["labeled_samples"] == 0
