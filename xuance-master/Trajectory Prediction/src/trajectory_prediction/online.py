"""Repeated forecasting from newly observed states, with offline scoring."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import scale_intention
from .metrics import displacement_errors
from .prediction import TrajectoryPredictor
from .prediction_train import _load_intention


@torch.no_grad()
def forecast_from_prefix(target_prefix: np.ndarray, defender_prefix: np.ndarray,
                         intention_model, intention_scaler: dict,
                         prediction_model: TrajectoryPredictor,
                         device: torch.device) -> dict[str, np.ndarray]:
    """Forecast from state prefixes only; caller owns any future ground truth."""
    if target_prefix.shape != defender_prefix.shape or target_prefix.shape[0] < 96:
        raise ValueError("Need aligned target/defender prefixes of at least 96 steps")
    origin = target_prefix[-1, :3]
    intention_target = target_prefix[-96::5, :6][None].copy()
    intention_defender = defender_prefix[-1:, :6].copy()
    intention_target[:, :, :3] -= origin
    intention_defender[:, :3] -= origin
    intention_target, intention_defender = scale_intention(
        intention_target, intention_defender, intention_scaler)
    probabilities = intention_model(
        torch.from_numpy(intention_target).to(device),
        torch.from_numpy(intention_defender).to(device)).softmax(dim=-1)
    result = prediction_model(
        torch.from_numpy(target_prefix[-80:, :6][None]).to(device),
        torch.from_numpy(defender_prefix[-80:, :6][None]).to(device),
        probabilities)
    return {"future_xyz": result["future_xyz"][0].cpu().numpy(),
            "intention_prob": probabilities[0].cpu().numpy(),
            "candidate_weights": result["candidate_weights"][0].cpu().numpy()}


def rolling_evaluation(track_path: Path, intention_checkpoint: Path,
                       prediction_checkpoint: Path, output: Path,
                       update_seconds: float, device_name: str) -> dict:
    saved = torch.load(prediction_checkpoint, map_location="cpu", weights_only=False)
    actual_hash = hashlib.sha256(intention_checkpoint.read_bytes()).hexdigest()
    if actual_hash != saved["intention_checkpoint_sha256"]:
        raise ValueError("Intention checkpoint differs from predictor training")
    if track_path.parent.resolve() == Path(saved["train_data"]):
        raise ValueError("Rolling evaluation track must be outside the training data directory")
    device = torch.device(device_name)
    intention_model, scaler = _load_intention(intention_checkpoint, device)
    predictor = TrajectoryPredictor(**saved["model_config"]).to(device)
    predictor.load_state_dict(saved["model"])
    predictor.eval()
    with np.load(track_path, allow_pickle=False) as track:
        target = track["target"]
        defender = track["defender"]
        times = track["time"]
    dt = float(saved["model_config"]["dt"])
    stride = int(round(update_seconds / dt))
    if stride < 1 or not np.isclose(stride * dt, update_seconds):
        raise ValueError("Update interval must be a positive multiple of model dt")
    end_indices = range(95, len(target) - predictor.future_steps, stride)
    forecasts, truths, probabilities, weights, observation_times, errors = [], [], [], [], [], []
    for end in end_indices:
        result = forecast_from_prefix(target[:end + 1], defender[:end + 1],
                                      intention_model, scaler, predictor, device)
        future = target[end + 1:end + 1 + predictor.future_steps, :3]
        forecasts.append(result["future_xyz"])
        truths.append(future)
        probabilities.append(result["intention_prob"])
        weights.append(result["candidate_weights"])
        observation_times.append(float(times[end]))
        errors.append(displacement_errors(result["future_xyz"], future))
    if not forecasts:
        raise ValueError("Track is too short for a 10 s prefix and 6 s forecast")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "rolling_forecasts.npz",
                        forecast_xyz=np.stack(forecasts), truth_xyz=np.stack(truths),
                        intention_prob=np.stack(probabilities),
                        candidate_weights=np.stack(weights),
                        observation_time=np.asarray(observation_times))
    metrics = {"track": track_path.name, "update_seconds": update_seconds,
               "forecasts": len(forecasts),
               "ADE_m": float(np.mean(errors, axis=0)[0]),
               "FDE_m": float(np.mean(errors, axis=0)[1]),
               "per_update": [{"time_s": time, "ADE_m": float(error[0]),
                               "FDE_m": float(error[1])}
                              for time, error in zip(observation_times, errors)]}
    (output / "rolling_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
