"""Training and independent evaluation for the graph/candidate predictor."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import prediction_windows, scale_intention, split_prediction_tracks, track_files
from .intention import IntentionBiLSTM
from .metrics import constant_acceleration, constant_velocity, displacement_errors
from .prediction import TrajectoryPredictor


def _load_intention(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = IntentionBiLSTM(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["scaler"]


def _dataset(paths: list[Path], intention_model: IntentionBiLSTM,
             scaler: dict, device: torch.device, batch_size: int) -> TensorDataset:
    windows = list(prediction_windows(paths, intention_seconds=10.0))
    if not windows:
        raise ValueError("No 10 s intention + 8 s prediction windows in these tracks")
    target = np.stack([item["target_history"] for item in windows])
    defender = np.stack([item["defender_history"] for item in windows])
    truth = np.stack([item["future_xyz"] for item in windows])
    acceleration = np.stack([item["target_acceleration"] for item in windows])
    intention_target = np.stack([item["intention_target_history"] for item in windows])
    intention_defender = np.stack([item["intention_defender_state"] for item in windows])
    origin = target[:, -1:, :3]
    intention_target[:, :, :3] -= origin
    intention_defender[:, :3] -= origin[:, 0]
    intention_target, intention_defender = scale_intention(
        intention_target, intention_defender, scaler)
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            end = start + batch_size
            logits = intention_model(
                torch.from_numpy(intention_target[start:end]).to(device),
                torch.from_numpy(intention_defender[start:end]).to(device))
            probabilities.append(logits.softmax(dim=-1).cpu())
    return TensorDataset(torch.from_numpy(target), torch.from_numpy(defender),
                         torch.cat(probabilities), torch.from_numpy(truth),
                         torch.from_numpy(acceleration))


def _loader(dataset: TensorDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _errors(predicted: torch.Tensor, truth: torch.Tensor):
    distances = (predicted - truth).norm(dim=-1)
    return distances.mean(dim=-1), distances[:, -1]


@torch.no_grad()
def _score(model: TrajectoryPredictor, loader: DataLoader,
           device: torch.device) -> dict[str, float]:
    model.eval()
    total_ade = total_fde = total = 0
    for target, defender, probabilities, truth, _ in loader:
        predicted = model(target.to(device), defender.to(device),
                          probabilities.to(device))["future_xyz"]
        ade, fde = _errors(predicted, truth.to(device))
        total_ade += float(ade.sum())
        total_fde += float(fde.sum())
        total += len(target)
    return {"ADE_m": total_ade / total, "FDE_m": total_fde / total,
            "windows": total}


def _write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def train_prediction(data: Path, intention_checkpoint: Path, output: Path,
                     epochs: int, batch_size: int, learning_rate: float,
                     seed: int, device_name: str, config: dict) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_name)
    intention_model, scaler = _load_intention(intention_checkpoint, device)
    split = split_prediction_tracks(track_files(data), seed)
    loaders = {name: _loader(_dataset(paths, intention_model, scaler, device,
                                      batch_size), batch_size, name == "train")
               for name, paths in split.items()}
    model_config = config["model"]
    if model_config["future_steps"] != 60 or not np.isclose(model_config["dt"], 0.1):
        raise ValueError("Current paper windows require 60 future steps at 0.1 s")
    if not 0.0 <= model_config["candidate_acceleration_g"] <= 9.0:
        raise ValueError("Candidate acceleration must respect the 9g target limit")
    ade_weight = float(config["loss"]["ade_weight"])
    fde_weight = float(config["loss"]["fde_weight"])
    if ade_weight < 0 or fde_weight < 0 or not np.isclose(ade_weight + fde_weight, 1.0):
        raise ValueError("ADE and FDE weights must be nonnegative and sum to one")
    model = TrajectoryPredictor(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best = float("inf")
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for target, defender, probabilities, truth, _ in loaders["train"]:
            prediction = model(target.to(device), defender.to(device),
                               probabilities.to(device))["future_xyz"]
            ade, fde = _errors(prediction, truth.to(device))
            loss = (ade_weight * ade.mean() + fde_weight * fde.mean()) / 1000.0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation = _score(model, loaders["eval"], device)
        selection = ade_weight * validation["ADE_m"] + fde_weight * validation["FDE_m"]
        history.append({"epoch": epoch, **validation})
        print(f"epoch={epoch} val_ADE={validation['ADE_m']:.2f}m "
              f"val_FDE={validation['FDE_m']:.2f}m", flush=True)
        if selection < best:
            best = selection
            best_state = copy.deepcopy(model.state_dict())
    output.mkdir(parents=True, exist_ok=True)
    intention_hash = hashlib.sha256(intention_checkpoint.read_bytes()).hexdigest()
    torch.save({"model": best_state, "model_config": model_config,
                "experiment_config": config,
                "intention_checkpoint_sha256": intention_hash,
                "train_data": str(data.resolve()), "seed": seed},
               output / "prediction_best.pt")
    result = {"split_tracks": {name: [p.stem for p in paths]
                                for name, paths in split.items()},
              "window_counts": {name: len(loader.dataset) for name, loader in loaders.items()},
              "history": history, "best_weighted_error_m": best,
              "experiment_config": config,
              "intention_checkpoint_sha256": intention_hash}
    _write_json(output / "prediction_validation.json", result)
    return result


def evaluate_prediction(data: Path, intention_checkpoint: Path,
                        prediction_checkpoint: Path, output: Path,
                        batch_size: int, device_name: str) -> dict:
    device = torch.device(device_name)
    saved = torch.load(prediction_checkpoint, map_location="cpu", weights_only=False)
    intention_hash = hashlib.sha256(intention_checkpoint.read_bytes()).hexdigest()
    if intention_hash != saved["intention_checkpoint_sha256"]:
        raise ValueError("Intention checkpoint differs from the one used for training")
    if data.resolve() == Path(saved["train_data"]):
        raise ValueError("Evaluation data must be a separate generated directory")
    intention_model, scaler = _load_intention(intention_checkpoint, device)
    dataset = _dataset(track_files(data), intention_model, scaler, device, batch_size)
    model = TrajectoryPredictor(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model"])
    metrics = {"graph_candidate_GRU": _score(model,
               _loader(dataset, batch_size, False), device)}
    errors = {"CV": [], "CA": []}
    for target, _, _, truth, acceleration in _loader(dataset, batch_size, False):
        for history, future, accel in zip(target.numpy(), truth.numpy(),
                                          acceleration.numpy()):
            errors["CV"].append(displacement_errors(
                constant_velocity(history, 0.1, len(future)), future))
            errors["CA"].append(displacement_errors(
                constant_acceleration(history, accel, 0.1, len(future)), future))
    for name, values in errors.items():
        metrics[name] = {"ADE_m": float(np.mean(values, axis=0)[0]),
                         "FDE_m": float(np.mean(values, axis=0)[1]),
                         "windows": len(values)}
    _write_json(output / "prediction_test.json", metrics)
    return metrics
