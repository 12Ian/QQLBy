"""Reproducible data, intention, prediction, and baseline commands."""

import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import (fit_intention_scaler, intention_windows, prediction_windows,
                   scale_intention, split_prediction_tracks, split_tracks, track_files)
from .intention import IntentionBiLSTM
from .metrics import constant_acceleration, constant_velocity, displacement_errors
from .online import rolling_evaluation
from .prediction_train import evaluate_prediction, train_prediction
from .simulator import INTENT_NAMES, ScenarioConfig, simulate_track


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _generate(args) -> None:
    config_values = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = ScenarioConfig(**config_values["scenario"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for index in range(args.tracks):
        seed = args.seed + index
        track = simulate_track(config, seed)
        np.savez_compressed(output / f"track_{index:06d}.npz", **track)
    _json(output / "manifest.json", {
        "source": "yu2026_point_mass_reproduction_assumptions",
        "scenario": asdict(config), "tracks": args.tracks,
        "first_seed": args.seed, "intent_names": INTENT_NAMES,
        "notes": ["Defender guidance and primitive timing are explicit reproduction assumptions.",
                  "Target labels and accelerations describe the completed preceding interval."],
    })
    print(f"Generated {args.tracks} tracks in {output}")


def _loader(target, defender, labels, batch_size, shuffle):
    tensors = TensorDataset(torch.from_numpy(target.astype(np.float32)),
                            torch.from_numpy(defender.astype(np.float32)),
                            torch.from_numpy(labels))
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _score_intention(model, loader, device):
    model.eval()
    total_loss = total = correct = 0
    confusion = np.zeros((len(INTENT_NAMES), len(INTENT_NAMES)), dtype=np.int64)
    for target, defender, labels in loader:
        target, defender, labels = (target.to(device), defender.to(device),
                                    labels.to(device))
        logits = model(target, defender)
        total_loss += float(nn.functional.cross_entropy(
            logits, labels, reduction="sum"))
        predicted = logits.argmax(dim=-1)
        correct += int((predicted == labels).sum())
        total += len(labels)
        np.add.at(confusion, (labels.cpu().numpy(), predicted.cpu().numpy()), 1)
    return {"loss": total_loss / total, "accuracy": correct / total,
            "count": total, "confusion": confusion.tolist()}


def _train_intention(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    paths = split_tracks(track_files(Path(args.data)), args.seed)
    windows = {name: intention_windows(group) for name, group in paths.items()}
    scaler = fit_intention_scaler(*windows["train"][:2])
    loaders = {}
    for name, (target, defender, labels) in windows.items():
        target, defender = scale_intention(target, defender, scaler)
        loaders[name] = _loader(target, defender, labels, args.batch_size,
                                shuffle=(name == "train"))
    device = torch.device(args.device)
    model = IntentionBiLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.5)
    best_accuracy = -1.0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for target, defender, labels in loaders["train"]:
            target, defender, labels = (target.to(device), defender.to(device),
                                        labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(target, defender), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        validation = _score_intention(model, loaders["val"], device)
        history.append({"epoch": epoch, "val_loss": validation["loss"],
                        "val_accuracy": validation["accuracy"]})
        print(f"epoch={epoch} val_accuracy={validation['accuracy']:.4f} "
              f"val_loss={validation['loss']:.4f}")
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "model_config": {
        "hidden_size": 128, "projection_size": 64, "dropout": 0.2,
        "classes": len(INTENT_NAMES)}, "intent_names": INTENT_NAMES,
        "scaler": scaler}, output / "intention_best.pt")
    result = {"split_tracks": {name: [p.stem for p in group]
                                for name, group in paths.items()},
              "window_counts": {name: len(windows[name][2]) for name in windows},
              "history": history,
              "test": _score_intention(model, loaders["test"], device)}
    _json(output / "intention_metrics.json", result)
    print(f"test_accuracy={result['test']['accuracy']:.4f}; saved to {output}")


def _eval_baselines(args) -> None:
    paths = split_prediction_tracks(track_files(Path(args.data)), args.seed)
    errors = {"CV": [], "CA": []}
    for window in prediction_windows(paths["eval"]):
        truth = window["future_xyz"]
        history = window["target_history"]
        steps = len(truth)
        errors["CV"].append(displacement_errors(
            constant_velocity(history, args.dt, steps), truth))
        errors["CA"].append(displacement_errors(
            constant_acceleration(history, window["target_acceleration"],
                                  args.dt, steps), truth))
    if not errors["CV"]:
        raise ValueError("No prediction windows; increase track duration")
    result = {name: {"ADE_m": float(np.mean(values, axis=0)[0]),
                     "FDE_m": float(np.mean(values, axis=0)[1]),
                     "windows": len(values)} for name, values in errors.items()}
    _json(Path(args.output) / "baseline_metrics.json", result)
    print(json.dumps(result, indent=2))


def _train_prediction(args) -> None:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = train_prediction(Path(args.data), Path(args.intention_checkpoint),
                              Path(args.output), args.epochs, args.batch_size,
                              args.lr, args.seed, args.device, config)
    print(f"best_weighted_error_m={result['best_weighted_error_m']:.2f}; "
          f"saved to {args.output}")


def _eval_prediction(args) -> None:
    result = evaluate_prediction(Path(args.data), Path(args.intention_checkpoint),
                                 Path(args.prediction_checkpoint), Path(args.output),
                                 args.batch_size, args.device)
    print(json.dumps(result, indent=2))


def _rollout(args) -> None:
    result = rolling_evaluation(Path(args.track), Path(args.intention_checkpoint),
                                Path(args.prediction_checkpoint), Path(args.output),
                                args.update_seconds, args.device)
    print(f"forecasts={result['forecasts']} ADE={result['ADE_m']:.2f}m "
          f"FDE={result['FDE_m']:.2f}m; saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate paper-scale 3-D tracks")
    generate.add_argument("--config", default="configs/paper_scene.yaml")
    generate.add_argument("--output", default="outputs/paper_tracks")
    generate.add_argument("--tracks", type=int, default=100)
    generate.add_argument("--seed", type=int, default=1)
    generate.set_defaults(run=_generate)
    intention = commands.add_parser("train-intention", help="Train nine-class BiLSTM")
    intention.add_argument("--data", default="outputs/paper_tracks")
    intention.add_argument("--output", default="outputs/intention")
    intention.add_argument("--epochs", type=int, default=70)
    intention.add_argument("--batch-size", type=int, default=256)
    intention.add_argument("--lr", type=float, default=1e-3)
    intention.add_argument("--seed", type=int, default=1)
    intention.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    intention.set_defaults(run=_train_intention)
    baseline = commands.add_parser("eval-baselines", help="Evaluate CV and CA on held-out tracks")
    baseline.add_argument("--data", default="outputs/paper_tracks")
    baseline.add_argument("--output", default="outputs/baselines")
    baseline.add_argument("--dt", type=float, default=0.1)
    baseline.add_argument("--seed", type=int, default=1)
    baseline.set_defaults(run=_eval_baselines)
    prediction = commands.add_parser("train-prediction", help="Train graph/candidate/GRU predictor")
    prediction.add_argument("--data", required=True)
    prediction.add_argument("--intention-checkpoint", required=True)
    prediction.add_argument("--config", default="configs/prediction.yaml")
    prediction.add_argument("--output", default="outputs/prediction")
    prediction.add_argument("--epochs", type=int, default=200)
    prediction.add_argument("--batch-size", type=int, default=256)
    prediction.add_argument("--lr", type=float, default=1e-3)
    prediction.add_argument("--seed", type=int, default=1)
    prediction.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    prediction.set_defaults(run=_train_prediction)
    evaluation = commands.add_parser("eval-prediction", help="Evaluate on independent generated tracks")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--intention-checkpoint", required=True)
    evaluation.add_argument("--prediction-checkpoint", required=True)
    evaluation.add_argument("--output", default="outputs/prediction_test")
    evaluation.add_argument("--batch-size", type=int, default=256)
    evaluation.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    evaluation.set_defaults(run=_eval_prediction)
    rollout = commands.add_parser("rollout", help="Refresh forecast as new states arrive")
    rollout.add_argument("--track", required=True)
    rollout.add_argument("--intention-checkpoint", required=True)
    rollout.add_argument("--prediction-checkpoint", required=True)
    rollout.add_argument("--output", default="outputs/rolling_test")
    rollout.add_argument("--update-seconds", type=float, default=0.5)
    rollout.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    rollout.set_defaults(run=_rollout)
    args = parser.parse_args()
    if args.command == "generate" and args.tracks < 3:
        parser.error("Need at least three tracks")
    if args.command in ("train-intention", "train-prediction") and args.epochs < 1:
        parser.error("Need at least one training epoch")
    if hasattr(args, "batch_size") and args.batch_size < 1:
        parser.error("Batch size must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
