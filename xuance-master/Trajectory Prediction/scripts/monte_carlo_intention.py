"""Per-label Monte Carlo intention evaluation (paper IV-A protocol).

The paper builds the held-out test set with "1000 independent Monte Carlo
trials with distinct random seeds for each maneuver label".  This entry point
reproduces that protocol on top of a trained checkpoint:

  * the checkpoint's own splits are honoured (no scene leakage);
  * for every intention class the labelled windows of the reporting split are
    resampled with replacement, ``--trials`` times per label, each trial using
    its own seed, and pooled into one balanced test set;
  * accuracy, per-class precision/recall/F1, macro F1, balanced accuracy and
    the majority-class baseline are reported, together with the confusion
    matrix in the original and row-normalized form.

Usage:
    python scripts/monte_carlo_intention.py --data data/engagement_intention.npz \
        --checkpoint outputs/run/best.pt --trials 1000 --output outputs/mc.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from trajectory_prediction.config import Config
from trajectory_prediction.data import Windows, load_tracks, validate_split
from trajectory_prediction.engine import (choose_device, configure_runtime, load_checkpoint,
                                          make_loader, run_epoch)
from trajectory_prediction.learner import TrajectoryLearner
from trajectory_prediction.metrics import CLASS_NAMES, EvaluationDetails
from trajectory_prediction.model import build_forecaster


def class_indices(dataset):
    """Map each intention class to the window indices carrying that label."""
    grouped = {}
    for index in range(len(dataset)):
        label = int(dataset[index]["label"])
        if label >= 0:
            grouped.setdefault(label, []).append(index)
    return grouped


def monte_carlo_indices(grouped, trials, classes, seed):
    """Balanced resample: ``trials`` draws per label, each with its own seed."""
    rng = np.random.default_rng(seed)
    sampled, per_class = [], {}
    for label in range(classes):
        pool = grouped.get(label, [])
        per_class[label] = len(pool)
        if not pool:
            continue
        draws = rng.choice(len(pool), size=trials, replace=len(pool) < trials)
        sampled.extend(pool[int(k)] for k in draws)
    return sampled, per_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--split", default=None,
                        help="Reporting split; defaults to test, or validation when no test partition exists")
    parser.add_argument("--trials", type=int, default=1000, help="Independent trials per label (paper: 1000)")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    cfg = Config(**checkpoint["config"])
    device = choose_device(args.device)
    configure_runtime(cfg)
    split = args.split or ("test" if checkpoint["splits"].get("test") else "validation")
    tracks = load_tracks(args.data, cfg.dt, cfg.observation_steps + cfg.prediction_steps, cfg.intention_classes)
    validate_split(tracks, checkpoint["splits"], bool(getattr(cfg, "holdout_test", True)))

    dataset = Windows(tracks, checkpoint["splits"][split], cfg)
    grouped = class_indices(dataset)
    sampled, per_class_available = monte_carlo_indices(grouped, args.trials, cfg.intention_classes, args.seed)
    if not sampled:
        raise SystemExit(f"No labelled windows in split '{split}'; check the checkpoint splits and data")
    subset = torch.utils.data.Subset(dataset, sampled)

    model = build_forecaster(cfg, checkpoint["normalizer"]).to(device)
    model.load_state_dict(checkpoint["model"])
    learner = TrajectoryLearner(cfg, model, device)
    if "criterion" in checkpoint:
        learner.objective.load_state_dict(checkpoint["criterion"])
    details = EvaluationDetails(cfg)
    result = run_epoch(model, make_loader(subset, cfg, device), cfg, device,
                       include_confusion=True, learner=learner, details=details)
    details.finish(result)

    result.update(split=split, trials_per_label=args.trials, sampling_seed=args.seed,
                  windows_per_label_available={str(k): v for k, v in per_class_available.items()},
                  monte_carlo_windows=len(sampled),
                  class_names={str(i): (CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i))
                               for i in range(cfg.intention_classes)},
                  checkpoint=str(Path(args.checkpoint).resolve()),
                  protocol=dict(observation_steps=cfg.observation_steps, dt_s=cfg.dt,
                                intention_task=getattr(cfg, "intention_task", "semantic9")))
    printable = {key: (round(value, 4) if isinstance(value, float) else value)
                 for key, value in result.items()
                 if key in ("intention_accuracy", "intention_macro_f1", "intention_balanced_accuracy",
                            "intention_majority_baseline", "intention_classes_present",
                            "monte_carlo_windows", "labeled_samples")}
    print(json.dumps(printable, indent=2))
    print("windows per label (available):", per_class_available)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
