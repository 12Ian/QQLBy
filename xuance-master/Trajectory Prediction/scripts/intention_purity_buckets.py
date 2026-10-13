"""Purity-bucketed intention accuracy (IV-A label-noise floor measurement).

Motivation: the semantic9 window label is the *majority vote* over the 20
per-step labels of the 10 s observation window, while maneuver primitives last
only ``primitive_duration_range`` = 2-6 s (engineering default; the paper does
not disclose its segment-duration bounds).  A 10 s window therefore spans
~2.5 primitives on average and carries 1-2 full distractor maneuvers under a
single label.  This script measures how much of the observed accuracy deficit
is explained by that intrinsic label ambiguity.

For every window of the (balanced) reporting split it computes

    purity = (# steps whose per-step semantic label equals the window label) / 20

then reports accuracy per purity bucket, per class, and pre/post evasion
trigger, using the trained checkpoint.  Monotonically rising accuracy with
purity -- and a much higher accuracy in the purity=1.0 bucket -- localises the
deficit to label noise rather than model capacity.

Usage:
    python scripts/intention_purity_buckets.py \
        --data data/engagement_intention.npz \
        --checkpoint outputs/run/intention/best.pt --device cuda:0
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import default_collate

from trajectory_prediction.config import Config
from trajectory_prediction.data import Windows, intention_label_source, load_tracks, validate_split
from trajectory_prediction.engine import (choose_device, configure_runtime, load_checkpoint,
                                          to_device)
from trajectory_prediction.metrics import CLASS_NAMES
from trajectory_prediction.model import build_forecaster

BUCKETS = ((1.0, 1.01, "purity = 1.00"),
           (0.90, 1.0, "0.90-1.00"),
           (0.75, 0.90, "0.75-0.90"),
           (0.50, 0.75, "0.50-0.75"),
           (0.00, 0.50, "< 0.50"))


def bucket_of(purity):
    for lo, hi, name in BUCKETS:
        if lo <= purity < hi:
            return name
    return BUCKETS[-1][2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    cfg = Config(**checkpoint["config"]).validate()
    configure_runtime(cfg)
    device = choose_device(args.device)

    tracks = load_tracks(args.data, cfg.dt, cfg.observation_steps + cfg.prediction_steps,
                         cfg.intention_classes)
    validate_split(tracks, checkpoint["splits"], bool(getattr(cfg, "holdout_test", True)))
    split = args.split if checkpoint["splits"].get(args.split) else "validation"
    dataset = Windows(tracks, checkpoint["splits"][split], cfg)

    model = build_forecaster(cfg, checkpoint["normalizer"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    task = getattr(cfg, "intention_task", "semantic9")
    obs = cfg.observation_steps

    # --- per-window purity from the same per-step label source the dataset uses
    purity = np.full(len(dataset), np.nan)
    for k in range(len(dataset)):
        i, start = dataset.index[k]
        track = tracks[i]
        stop = start + obs
        source = np.asarray(intention_label_source(track, task), dtype=np.int64)
        window = source[start:stop]
        if window.size == 0:
            continue
        counts = np.bincount(window, minlength=cfg.intention_classes)
        majority = int(counts.argmax())
        purity[k] = float((window == majority).mean()) if counts[majority] else np.nan

    # --- forward pass in index order (no shuffle), collect predictions
    preds = np.full(len(dataset), -1, dtype=np.int64)
    labels = np.full(len(dataset), -1, dtype=np.int64)
    evaded = np.zeros(len(dataset), dtype=bool)
    with torch.no_grad():
        for k0 in range(0, len(dataset), args.batch_size):
            chunk = [dataset[k] for k in range(k0, min(k0 + args.batch_size, len(dataset)))]
            batch = to_device(default_collate(chunk), device)
            output = model(batch["observed"], context=batch.get("context"),
                           context_mask=batch.get("context_mask"),
                           engagement=batch.get("engagement"))
            guess = output["logits"].argmax(-1).cpu().numpy()
            lab = batch["label"].cpu().numpy()
            sl = slice(k0, k0 + len(chunk))
            preds[sl] = guess
            labels[sl] = lab
            if "evaded" in batch:
                evaded[sl] = batch["evaded"].cpu().numpy().astype(bool)

    known = labels >= 0
    correct = preds == labels
    result = {"split": split, "windows": int(known.sum()),
              "overall_accuracy": float(correct[known].mean()),
              "mean_purity": float(np.nanmean(purity[known])),
              "purity_buckets": {}, "per_class": {}, "evasion": {}}

    print(f"split={split}  labelled windows={int(known.sum())}  "
          f"overall accuracy={result['overall_accuracy']:.3f}  "
          f"mean purity={result['mean_purity']:.3f}")
    print()
    print(f"{'bucket':>12} {'windows':>8} {'share':>7} {'accuracy':>9}")
    for lo, hi, name in BUCKETS:
        mask = known & (purity >= lo) & (purity < hi)
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n else float("nan")
        result["purity_buckets"][name] = {"windows": n,
                                          "share": n / max(1, int(known.sum())),
                                          "accuracy": acc}
        print(f"{name:>12} {n:>8} {n / max(1, int(known.sum())):>6.1%} {acc:>9.3f}")
    print()
    print(f"{'class':>10} {'windows':>8} {'mean_purity':>12} {'accuracy':>9}")
    for c in range(cfg.intention_classes):
        mask = known & (labels == c)
        n = int(mask.sum())
        row = {"windows": n,
               "mean_purity": float(np.nanmean(purity[mask])) if n else None,
               "accuracy": float(correct[mask].mean()) if n else None}
        result["per_class"][CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)] = row
        print(f"{CLASS_NAMES[c] if c < len(CLASS_NAMES) else c:>10} {n:>8} "
              f"{row['mean_purity']:>12.3f} {row['accuracy']:>9.3f}")
    if evaded[known].any():
        print()
        for flag, name in ((False, "pre-evasion"), (True, "evasion-active")):
            mask = known & (evaded == flag)
            if mask.sum():
                acc = float(correct[mask].mean())
                result["evasion"][name] = {"windows": int(mask.sum()), "accuracy": acc,
                                           "mean_purity": float(np.nanmean(purity[mask]))}
                print(f"{name:>14}: windows={int(mask.sum())}  accuracy={acc:.3f}  "
                      f"mean_purity={result['evasion'][name]['mean_purity']:.3f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=1))
        print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
