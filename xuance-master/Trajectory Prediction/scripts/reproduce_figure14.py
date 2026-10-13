"""Reproduce Fig. 14 / Table VI: four composite maneuver scenarios.

Table VI (paper Section IV-B) fixes the initial conditions of four composite
maneuvers - Left-Climb, Left-Dive, Right-Climb, Right-Dive - at r0 = 5.3 km
with e_L,0 = [0.98, +-0.15, +-0.09]^T, VT = 300 m/s, VM = 360 m/s, To = 8 s,
Tp = 6 s, aT,max = 9 g, aM,max = 15 g and theta_FOV = 60 deg.  The paper
reports terminal errors of 150 / 202 / 81 / 173 m for those four cases.

This entry point simulates the four scenarios deterministically, predicts the
6 s horizon from the 8 s observation window with a trained checkpoint (or the
constant-velocity baseline when no checkpoint is given), and prints the
"measured vs paper" comparison table.

Usage:
    python scripts/reproduce_figure14.py --output outputs/figure14 \
        [--checkpoint outputs/run/best.pt] [--device cpu|auto|cuda:0]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from trajectory_prediction.config import Config
from trajectory_prediction.data import load_tracks, relative_features, window_intention_label
from trajectory_prediction.engagement_scenarios import (G0, FIGURE14_OBSERVATION_S,
                                                        FIGURE14_PREDICTION_S, FIGURE14_SCENARIOS,
                                                        simulate_engagement_batch)
from trajectory_prediction.inference import Predictor
from trajectory_prediction.model import constant_velocity

# Paper Fig. 14 terminal errors (m), in Table VI row order.
PAPER_TERMINAL_ERRORS_M = {"Left-Climb": 150., "Left-Dive": 202.,
                           "Right-Climb": 81., "Right-Dive": 173.}
DT = .1


def scenario_conditions():
    conditions = []
    for scenario in FIGURE14_SCENARIOS:
        conditions.append(dict(scenario,
                               r0_m=scenario["r0_km"] * 1e3,
                               vT=300., vM=360.,
                               a_yaw=scenario["yaw_sign"] * abs(scenario["a_phi_0"]),
                               a_pitch=scenario["pitch_sign"] * abs(scenario["a_theta_0"])))
    return conditions


def predict_window(model_or_predictor, cfg, observed, context, context_mask, engagement):
    """Return the [P,3] predicted positions for one observed window."""
    if model_or_predictor is None:
        state = torch.from_numpy(observed[None].astype(np.float32))
        return constant_velocity(state, cfg.prediction_steps, cfg.dt)[0].numpy()
    result = model_or_predictor.predict(observed, context=context, context_mask=context_mask,
                                        engagement=engagement)
    return np.asarray(result["trajectory"], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    observation_steps = int(round(FIGURE14_OBSERVATION_S / DT))
    prediction_steps = int(round(FIGURE14_PREDICTION_S / DT))
    steps = observation_steps + prediction_steps

    data = output / "table_vi_scenarios.npz"
    if data.exists():
        data.unlink()
    metadata = simulate_engagement_batch(data, scenario_conditions(), steps=steps, dt=DT,
                                         seed=args.seed, aT_max=9. * G0, aM_max=15. * G0)
    (output / "table_vi_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    predictor, cfg = None, None
    if args.checkpoint:
        predictor = Predictor(args.checkpoint, args.device)
        cfg = predictor.cfg
    tracks = load_tracks(data, DT, steps)

    rows = []
    for track in tracks:
        observed = track.states[:observation_steps].astype(np.float32)
        truth = track.states[observation_steps:, :3].astype(np.float64)
        # Track stores peers as [T,N,6] / [T,N]; Predictor.predict expects the
        # entity axis first, [N,T,6] / [N,T].
        context = track.context[:observation_steps].transpose(1, 0, 2)
        mask = track.context_mask[:observation_steps].T
        engagement = track.engagement[:observation_steps]
        horizon = prediction_steps
        local_cfg = cfg
        if local_cfg is None or local_cfg.prediction_steps != horizon:
            local_cfg = Config(**{**(vars(cfg) if cfg else {}), "observation_steps": observation_steps,
                                  "prediction_steps": horizon, "dt": DT})
        prediction = predict_window(predictor, local_cfg, observed, context, mask, engagement)
        error = np.linalg.norm(prediction - truth, axis=-1)
        name = track.scenario
        paper = PAPER_TERMINAL_ERRORS_M.get(name)
        rows.append(dict(scenario=name, terminal_error_m=float(error[-1]),
                         ade_m=float(error.mean()), paper_terminal_error_m=paper,
                         absolute_gap_m=None if paper is None else float(error[-1] - paper),
                         relative_gap_percent=None if not paper else float(100 * (error[-1] - paper) / paper)))

    csv_path = output / "figure14_terminal_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "figure14_summary.json").write_text(json.dumps(
        {"predictor": "checkpoint" if predictor else "constant_velocity_baseline",
         "checkpoint": args.checkpoint, "rows": rows}, indent=2), encoding="utf-8")

    width = max(len(row["scenario"]) for row in rows)
    print(f"{'scenario'.ljust(width)}  {'measured (m)':>13}  {'paper (m)':>10}  {'gap':>8}")
    for row in rows:
        gap = "n/a" if row["absolute_gap_m"] is None else f"{row['absolute_gap_m']:+.1f}"
        print(f"{row['scenario'].ljust(width)}  {row['terminal_error_m']:13.1f}  "
              f"{str(row['paper_terminal_error_m']):>10}  {gap:>8}")
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
