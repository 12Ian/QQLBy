"""Extract DENSE training-return convergence: Train-Results/Episode-Rewards/rank_0
(~30-45k per-episode points), rolling-mean smoothed, downsampled to ~300 plot points.
One CSV per run: p3_results/curves/<run>_train.csv  cols: step,reward,reward_raw"""
import glob, os, csv, sys
import numpy as np
from scipy.ndimage import uniform_filter1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = sys.argv[1:] or [
    "s_N3_s1", "s_N3_s2", "s_N5_s1", "s_N5_s2",
    "s_full_s1", "s_full_s2", "s_full_s3", "s_N8_s1", "s_N8_s2",
]
TAG = "Train-Results/Episode-Rewards/rank_0"
WIN = 300      # rolling window in episodes
NPLOT = 300    # downsampled plot points
os.makedirs("p3_results/curves", exist_ok=True)

for run in RUNS:
    files = sorted(glob.glob(f"logs/maddpg/apollonius_3d_{run}/**/events.out.tfevents*", recursive=True))
    s2v = {}
    for f in files:
        try:
            ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
        except Exception:
            continue
        if TAG in ea.Tags().get("scalars", []):
            for s in ea.Scalars(TAG): s2v[s.step] = s.value
    if not s2v:
        print(f"{run}: no train reward (files={len(files)})"); continue
    steps = np.array(sorted(s2v)); vals = np.array([s2v[s] for s in steps], float)
    # centered rolling mean; mode="nearest" extends edge values -> no end spike
    sm = uniform_filter1d(vals, size=WIN, mode="nearest")
    idx = np.linspace(0, len(steps) - 1, min(NPLOT, len(steps))).astype(int)
    with open(f"p3_results/curves/{run}_train.csv", "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["step", "reward", "reward_raw"])
        for i in idx: w.writerow([int(steps[i]), round(float(sm[i]), 3), round(float(vals[i]), 3)])
    print(f"{run}: {len(steps)} eps -> {len(idx)} pts | smoothed {sm[0]:.1f} -> {sm[-1]:.1f}")
