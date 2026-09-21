"""Extract MADDPG strong-evader CONVERGENCE curves from tensorboard: the periodic
evaluation Win-Rate (success) and Episode-Reward (avg test return) vs env steps,
aggregated across the many per-run event files. One CSV per run:
p3_results/curves/<run>.csv  cols: step,winrate,ret"""
import glob, os, csv, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = sys.argv[1:] or [
    "s_N3_s1", "s_N3_s2", "s_N5_s1", "s_N5_s2",
    "s_full_s1", "s_full_s2", "s_full_s3", "s_N8_s1", "s_N8_s2",
]
WIN = "Test-Results/Win-Rate-Level-4"
RET = "Test-Results/Episode-Rewards-Level-4"
os.makedirs("p3_results/curves", exist_ok=True)

for run in RUNS:
    files = sorted(glob.glob(f"logs/maddpg/apollonius_3d_{run}/**/events.out.tfevents*", recursive=True))
    win, ret = {}, {}
    for f in files:
        try:
            ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
        except Exception:
            continue
        tags = ea.Tags().get("scalars", [])
        if WIN in tags:
            for s in ea.Scalars(WIN): win[s.step] = s.value
        if RET in tags:
            for s in ea.Scalars(RET): ret[s.step] = s.value
    steps = sorted(set(win) | set(ret))
    if not steps:
        print(f"{run}: no test scalars (files={len(files)})"); continue
    with open(f"p3_results/curves/{run}.csv", "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["step", "winrate", "ret"])
        for st in steps:
            w.writerow([st, win.get(st, ""), ret.get(st, "")])
    wl = [win[s] for s in sorted(win)]
    print(f"{run}: {len(steps)} eval pts | winrate {wl[0]*100:.0f}%->{wl[-1]*100:.0f}% (max {max(wl)*100:.0f}%)" if wl else f"{run}: no winrate")
