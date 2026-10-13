"""Reward-term ablation bar chart (Exp A): each term disabled vs the full method,
@medium, mean +/- seed-range. Reads p3_results eval JSONs."""
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "p3_results"
TERMS = ["r_pos", "r_gap", "r_near", "r_safe", "r_finish", "r_closure"]
LABELS = {"r_pos": "-r_pos\n(encircle)", "r_gap": "-r_gap\n(spacing)",
          "r_near": "-r_near\n(guide prog.)", "r_safe": "-r_safe\n(obstacle/mate)",
          "r_finish": "-r_finish\n(terminal)", "r_closure": "-r_closure\n(closure)"}


def succ(pattern):
    ss = []
    for f in sorted(glob.glob(f"{D}/{pattern}")):
        ss.append(json.load(open(f))["success_rate"] * 100)
    return ss


base = succ("eval_anchor_full_s*_medium.json") or succ("eval_full_s*_medium.json")
base_m = float(np.mean(base))

rows = []
for t in TERMS:
    ss = succ(f"eval_rewabl_{t}_s*_medium.json")
    rows.append((t, float(np.mean(ss)), min(ss), max(ss)))
rows.sort(key=lambda r: r[1])

fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=130)
x = np.arange(len(rows))
means = [r[1] for r in rows]
lo = [r[1] - r[2] for r in rows]
hi = [r[3] - r[1] for r in rows]
colors = ["#C44E52" if base_m - r[1] >= 8 else "#DD8452" if base_m - r[1] >= 3 else "#55A868" for r in rows]
ax.bar(x, means, yerr=[lo, hi], capsize=4, color=colors)
ax.axhline(base_m, color="#4C72B0", ls="--", lw=1.6, label=f"full method ({base_m:.0f}%)")
ax.set_xticks(x)
ax.set_xticklabels([LABELS[r[0]] for r in rows], fontsize=8.5)
ax.set_ylabel("success rate (%) @medium")
ax.set_ylim(0, 105)
ax.set_title("Reward-term ablation - remove one additive term (2 seeds)")
for i, r in enumerate(rows):
    ax.text(i, r[1] + max(hi[i], 2) + 1.5, f"{r[1]:.0f}%", ha="center", fontsize=8.5)
ax.legend(loc="lower right")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{D}/fig_reward_ablation.png")
print("saved fig_reward_ablation.png")
print("baseline full =", round(base_m, 1), "| ablations:", [(r[0], round(r[1], 1)) for r in rows])
