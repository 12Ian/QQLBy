"""Lambda (speed-ratio) sweep curve from p3_results eval JSONs.
x = evader speed cap V / pursuer mean speed (10); pursuers are 9-11 so V=10 is
near speed parity (10% top-speed margin), not strict parity."""
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "p3_results"
VS = [4, 6, 8, 10]


def pts(v):
    ss = []
    for f in sorted(glob.glob(f"{D}/eval_lsweep_V{v}_s*_medium.json")):
        ss.append(json.load(open(f))["success_rate"])
    return ss


lam = [v / 10.0 for v in VS]
means, mins, maxs, ns = [], [], [], []
for v in VS:
    ss = pts(v)
    means.append(np.mean(ss) * 100)
    mins.append(min(ss) * 100)
    maxs.append(max(ss) * 100)
    ns.append(len(ss))

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
ax.plot(lam, means, "o-", color="#4C72B0", lw=2, ms=7, label="mean (2 seeds)")
ax.fill_between(lam, mins, maxs, color="#4C72B0", alpha=0.18, label="seed range")
for x, y, v in zip(lam, means, VS):
    ax.annotate(f"V={v}", (x, y), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9)
ax.axvline(1.0, color="#C44E52", ls="--", lw=1, alpha=0.6)
ax.text(0.985, 8, "speed parity", rotation=90, color="#C44E52", fontsize=9, ha="right")
ax.set_xlabel("speed ratio λ = evader speed / pursuer mean speed")
ax.set_ylabel("success rate (%) @medium")
ax.set_title("Success vs evader speed ratio (constant-speed evader)")
ax.set_ylim(0, 105)
ax.grid(alpha=0.3)
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(f"{D}/fig_lambda_curve.png")
print("saved fig_lambda_curve.png; seeds per point:", ns)
