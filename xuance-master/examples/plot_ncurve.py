"""Success vs number of pursuers N (@medium), with N=4 flagged as a training-instability
outlier (learnable @empty 95%, but @medium training collapses). Reads p3_results JSONs."""
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "p3_results"


def succ(pattern):
    ss = []
    for f in sorted(glob.glob(f"{D}/{pattern}")):
        ss.append(json.load(open(f))["success_rate"] * 100)
    return ss


# N=6 anchor = full method re-eval; others = nsweep
data = {
    3: succ("eval_nsweep_N3_s*_medium.json"),
    4: succ("eval_nsweep_N4_s*_medium.json"),
    5: succ("eval_nsweep_N5_s*_medium.json"),
    6: succ("eval_anchor_full_s*_medium.json") or succ("eval_full_s*_medium.json"),
    8: succ("eval_nsweep_N8_s*_medium.json"),
}
Ns = sorted(data)
good = [n for n in Ns if n != 4]

fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=130)
gm = [np.mean(data[n]) for n in good]
glo = [np.mean(data[n]) - min(data[n]) for n in good]
ghi = [max(data[n]) - np.mean(data[n]) for n in good]
ax.errorbar(good, gm, yerr=[glo, ghi], fmt="o-", color="#4C72B0", lw=2, ms=8,
            capsize=4, label="converged (mean, seed range)")
# N=4 outlier
n4 = np.mean(data[4]) if data[4] else 0.0
ax.plot([4], [n4], "X", color="#C44E52", ms=14, label="N=4 (training diverged)")
ax.annotate("N=4: training collapse\n(learnable: 95% @empty)", (4, n4),
            textcoords="offset points", xytext=(12, 28), fontsize=8.5, color="#C44E52",
            arrowprops=dict(arrowstyle="->", color="#C44E52"))
ax.set_xlabel("number of pursuers N")
ax.set_ylabel("success rate (%) @medium")
ax.set_ylim(-3, 105)
ax.set_xticks(Ns)
ax.set_title("Success vs team size N (evader speed ratio ~0.6)")
ax.grid(alpha=0.3)
ax.legend(loc="center right")
plt.tight_layout()
plt.savefig(f"{D}/fig_ncurve.png")
print("saved fig_ncurve.png")
print("N-sweep:", {n: round(float(np.mean(v)), 1) if v else None for n, v in data.items()})
