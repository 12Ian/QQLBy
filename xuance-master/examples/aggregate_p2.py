"""Aggregate the P2 core-campaign per-seed eval JSONs into mean +/- std tables and
publication figures (method density curve, closure ablation, density generalization)."""
import json
import os
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "p2_results"
DENS = ["empty", "open", "medium", "complex"]
SEEDS = [1, 2, 3]


def agg(exp, dens):
    sr, cr, st = [], [], []
    for s in SEEDS:
        f = os.path.join(D, f"eval_{exp}_s{s}_{dens}.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        sr.append(d["success_rate"])
        cr.append(d["collision_rate"])
        if d.get("mean_capture_steps") is not None:
            st.append(d["mean_capture_steps"])

    def ms(x):
        if not x:
            return (None, None)
        return (statistics.mean(x), statistics.pstdev(x) if len(x) > 1 else 0.0)
    return {"success": ms(sr), "collision": ms(cr), "steps": ms(st), "n": len(sr)}


def fmt(t, pct=True):
    m, s = t
    if m is None:
        return "—"
    return f"{m*100:.1f}±{s*100:.1f}%" if pct else f"{m:.1f}±{s:.1f}"


# ---- table ----
print("\n## 统计表（3 种子，均值±标准差）\n")
print("| 实验 | 密度 | 成功率 | 碰撞率 | 平均围捕步数 |")
print("|---|---|---|---|---|")
rows = {}
for exp in ["full", "noclose", "domrand"]:
    for dens in DENS:
        a = agg(exp, dens)
        if a["n"] == 0:
            continue
        rows[(exp, dens)] = a
        print(f"| {exp} | {dens} | {fmt(a['success'])} | {fmt(a['collision'])} | {fmt(a['steps'], pct=False)} |")


def bars(ax, labels, groups, colors):
    # groups: list of (name, [(mean,std) per label]); grouped bars
    n = len(labels)
    w = 0.8 / max(len(groups), 1)
    x = np.arange(n)
    for gi, (gname, vals) in enumerate(groups):
        means = [v[0] * 100 if v[0] is not None else 0 for v in vals]
        stds = [v[1] * 100 if v[1] is not None else 0 for v in vals]
        ax.bar(x + gi * w, means, w, yerr=stds, capsize=4, label=gname, color=colors[gi])
    ax.set_xticks(x + w * (len(groups) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("success rate (%)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)


# ---- fig 1: method density curve (full) ----
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
vals = [rows[("full", d)]["success"] for d in DENS]
bars(ax, DENS, [("method (trained @medium)", vals)], ["#4C72B0"])
ax.set_title("Method success vs obstacle density (3 seeds)")
plt.tight_layout(); plt.savefig(f"{D}/fig_method_density.png"); print("saved fig_method_density.png")

# ---- fig 2: closure ablation @medium ----
fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=130)
fa = rows[("full", "medium")]; na = rows[("noclose", "medium")]
ax.bar([0, 1], [fa["success"][0]*100, na["success"][0]*100],
       yerr=[fa["success"][1]*100, na["success"][1]*100], capsize=5,
       color=["#55A868", "#C44E52"])
ax.set_xticks([0, 1]); ax.set_xticklabels(["with closure\n(full)", "no closure\n(ablation)"])
ax.set_ylabel("success rate (%) @medium")
for i, a in enumerate([fa, na]):
    ax.text(i, a["success"][0]*100 + 2, f"{a['steps'][0]:.0f} steps", ha="center")
ax.set_title("Closure reward ablation @medium (3 seeds)")
plt.tight_layout(); plt.savefig(f"{D}/fig_closure_ablation.png"); print("saved fig_closure_ablation.png")

# ---- fig 3: density generalization (full vs domrand) ----
fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=130)
full_v = [rows[("full", d)]["success"] for d in DENS]
dr_v = [rows[("domrand", d)]["success"] for d in DENS]
bars(ax, DENS, [("fixed-density (medium)", full_v), ("domain-randomized", dr_v)],
     ["#4C72B0", "#DD8452"])
ax.set_title("Cross-density generalization: fixed vs domain-randomized (3 seeds)")
plt.tight_layout(); plt.savefig(f"{D}/fig_density_generalization.png"); print("saved fig_density_generalization.png")
