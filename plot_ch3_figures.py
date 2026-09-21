"""Chapter-3 strong-evader figure set. Reads p3_results/eval_*.json, writes PNGs to
p3_results/curves/. Validated dataviz palette. Figures: N-sweep (success + outcome
decomposition), N=3 density generalization, evader-speed curve, N=3 reward ablation."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, RED, GREEN, VIOLET, AMBER = "#2a78d6", "#e34948", "#008300", "#4a3aa7", "#eda100"
INK, MUTED, GRID, SURF = "#0b0b0b", "#6b6f76", "#e1e0d9", "#fcfcfb"
GREY = "#b9b8b2"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7"})

def ev(name, d="medium"):
    try: return json.load(open(f"p3_results/eval_{name}_{d}.json"))
    except: return None
def vals(names, key="success_rate", d="medium"):
    out = []
    for n in names:
        r = ev(n, d)
        if r and r.get(key) is not None: out.append(r[key] * 100)
    return out

def style(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)

NS = [(3, ["s_N3_s1","s_N3_s2","s_N3_s3"]), (5, ["s_N5_s1","s_N5_s2","s_N5_s3"]),
      (6, ["s_full_s1","s_full_s2","s_full_s3"]), (8, ["s_N8_s1","s_N8_s2"])]

# ---------- Fig 1: N-sweep, success + outcome decomposition ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150)
xs = [str(n) for n, _ in NS]
succ = [np.mean(vals(s)) for _, s in NS]
sstd = [np.std(vals(s)) for _, s in NS]
cols = [RED if n == 3 else BLUE for n, _ in NS]
bars = a1.bar(xs, succ, yerr=sstd, color=cols, width=0.62, capsize=4,
              error_kw=dict(ecolor=MUTED, lw=1.2))
for x, s in zip(xs, succ): a1.text(x, s + 3.5, f"{s:.0f}%", ha="center", fontsize=10, color=INK, fontweight="600")
for _, seeds in NS:
    n = _; xi = xs.index(str(n))
    for v in vals(seeds): a1.plot(xi, v, "o", ms=4, mfc="white", mec=MUTED, mew=1, zorder=5)
a1.set_ylim(0, 105); a1.set_xlabel("Number of pursuers N", color=INK)
a1.set_ylabel("Capture success rate (%)", color=INK)
a1.set_title("(a) Success vs team size — N=3 is best", loc="left", color=INK, fontsize=11, pad=8)
style(a1)
# outcome decomposition (success / collision / escape)
S = [np.mean(vals(s, "success_rate")) for _, s in NS]
C = [np.mean(vals(s, "collision_rate")) for _, s in NS]
T = [np.mean(vals(s, "timeout_rate")) for _, s in NS]
a2.bar(xs, S, color=GREEN, width=0.62, label="Captured")
a2.bar(xs, C, bottom=S, color=RED, width=0.62, label="Collision")
a2.bar(xs, T, bottom=np.array(S)+np.array(C), color=AMBER, width=0.62, label="Evaded")
a2.set_ylim(0, 105); a2.set_xlabel("Number of pursuers N", color=INK)
a2.set_ylabel("Episode outcome (%)", color=INK)
a2.set_title("(b) Failure mode — large teams collide", loc="left", color=INK, fontsize=11, pad=8)
a2.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
style(a2)
fig.suptitle("Cooperative pursuit vs a strong evader (λ≈0.9, @medium) — pursuer-count sweep",
             fontsize=12.5, color=INK, y=1.02)
fig.tight_layout(); fig.savefig("p3_results/curves/fig_nsweep.png", bbox_inches="tight", facecolor=SURF)
plt.close(fig)

# ---------- Fig 2: N=3 density generalization ----------
fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
dens = ["empty", "open", "medium", "complex"]
N3 = ["s_N3_s1", "s_N3_s2", "s_N3_s3"]
dv = [np.mean(vals(N3, d=d)) for d in dens]
dc = [BLUE, BLUE, RED, GREY]
ax.bar(dens, dv, color=dc, width=0.6)
for i, d in enumerate(dens):
    ax.text(i, dv[i] + 3, f"{dv[i]:.0f}%", ha="center", color=INK, fontweight="600")
    for v in vals(N3, d=d): ax.plot(i, v, "o", ms=4, mfc="white", mec=MUTED, mew=1, zorder=5)
ax.set_ylim(0, 108); ax.set_ylabel("Capture success rate (%)", color=INK)
ax.set_xlabel("Obstacle density (trained @medium)", color=INK)
ax.set_title("N=3 flagship — density generalization", loc="left", color=INK, fontsize=12, pad=8)
ax.text(3, 6, "line-of-sight\nbroken", ha="center", fontsize=8.5, color=MUTED, style="italic")
style(ax)
fig.tight_layout(); fig.savefig("p3_results/curves/fig_density.png", bbox_inches="tight", facecolor=SURF)
plt.close(fig)

# ---------- Fig 3: evader-speed curve (N=6 sweep; N=3 vmax10 marker) ----------
fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
V = [8, 9, 10, 11, 12]
sp = [np.mean(vals([f"sv_p{v}", f"s_lam_V{v}_s2"])) for v in V]
lam = [v / 11.0 for v in V]  # pursuer max speed = 11
ax.axvspan(11, 12.4, color=RED, alpha=0.06)
ax.axvline(11, color=MUTED, ls="--", lw=1)
ax.text(11.05, 92, "λ=1 (parity)", fontsize=8.5, color=MUTED)
ax.plot(V, sp, "-o", color=BLUE, lw=2.2, ms=6, mfc=BLUE, mec="white", mew=1)
for v, s in zip(V, sp): ax.text(v, s + 4, f"{s:.0f}%", ha="center", fontsize=9, color=INK)
ax.set_xlim(7.6, 12.4); ax.set_ylim(-3, 100)
ax.set_xlabel("Evader max speed (pursuer max = 11)", color=INK)
ax.set_ylabel("Capture success rate (%)", color=INK)
ax.set_title("Success vs evader speed — capability boundary", loc="left", color=INK, fontsize=12, pad=8)
ax.text(11.7, 45, "faster evader\nescapes\n(physics limit)", ha="center", fontsize=8.5, color=RED, style="italic")
style(ax); ax.grid(True, axis="both", color=GRID, lw=0.7)
fig.tight_layout(); fig.savefig("p3_results/curves/fig_speed.png", bbox_inches="tight", facecolor=SURF)
plt.close(fig)

# ---------- Fig 4: N=3 reward ablation ----------
fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
BASE = 80.0
terms = [("r_safe\n(collision)", ["s3_ab_r_safe_s1","s3_ab_r_safe_s2"]),
         ("closure\n(noclose)", ["s3_noclose_s1","s3_noclose_s2"]),
         ("r_pos\n(position)", ["s3_ab_r_pos_s1","s3_ab_r_pos_s2"]),
         ("r_finish", ["s3_ab_r_finish_s1","s3_ab_r_finish_s2"]),
         ("r_gap", ["s3_ab_r_gap_s1","s3_ab_r_gap_s2"]),
         ("r_near", ["s3_ab_r_near_s1","s3_ab_r_near_s2"]),
         ("r_closure", ["s3_ab_r_closure_s1","s3_ab_r_closure_s2"])]
labels = [t[0] for t in terms]
means = [np.mean(vals(t[1])) for t in terms]
order = np.argsort(means)  # ascending -> biggest drop (lowest) at bottom
labels = [labels[i] for i in order]; means = [means[i] for i in order]
segs = [terms[i][1] for i in order]
y = np.arange(len(labels))
ax.axvline(BASE, color=GREEN, lw=1.6, ls="--"); ax.text(BASE + 0.5, len(labels) - 0.3, "full reward 80%", color=GREEN, fontsize=9)
ax.barh(y, means, color=[RED if m < 60 else BLUE for m in means], height=0.6)
for yi, m, sg in zip(y, means, segs):
    ax.text(m + 1, yi, f"{m:.0f}%", va="center", fontsize=9.5, color=INK, fontweight="600")
    for v in vals(sg): ax.plot(v, yi, "o", ms=4, mfc="white", mec=MUTED, mew=1, zorder=5)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlim(0, 100); ax.set_xlabel("Success rate with term removed (%)", color=INK)
ax.set_title("N=3 reward ablation — collision-avoidance matters most", loc="left", color=INK, fontsize=12, pad=8)
ax.grid(True, axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
for sp2 in ["top", "right"]: ax.spines[sp2].set_visible(False)
ax.tick_params(colors=MUTED, labelsize=9)
fig.tight_layout(); fig.savefig("p3_results/curves/fig_ablation.png", bbox_inches="tight", facecolor=SURF)
plt.close(fig)

print("saved fig_nsweep, fig_density, fig_speed, fig_ablation")
print("nsweep succ:", [f"{s:.0f}" for s in succ])
print("density:", [f"{d:.0f}" for d in dv])
print("speed:", [f"{s:.0f}" for s in sp])
print("ablation(sorted):", list(zip(labels, [f"{m:.0f}" for m in means])))
