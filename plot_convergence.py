"""Strong-evader MADDPG convergence figure, v2.
(a) eval success-rate vs steps (periodic eval, 21 pts, markers) - cross-N comparable.
(b) DENSE smoothed training return vs steps (~300 pts, continuous) - convergence dynamics.
Seed mean + min-max band; train curves interpolated onto a common step grid to average.
N=5 flagship emphasized. Saves p3_results/curves/convergence_strong.png"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CFG = [
    ("N=3", "#2a78d6", ["s_N3_s1", "s_N3_s2"], False),
    ("N=5 (flagship)", "#e34948", ["s_N5_s1", "s_N5_s2", "s_N5_s3"], True),
    ("N=6", "#008300", ["s_full_s1", "s_full_s2", "s_full_s3"], False),
    ("N=8", "#4a3aa7", ["s_N8_s1", "s_N8_s2"], False),
]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
GRIDX = np.linspace(0, 5e6, 300)   # common grid for dense-return averaging

def load_eval(run):
    p = f"p3_results/curves/{run}.csv"
    if not os.path.exists(p): return None
    st, wr = [], []
    for r in csv.DictReader(open(p)):
        st.append(int(float(r["step"])))
        wr.append(float(r["winrate"]) if r["winrate"] else np.nan)
    return np.array(st), np.array(wr)

def load_train(run):
    p = f"p3_results/curves/{run}_train.csv"
    if not os.path.exists(p): return None
    st, rw = [], []
    for r in csv.DictReader(open(p)):
        st.append(int(float(r["step"]))); rw.append(float(r["reward"]))
    st, rw = np.array(st), np.array(rw)
    return np.interp(GRIDX, st, rw)   # onto common grid

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

# ---- (a) success rate (eval, sparse markers) ----
ax = axes[0]
for label, color, seeds, emph in CFG:
    curves, steps = [], None
    for s in seeds:
        d = load_eval(s)
        if d is None: continue
        steps, wr = d; curves.append(wr)
    if not curves: continue
    M = np.vstack([c[:len(steps)] for c in curves]) * 100
    mean = np.nanmean(M, axis=0)
    x = steps / 1e6
    lw = 2.6 if emph else 1.6
    if M.shape[0] > 1:
        ax.fill_between(x, np.nanmin(M, 0), np.nanmax(M, 0), color=color,
                        alpha=0.10 if emph else 0.06, lw=0, zorder=2)
    ax.plot(x, mean, color=color, lw=lw, marker="o", ms=4.5 if emph else 3.2,
            mfc=color, mec="white", mew=0.5, label=label, zorder=5 if emph else 3,
            solid_capstyle="round")
ax.set_ylabel("Capture success rate (%)", fontsize=10, color=INK)
ax.set_title("(a) Success convergence — periodic evaluation", fontsize=11, color=INK, loc="left", pad=8)
ax.set_ylim(-2, 100)
ax.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=INK)

# ---- (b) dense training return ----
ax = axes[1]
for label, color, seeds, emph in CFG:
    curves = [load_train(s) for s in seeds]
    curves = [c for c in curves if c is not None]
    if not curves: continue
    M = np.vstack(curves)
    mean = M.mean(0); x = GRIDX / 1e6
    lw = 2.6 if emph else 1.6
    if M.shape[0] > 1:
        ax.fill_between(x, M.min(0), M.max(0), color=color,
                        alpha=0.10 if emph else 0.06, lw=0, zorder=2)
    ax.plot(x, mean, color=color, lw=lw, label=label, zorder=5 if emph else 3,
            solid_capstyle="round")
ax.set_ylabel("Avg training return (smoothed)", fontsize=10, color=INK)
ax.set_title("(b) Return convergence — dense (≈300 pts/seed)", fontsize=11, color=INK, loc="left", pad=8)

for ax in axes:
    ax.set_xlabel("Environment steps (millions)", fontsize=10, color=INK)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.9); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9); ax.set_xlim(0, 5)

fig.suptitle("3D Multi-UAV Cooperative Pursuit vs a Strong Evader (λ≈0.9) — MADDPG Convergence",
             fontsize=12.5, color=INK, y=1.02)
fig.tight_layout()
out = "p3_results/curves/convergence_strong.png"
fig.savefig(out, bbox_inches="tight", facecolor="#fcfcfb")
print("saved", out)
