"""Reward-convergence curves from XuanCe TensorBoard event files (tf-free reader).
Reads Train-Results/Episode-Rewards across all parallel envs, bins by step, plots:
  A) full method 3-seed convergence (mean +/- std band)
  B) closure vs no-closure convergence (innovation 2 diagnostic)
  C) N-sweep convergence (shows the N=4 training divergence)
  D) full-method per-term sub-reward evolution
"""
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG = "logs/maddpg"
OUT = "p3_results"
AGENTS = ["uav_0", "uav_1", "uav_2", "uav_3", "uav_4", "uav_5"]


def _scalars(d):
    ea = EventAccumulator(d)
    ea.Reload()
    tags = ea.Tags()["scalars"]
    if not tags:
        return None, None
    s = ea.Scalars(tags[0])
    return np.array([p.step for p in s], float), np.array([p.value for p in s], float)


def load_reward(name, seed=None):
    # run dir is apollonius_3d_<name> (name already encodes the seed, e.g. full_s1)
    base = [b for b in glob.glob(f"{LOG}/apollonius_3d_{name}/seed_*") if os.path.isdir(b)]
    if not base:
        return None, None
    dirs = glob.glob(f"{base[0]}/Train-Results_Episode-Rewards_rank_0_env-*")
    xs, ys = [], []
    for d in dirs:
        x, y = _scalars(d)
        if x is not None:
            xs.append(x); ys.append(y)
    if not xs:
        return None, None
    X = np.concatenate(xs); Y = np.concatenate(ys)
    o = np.argsort(X)
    return X[o], Y[o]


def binmean(X, Y, bins=160):
    edges = np.linspace(X.min(), X.max(), bins + 1)
    idx = np.clip(np.digitize(X, edges) - 1, 0, bins - 1)
    xm, ym = [], []
    for b in range(bins):
        m = idx == b
        if m.any():
            xm.append(X[m].mean()); ym.append(Y[m].mean())
    return np.array(xm), np.array(ym)


def multiseed_band(ax, runnames, label, color, bins=160):
    grid = np.linspace(0, 5e6, bins)
    ser = []
    for rn in runnames:
        X, Y = load_reward(rn)
        if X is None:
            continue
        x, y = binmean(X, Y, bins)
        ser.append(np.interp(grid, x, y))
    if not ser:
        return
    ser = np.array(ser)
    m = ser.mean(0)
    ax.plot(grid / 1e6, m, color=color, label=label, lw=2)
    if len(ser) > 1:
        sd = ser.std(0)
        ax.fill_between(grid / 1e6, m - sd, m + sd, color=color, alpha=0.18)


FULL = ["full_s1", "full_s2", "full_s3"]
NOCLOSE = ["noclose_s1", "noclose_s2", "noclose_s3"]

# ---- A: full method 3-seed convergence ----
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
multiseed_band(ax, FULL, "full method (3 seeds)", "#4C72B0")
ax.set_xlabel("training steps (millions)"); ax.set_ylabel("episode reward (train)")
ax.set_title("Reward convergence — validated method @medium")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig_reward_full.png"); print("saved fig_reward_full.png")

# ---- B: closure vs no-closure ----
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
multiseed_band(ax, FULL, "with closure (full)", "#55A868")
multiseed_band(ax, NOCLOSE, "no closure (ablation)", "#C44E52")
ax.set_xlabel("training steps (millions)"); ax.set_ylabel("episode reward (train)")
ax.set_title("Reward convergence — closure design ablation")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig_reward_closure.png"); print("saved fig_reward_closure.png")

# ---- C: N-sweep convergence (N=4 divergence) ----
fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=130)
NCOL = {"3": "#8172B3", "4": "#C44E52", "5": "#DD8452", "8": "#937860"}
for N, c in NCOL.items():
    multiseed_band(ax, [f"nsweep_N{N}_s1", f"nsweep_N{N}_s2"], f"N={N}", c)
multiseed_band(ax, FULL, "N=6 (anchor)", "#4C72B0")
ax.set_xlabel("training steps (millions)"); ax.set_ylabel("episode reward (train)")
ax.set_title("Reward convergence vs team size N  (N=4 diverges)")
ax.grid(alpha=0.3); ax.legend(ncol=2, fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_reward_Nsweep.png"); print("saved fig_reward_Nsweep.png")

# ---- D: full-method per-term sub-reward evolution (seed 1, summed over agents) ----
fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=130)
TERMS = {"r_pos": "#4C72B0", "r_gap": "#DD8452", "r_finish": "#55A868",
         "r_near": "#C44E52", "r_safe": "#8172B3"}
base = glob.glob(f"{LOG}/apollonius_3d_full_s1/seed_*")
if base:
    for term, c in TERMS.items():
        xs, ys = [], []
        for a in AGENTS:
            for d in glob.glob(f"{base[0]}/Train-SubRewards-{a}_{term}_rank_0_env-*"):
                x, y = _scalars(d)
                if x is not None:
                    xs.append(x); ys.append(y)
        if not xs:
            continue
        X = np.concatenate(xs); Y = np.concatenate(ys)
        o = np.argsort(X); xm, ym = binmean(X[o], Y[o], 160)
        ax.plot(xm / 1e6, ym, color=c, label=term, lw=1.8)
    ax.set_xlabel("training steps (millions)"); ax.set_ylabel("mean per-term sub-reward")
    ax.set_title("Per-term reward evolution — full method (seed 1)")
    ax.grid(alpha=0.3); ax.legend(ncol=3, fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_subrewards_full.png"); print("saved fig_subrewards_full.png")
print("DONE")
