"""GAT ablation (3.3) + Ch4 reward ablation (4.4) figures. English labels (font-safe)."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, RED, GREEN, VIOLET, AMBER, GREY = "#2a78d6", "#e34948", "#008300", "#4a3aa7", "#eda100", "#b9b8b2"
INK, MUTED, GRID, SURF = "#0b0b0b", "#6b6f76", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7"})

def sr(name, d="medium"):
    try: return json.load(open(f"p3_results/eval_{name}_{d}.json"))["success_rate"]*100
    except: return None
def srmt(name):
    try: return json.load(open(f"p3_results/eval_{name}.json"))["success_rate"]*100
    except: return None
def mean(xs): xs=[x for x in xs if x is not None]; return np.mean(xs) if xs else None
def style(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)

# ---- Fig: GAT ablation ----
fig, ax = plt.subplots(figsize=(7,4.5),dpi=150)
noGAT = mean([sr("s_N3_s1"), sr("s_N3_s2"), sr("s_N3_s3")])
gatT  = [sr("s3_gat_t_s1"), sr("s3_gat_t_s2")]
gatF  = [sr("s3_gat_full_s1"), sr("s3_gat_full_s2")]
labels=["No GAT\n(flagship)","GAT-T only","Full GAT"]
vals=[noGAT, mean(gatT), mean(gatF)]; cols=[RED, BLUE, VIOLET]
ax.bar(labels, vals, color=cols, width=0.58)
for i,v in enumerate(vals): ax.text(i, v+2, f"{v:.0f}%", ha="center", fontweight="600", color=INK)
for pts,xi in [(gatT,1),(gatF,2)]:
    for p in pts:
        if p is not None: ax.plot(xi, p, "o", ms=4, mfc="white", mec=MUTED, mew=1, zorder=5)
ax.set_ylim(0,100); ax.set_ylabel("Capture success (%)", color=INK)
ax.set_title("GAT ablation (N=3) — GAT does not help a small team", loc="left", fontsize=12, color=INK, pad=8); style(ax)
fig.tight_layout(); fig.savefig("p3_results/curves/fig_gat.png", bbox_inches="tight", facecolor=SURF); plt.close(fig)

# ---- Fig: Ch4 reward ablation ----
fig, ax = plt.subplots(figsize=(7.5,4.6),dpi=150)
BASE=77.0
terms=[("r_gap",srmt("ch4abl_r_gap_s1")),("r_pos",srmt("ch4abl_r_pos_s1")),
       ("r_safe",srmt("ch4abl_r_safe_s1")),("r_near",srmt("ch4abl_r_near_s1")),
       ("r_finish",srmt("ch4abl_r_finish_s1")),("r_closure",srmt("ch4abl_r_closure_s1"))]
terms=[(t,v) for t,v in terms if v is not None]
terms.sort(key=lambda x:x[1])
labs=[t[0] for t in terms]; vs=[t[1] for t in terms]; y=np.arange(len(labs))
ax.axvline(BASE, color=GREEN, lw=1.6, ls="--"); ax.text(BASE+0.8, len(labs)-0.35, "full reward 77%", color=GREEN, fontsize=9)
ax.barh(y, vs, color=[RED if v<60 else BLUE for v in vs], height=0.6)
for yi,v in zip(y,vs): ax.text(v+1, yi, f"{v:.0f}%", va="center", fontsize=9.5, fontweight="600", color=INK)
ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=10)
ax.set_xlim(0,100); ax.set_xlabel("All-captured success with term removed (%)", color=INK)
ax.set_title("Ch4 reward ablation (M=2) — distribution reward (r_gap) matters most", loc="left", fontsize=11.5, color=INK, pad=8)
style(ax, "x")
fig.tight_layout(); fig.savefig("p3_results/curves/fig_ch4_ablation.png", bbox_inches="tight", facecolor=SURF); plt.close(fig)

print("saved fig_gat, fig_ch4_ablation")
print("GAT noGAT/T/full:", round(noGAT), [None if v is None else round(v) for v in gatT], [None if v is None else round(v) for v in gatF])
print("ch4 abl sorted:", [(t,round(v)) for t,v in terms])
