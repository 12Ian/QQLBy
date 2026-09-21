"""Ch4 + baseline + updated N=3 speed figures (English labels, server-font-safe).
Reads p3_results/eval_*.json. Writes fig_ch4_alloc/ch4_density/baseline/speed_n3."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, RED, GREEN, VIOLET, AMBER, GREY = "#2a78d6", "#e34948", "#008300", "#4a3aa7", "#eda100", "#b9b8b2"
INK, MUTED, GRID, SURF = "#0b0b0b", "#6b6f76", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7"})

def ev(name, d=None):
    f = f"p3_results/eval_{name}_{d}.json" if d else f"p3_results/eval_{name}.json"
    try: return json.load(open(f))
    except: return None
def sr(name, d=None):
    r = ev(name, d); return r["success_rate"]*100 if r else None
def mean(xs): xs=[x for x in xs if x is not None]; return np.mean(xs) if xs else None
def style(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)

# ---------- Fig A: Ch4 allocation (dynamic vs static) + scalability ----------
fig, (a1,a2) = plt.subplots(1,2,figsize=(11,4.5),dpi=150)
dyn = mean([sr("ch4_M2_s1"), sr("ch4_M2_s2")]); stat = sr("ch4_M2_static_s1")
a1.bar(["Dynamic","Static"], [dyn, stat], color=[GREEN, GREY], width=0.55)
for i,v in enumerate([dyn,stat]): a1.text(i, v+2, f"{v:.0f}%", ha="center", fontweight="600", color=INK)
a1.set_ylim(0,100); a1.set_ylabel("All-captured success (%)", color=INK)
a1.set_title("(a) Allocation ablation (M=2) — dynamic ~2x static", loc="left", fontsize=11, color=INK, pad=8); style(a1)
Ms=["M=2 (N=6)","M=3 (N=9)"]; mv=[dyn, sr("ch4_M3_s1")]
a2.bar(Ms, mv, color=[BLUE, VIOLET], width=0.5)
for i,v in enumerate(mv): a2.text(i, v+2, f"{v:.0f}%", ha="center", fontweight="600", color=INK)
a2.set_ylim(0,100); a2.set_ylabel("All-captured success (%)", color=INK)
a2.set_title("(b) Scalability — multi-target pursuit", loc="left", fontsize=11, color=INK, pad=8); style(a2)
fig.suptitle("Ch4 Multi-target Pursuit (strong evader) — Dynamic Allocation & Scalability", fontsize=12.5, color=INK, y=1.02)
fig.tight_layout(); fig.savefig("p3_results/curves/fig_ch4_alloc.png", bbox_inches="tight", facecolor=SURF); plt.close(fig)

# ---------- Fig B: Ch4 density generalization (M=2) ----------
fig, ax = plt.subplots(figsize=(7,4.5),dpi=150)
dens=["empty","open","medium","complex"]
def sr_ch4(name, d): return sr(name) if d=="medium" else sr(name, d)  # medium eval has no suffix
dv=[mean([sr_ch4("ch4_M2_s1",d), sr_ch4("ch4_M2_s2",d)]) for d in dens]
ax.bar(dens, [v if v is not None else 0 for v in dv], color=[BLUE,BLUE,GREEN,GREY], width=0.6)
for i,v in enumerate(dv):
    if v is not None: ax.text(i, v+2, f"{v:.0f}%", ha="center", fontweight="600", color=INK)
ax.set_ylim(0,100); ax.set_ylabel("All-captured success (%)", color=INK); ax.set_xlabel("Obstacle density (trained @medium)", color=INK)
ax.set_title("Ch4 density generalization (M=2, N=6)", loc="left", fontsize=12, color=INK, pad=8); style(ax)
fig.tight_layout(); fig.savefig("p3_results/curves/fig_ch4_density.png", bbox_inches="tight", facecolor=SURF); plt.close(fig)

# ---------- Fig C: baseline comparison ----------
fig, ax = plt.subplots(figsize=(8,4.5),dpi=150)
maddpg = mean([sr("s_N3_s1","medium"), sr("s_N3_s2","medium"), sr("s_N3_s3","medium")])
rows = [("Random", sr("base_random","medium"), GREY),
        ("MASAC", None, GREY),
        ("MAPPO", sr("base_mappo_s2","medium"), VIOLET),
        ("Greedy", sr("base_greedy","medium"), AMBER),
        ("MADDPG (ours)", maddpg, RED)]
labels=[r[0] for r in rows]; vals=[r[1] if r[1] is not None else 0 for r in rows]; cols=[r[2] for r in rows]
ax.bar(labels, vals, color=cols, width=0.62)
for i,(lab,v,c) in enumerate(rows):
    if v is None: ax.text(i, 3, "OOM\n(infeasible)", ha="center", fontsize=9, color=MUTED, style="italic")
    else: ax.text(i, v+2, f"{v:.0f}%", ha="center", fontweight="600", color=INK)
ax.set_ylim(0,100); ax.set_ylabel("Capture success (%)", color=INK)
ax.set_title("Baseline comparison (N=3, strong evader @medium) — MADDPG leads", loc="left", fontsize=12, color=INK, pad=8); style(ax)
fig.tight_layout(); fig.savefig("p3_results/curves/fig_baseline.png", bbox_inches="tight", facecolor=SURF); plt.close(fig)

# ---------- Fig D: updated N=3 speed curve ----------
fig, ax = plt.subplots(figsize=(7,4.5),dpi=150)
V=[8,9,10,11,12]
sp=[sr("s3_lam_V8","medium"), sr("s3_lam_V9","medium"), maddpg, sr("s3_lam_V11","medium"), sr("s3_lam_V12","medium")]
sp=[float("nan") if s is None else s for s in sp]
ax.axvspan(11,12.4,color=RED,alpha=0.06); ax.axvline(11,color=MUTED,ls="--",lw=1)
ax.text(11.05,92,"lambda=1 (parity)",fontsize=8.5,color=MUTED)
ax.plot(V, sp, "-o", color=BLUE, lw=2.4, ms=6.5, mfc=BLUE, mec="white", mew=1)
for v,s in zip(V,sp):
    if s==s: ax.text(v, s+4, f"{s:.0f}%", ha="center", fontsize=9, color=INK)
ax.set_xlim(7.6,12.4); ax.set_ylim(-3,108); ax.set_xlabel("Evader max speed (pursuer=11)", color=INK)
ax.set_ylabel("Capture success (%)", color=INK)
ax.set_title("Speed capability boundary (N=3 flagship) — 27-40% even past parity", loc="left", fontsize=12, color=INK, pad=8)
ax.text(11.7,58,"faster evader\n(plan lambda<1)",ha="center",fontsize=8.5,color=RED,style="italic")
style(ax); ax.grid(True, axis="both", color=GRID, lw=0.7)
fig.tight_layout(); fig.savefig("p3_results/curves/fig_speed_n3.png", bbox_inches="tight", facecolor=SURF); plt.close(fig)

print("saved ch4_alloc/ch4_density/baseline/speed_n3")
print("ch4 dyn/stat/M3:", dyn, stat, sr("ch4_M3_s1"))
print("ch4 density:", [None if v is None else round(v) for v in dv])
print("baseline maddpg/greedy/mappo/random:", round(maddpg), sr("base_greedy","medium"), sr("base_mappo_s2","medium"), sr("base_random","medium"))
print("speed n3:", [None if s is None else round(s) for s in sp])
