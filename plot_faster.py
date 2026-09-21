"""Correct-regime (evader FASTER, vmax12>11) figures: one-sided vs surround-spawn N-sweep,
+ failure-mode breakdown. English labels (server-font-safe)."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, RED, GREEN, VIOLET, AMBER, GREY = "#2a78d6", "#e34948", "#008300", "#4a3aa7", "#eda100", "#b9b8b2"
INK, MUTED, GRID, SURF = "#0b0b0b", "#6b6f76", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7"})

def val(name, key="success_rate"):
    try: return json.load(open(f"p3_results/eval_{name}_medium.json"))[key]*100
    except: return None
def m(names, key="success_rate"):
    v=[val(n,key) for n in names]; v=[x for x in v if x is not None]; return np.mean(v) if v else None
def style(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)

Ns=[3,4,5,6,8]
onesided=[m([f"f_N{N}_s1",f"f_N{N}_s2"]) for N in Ns]  # N=4 one-sided not run (was death-spiral)
surround=[m([f"fs_N{N}_s1",f"fs_N{N}_s2"]) for N in Ns]

# ---- Fig 1: surround N-sweep incl rescued N=4 ----
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.6),dpi=150)
x=np.arange(len(Ns))
cols=[RED if n==4 else BLUE for n in Ns]
a1.bar(x,[v or 0 for v in surround],0.6,color=cols)
for i,(n,v) in enumerate(zip(Ns,surround)):
    if v is not None: a1.text(i,v+1.5,f"{v:.0f}%",ha="center",fontsize=10,color=INK,fontweight="600")
a1.set_xticks(x); a1.set_xticklabels([f"N={n}" for n in Ns])
a1.set_ylim(0,52); a1.set_ylabel("Capture success (%)",color=INK)
a1.set_title("(a) FASTER evader + surround — N=4 RESCUED by safe-spawn",loc="left",fontsize=11,color=INK,pad=8)
a1.annotate("N=4: safe-spawn rescue\n0% (death-spiral) → 33%",xy=(1,33),xytext=(1.5,46),
            fontsize=8.5,color=RED,ha="left",
            arrowprops=dict(arrowstyle="->",color=RED,lw=1.2))
style(a1)

# ---- Fig 2: failure-mode breakdown (surround, per N) ----
S=[m([f"fs_N{N}_s1",f"fs_N{N}_s2"],"success_rate") for N in Ns]
C=[m([f"fs_N{N}_s1",f"fs_N{N}_s2"],"collision_rate") for N in Ns]
T=[m([f"fs_N{N}_s1",f"fs_N{N}_s2"],"timeout_rate") for N in Ns]
xs=[f"N={n}" for n in Ns]
a2.bar(xs,S,color=GREEN,label="Captured")
a2.bar(xs,C,bottom=S,color=RED,label="Collision")
a2.bar(xs,T,bottom=np.array(S)+np.array(C),color=AMBER,label="Evaded (escaped)")
a2.set_ylim(0,105); a2.set_ylabel("Episode outcome (%)",color=INK)
a2.set_title("(b) Surround failure mode — small teams best, big teams collide",loc="left",fontsize=11,color=INK,pad=8)
a2.legend(frameon=False,fontsize=8.5,loc="upper right",labelcolor=INK); style(a2)
fig.suptitle("Correct regime: catching a FASTER evader via cooperative encirclement",fontsize=12.5,color=INK,y=1.02)
fig.tight_layout(); fig.savefig("p3_results/curves/fig_faster.png",bbox_inches="tight",facecolor=SURF); plt.close(fig)
print("saved fig_faster")
print("onesided:",[None if v is None else round(v) for v in onesided])
print("surround:",[None if v is None else round(v) for v in surround])
