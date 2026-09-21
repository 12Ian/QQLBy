"""Figure 7 (plan): visibility-constrained Apollonius escape-solid-angle schematic.
Two cases: (a) obstacle occludes a pursuer -> its blockade is voided (Euclidean over-counts);
(b) obstacle blocks the evader -> direction physically sealed (both criteria agree)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

BLUE, RED, GREEN, GREY, INK, MUTED, SURF = "#2a78d6", "#e34948", "#008300", "#9c9a94", "#0b0b0b", "#5c6066", "#fcfcfb"

def arrow(ax, p0, p1, color, lw=2.2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16,
                 color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0, zorder=6))

fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 5), dpi=150)
for ax in (a, b):
    ax.set_xlim(-1, 6); ax.set_ylim(-2.6, 2.8); ax.set_aspect("equal"); ax.axis("off")

# ---- (a) obstacle occludes a pursuer -> blockade voided ----
a.set_title("(a) Obstacle occludes a pursuer  ->  its blockade is VOID",
            fontsize=11.5, color=INK, loc="left", pad=6)
a.add_patch(Circle((0, 0), 0.22, color=BLUE, zorder=5))
a.text(0, -0.55, "Evader", ha="center", fontsize=10, color=BLUE, fontweight="600")
a.add_patch(Circle((5, 0), 0.22, color=RED, zorder=5))
a.text(5, -0.55, "Pursuer", ha="center", fontsize=10, color=RED, fontweight="600")
# obstacle between them
a.add_patch(Rectangle((2.3, -0.7), 0.8, 1.4, color=GREY, zorder=4))
a.text(2.7, 1.0, "obstacle", ha="center", fontsize=9, color=MUTED)
# occluded line of sight
a.plot([0.22, 5-0.22], [0, 0], ls=(0,(4,3)), color=MUTED, lw=1.4, zorder=3)
a.text(2.7, -1.15, "line-of-sight occluded", ha="center", fontsize=8.5, color=MUTED, style="italic")
a.text(2.7, -0.02, "X", ha="center", va="center", fontsize=15, color=RED, fontweight="700", zorder=7)
# escape arrow (evader escapes along d, upward-right, around)
arrow(a, (0.15, 0.15), (1.7, 1.9), GREEN, lw=2.6)
a.text(1.8, 2.15, "evader escapes", ha="left", fontsize=9.5, color=GREEN, fontweight="600")
# verdicts
a.text(3.0, -2.25, "Euclidean: BLOCKED (false)   -   Visibility: OPEN (correct)",
       ha="center", fontsize=9.5, color=INK)

# ---- (b) obstacle blocks the evader -> direction sealed ----
b.set_title("(b) Obstacle blocks the evader  ->  direction SEALED",
            fontsize=11.5, color=INK, loc="left", pad=6)
b.add_patch(Circle((0, 0), 0.22, color=BLUE, zorder=5))
b.text(0, -0.55, "Evader", ha="center", fontsize=10, color=BLUE, fontweight="600")
# obstacle directly on the escape direction (to the right)
b.add_patch(Rectangle((2.6, -0.9), 1.0, 1.8, color=GREY, zorder=4))
b.text(3.1, 1.2, "obstacle", ha="center", fontsize=9, color=MUTED)
# escape direction blocked by obstacle
arrow(b, (0.22, 0), (2.5, 0), RED, lw=2.4, style="-|>")
b.text(1.3, 0.32, "escape dir", ha="center", fontsize=9, color=MUTED)
b.text(3.1, -0.02, "|", ha="center", va="center", fontsize=1, color=RED)
b.plot([2.6, 2.6], [-0.9, 0.9], color=RED, lw=3, zorder=7)
b.text(3.0, -2.25, "Euclidean = Visibility: BLOCKED (both agree)",
       ha="center", fontsize=9.5, color=INK)

fig.suptitle("Visibility-constrained Apollonius escape criterion (obstacle occlusion, both directions)",
             fontsize=12.5, color=INK, y=0.99)
fig.tight_layout()
fig.savefig("p3_results/curves/fig_criterion_schematic.png", bbox_inches="tight", facecolor=SURF)
print("saved fig_criterion_schematic")
