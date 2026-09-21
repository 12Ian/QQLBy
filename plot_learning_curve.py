"""Plot the benchmark learning_curve.csv (step, avg_return) to a PNG."""
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else \
    "results/maddpg/apollonius_3d/learning_curve.csv"
out_path = sys.argv[2] if len(sys.argv) > 2 else \
    "results/maddpg/apollonius_3d/learning_curve.png"

steps, rets = [], []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        steps.append(int(float(row["step"])))
        rets.append(float(row["avg_return"]))

plt.figure(figsize=(8, 5), dpi=130)
plt.plot(steps, rets, marker="o", ms=3, color="#1f77b4", linewidth=1.6)
plt.xlabel("Environment steps")
plt.ylabel("Avg test return (curriculum level 4)")
plt.title("3D Multi-UAV Cooperative Pursuit (MADDPG) — Learning Curve")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_path, bbox_inches="tight")
print(f"saved {out_path}  points={len(steps)}  last_return={rets[-1]:.1f}" if rets
      else "no data")
