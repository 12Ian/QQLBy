"""Plot eval JSONs as bar charts (with 95% CI error bars) or success-vs-N lines.
Usage:
  python plot_ablation.py bars out.png metric label1=a.json label2=b.json ...
  python plot_ablation.py nvsn out.png a.json b.json ...   (uses num_agents field)
metric in {success_rate, false_capture_rate_euclidean, false_capture_rate_visibility,
           collision_rate}."""
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CI_KEY = {"success_rate": "success_ci", "collision_rate": "collision_ci",
          "false_capture_rate_euclidean": "fc_euc_ci",
          "false_capture_rate_visibility": "fc_vis_ci"}


def _load(path):
    with open(path) as f:
        return json.load(f)


def bars(out, metric, items):
    labels, vals, los, his = [], [], [], []
    for label, path in items:
        d = _load(path)
        v = d.get(metric)
        v = 0.0 if v is None else v
        ci = d.get(CI_KEY.get(metric, ""), (v, v)) or (v, v)
        lo = ci[0] if ci[0] is not None else v
        hi = ci[1] if ci[1] is not None else v
        labels.append(label); vals.append(v)
        los.append(max(0.0, v - lo)); his.append(max(0.0, hi - v))
    plt.figure(figsize=(7, 4.5), dpi=130)
    plt.bar(range(len(vals)), vals, yerr=[los, his], capsize=5, color="#4C72B0")
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel(metric); plt.title(metric + " (95% CI)")
    plt.tight_layout(); plt.savefig(out); print("saved", out)


def nvsn(out, paths):
    pts = sorted((_load(p)["num_agents"], _load(p)["success_rate"]) for p in paths)
    xs = [n for n, _ in pts]; ys = [s for _, s in pts]
    plt.figure(figsize=(7, 4.5), dpi=130)
    plt.plot(xs, ys, marker="o", color="#C44E52")
    plt.xlabel("number of pursuers N"); plt.ylabel("success rate")
    plt.title("Success rate vs N"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out); print("saved", out)


if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    if mode == "bars":
        metric = sys.argv[3]
        items = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in sys.argv[4:]]
        bars(out, metric, items)
    elif mode == "nvsn":
        nvsn(out, sys.argv[3:])
