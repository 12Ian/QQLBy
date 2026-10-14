"""Report numbered pursuit experiments without equating curriculum progress with success."""
import argparse
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    return list(csv.DictReader(io.StringIO(path.read_text())))


def analyze(root, name):
    directory = root / "outputs/experiments" / name
    process = json.loads((directory / "process.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    method = metadata.get("method", "maddpg")
    results = root / "outputs/results" / method / f"baseline_{name}"
    config = json.loads((results / "config.json").read_text())
    evaluations = read_csv(results / "curriculum_eval.csv")
    episodes = read_csv(results / "training_episodes.csv")
    updates = read_csv(results / "training_updates.csv")
    assert all(int(row["episodes"]) == config["test_episode"] for row in evaluations)
    assert all(math.isfinite(float(row[key])) for row in updates
               for key in ("loss_critic_mean", "predictQ_mean", "loss_critic_max", "predictQ_min", "predictQ_max"))
    assert all(not row["loss_actor_mean"] or math.isfinite(float(row["loss_actor_mean"])) for row in updates)
    assert all(math.isclose(sum(float(value) for key, value in row.items() if key.startswith("r_")),
                            float(row["mean_return"]), rel_tol=1e-6, abs_tol=1e-5) for row in episodes)
    with zipfile.ZipFile(directory / "source_snapshot.zip") as snapshot:
        assert all(hashlib.sha256(snapshot.read(path)).hexdigest() == expected
                   for path, expected in metadata["source_hashes"].items())
    current = [row for row in evaluations if row["phase"] == "current"]
    final = [row for row in evaluations if int(row["level"]) == 4]
    key = lambda row: (float(row["safe_success_rate"]), -float(row["collision_rate"]), float(row["mean_return"]))
    best = {level: max((row for row in current if int(row["level"]) == level), key=key)
            for level in sorted({int(row["level"]) for row in current})}
    pid_path = Path(f"/proc/{process['child_pid']}/cmdline")
    live = pid_path.exists() and b"run_baseline.py" in pid_path.read_bytes()
    summary_path = results / "baseline_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    actor_updates = int(updates[-1].get("actor_updates", updates[-1]["updates"])) if updates else 0
    return dict(name=name, method=method, process=process, live=live, training_episodes=len(episodes),
                training_safe_captures=sum(int(row["safe_success"]) for row in episodes),
                highest_training_level=max((int(row["level"]) for row in episodes), default=None),
                critic_updates=int(updates[-1]["updates"]) if updates else 0,
                actor_updates=actor_updates, actor_update_delay=config.get(
                    "actor_update_delay", 2) if method == "matd3" else config.get("maddpg_actor_update_delay", 1),
                best_by_stage=best, latest_current=current[-1] if current else None,
                best_final=max(final, key=key) if final else None,
                latest_final=final[-1] if final else None, summary=summary,
                final_task_spec=config["curriculum_schedule"][-1],
                source_snapshot_verified=True, reward_sum_verified=True,
                completion_requires_heldout_validation=True), evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), experiments=[])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, name in enumerate(args.experiments):
        data, rows = analyze(root, name)
        report["experiments"].append(data)
        color = colors[index % len(colors)]
        for axis, selected in [(axes[0], [r for r in rows if r["phase"] == "current"]),
                               (axes[1], [r for r in rows if int(r["level"]) == 4])]:
            x = [int(row["step"]) / 1e6 for row in selected]
            axis.plot(x, [float(row["safe_success_rate"]) for row in selected], "o-", color=color,
                      markersize=3, label=f"{name} safe capture")
            axis.plot(x, [float(row["collision_rate"]) for row in selected], "x--", color=color,
                      markersize=3, alpha=.65, label=f"{name} collision")
        previous_level = None
        for row in [r for r in rows if r["phase"] == "current"]:
            level = int(row["level"])
            if level != previous_level:
                axes[0].annotate(f"L{level+1}", (int(row["step"])/1e6, float(row["safe_success_rate"])),
                                 xytext=(3, 7), textcoords="offset points", color=color)
                previous_level = level
    for axis, title in zip(axes, ("Current curriculum task (L1-L5)", "Original final task: 100 buildings, 9 m/s, 15 m")):
        axis.axhline(.8, color="green", linestyle=":", linewidth=1, label="safe capture threshold")
        axis.axhline(.1, color="red", linestyle=":", linewidth=1, label="collision threshold")
        axis.set(title=title, xlabel="Environment transitions (million)", ylabel="Episode fraction", ylim=(-.02, 1.05))
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    fig.savefig(args.output / "level5_progress.png", dpi=160)
    plt.close(fig)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    lines = ["# 第五级目标实验结果", "", "原始完整回合评估：每次 100 回合，无探索噪声；碰撞不提前终止评估回合。", "",
             "训练等级与单次评估不能单独证明目标完成；第五级达标后还需要新种子复核。", "",
             "| 实验 | 状态 | 最高训练级别 | 最新课程安全成功/碰撞 | 最新第五级安全成功/碰撞 | Critic/Actor 更新 |",
             "|---|---|---:|---|---|---:|"]
    rates = lambda row: f"{float(row['safe_success_rate']):.0%}/{float(row['collision_rate']):.0%}" if row else "无"
    for data in report["experiments"]:
        level = data["highest_training_level"]
        lines.append(f"| {data['name']} | {data['process']['status']} | {level+1 if level is not None else '无'} | "
                     f"{rates(data['latest_current'])} | {rates(data['latest_final'])} | "
                     f"{data['critic_updates']}/{data['actor_updates']} |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output), "experiments": [
        {k: data[k] for k in ("name", "live", "highest_training_level", "critic_updates", "actor_updates")}
        for data in report["experiments"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
