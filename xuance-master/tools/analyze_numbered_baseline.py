"""Compare completed numbered UAV baseline experiments and verify their artifacts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def numeric(row, key):
    return float(row[key])


def analyze(root, number):
    launch = root / "outputs/experiments" / f"exp{number}"
    results = root / "outputs/results/maddpg" / f"baseline_exp{number}"
    state = json.loads((launch / "process.json").read_text())
    assert state["status"] == "completed" and state["exit_code"] == 0, state
    assert not Path(f"/proc/{state['child_pid']}/cmdline").exists(), state
    config = json.loads((results / "config.json").read_text())
    summary = json.loads((results / "baseline_summary.json").read_text())
    evaluation = read_csv(results / "curriculum_eval.csv")
    episodes = read_csv(results / "training_episodes.csv")
    updates = read_csv(results / "training_updates.csv")
    assert summary["environment_steps"] == int(config["running_steps"])
    assert all(int(r["episodes"]) == int(config["test_episode"]) for r in evaluation)
    expected_steps = list(range(0, int(config["running_steps"]) + 1, int(config["eval_interval"])))
    for phase in ("current", "final_task"):
        assert [int(r["step"]) for r in evaluation if r["phase"] == phase] == expected_steps
    for row in episodes:
        assert int(row["safe_success"]) == (int(row["success"]) and not int(row["collision"]))
        component_sum = sum(numeric(row, k) for k in
                            ("r_distance", "r_obstacle", "r_boundary", "r_teammate", "r_collision", "r_terminal"))
        assert math.isclose(component_sum, numeric(row, "mean_return"), abs_tol=0.02)
    for row in updates:
        assert all(math.isfinite(float(value)) for value in row.values())

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    log_base = root / "outputs/logs/maddpg" / f"baseline_exp{number}"
    event_files = [p for directory in log_base.glob("seed_*")
                   for p in directory.glob("events.out.tfevents.*")]
    tags = {key: [] for key in ("uav_0/loss_actor", "uav_0/loss_critic", "uav_0/predictQ")}
    for path in event_files:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in tags:
            if tag in accumulator.Tags()["scalars"]:
                tags[tag].extend(accumulator.Scalars(tag))
    recorded_updates = sum(int(r["window_updates"]) for r in updates)
    tensorboard = {}
    for tag, events in tags.items():
        assert len(events) == recorded_updates, (tag, len(events), recorded_updates)
        values = [event.value for event in events]
        assert all(math.isfinite(v) for v in values)
        tensorboard[tag] = dict(count=len(events), minimum=min(values), maximum=max(values),
                                last=values[-1], first_step=events[0].step, last_step=events[-1].step,
                                all_finite=True)

    models = list((root / "outputs/models/maddpg" / f"baseline_exp{number}").rglob("*.pth"))
    models += list((results / "best_model").glob("*.pth"))
    assert any(p.name == "final_train_model.pth" for p in models)
    assert any(p.name == "latest_train_model.pth" for p in models)
    assert any(p.name == "best_stage_0.pth" for p in models)
    checkpoints = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in models}
    import torch
    final_model = next(p for p in models if p.name == "final_train_model.pth")
    policy = torch.load(final_model, map_location="cpu", weights_only=False)["policy"]
    policy_tensors = [v for v in policy.values() if torch.is_tensor(v)]
    assert policy_tensors and all(torch.isfinite(v).all().item() for v in policy_tensors)
    final_current = next(r for r in reversed(evaluation) if r["phase"] == "current")
    final_task = next(r for r in reversed(evaluation) if r["phase"] == "final_task")
    recent = episodes[-100:]
    first_success = next((r for r in episodes if int(r["success"])), None)
    data = dict(number=number, critic_learning_rate=config["learning_rate_critic"],
                seed=config["seed"], evaluation_env_seed=config["evaluation_env_seed"],
                environment_steps=summary["environment_steps"], learning_updates=recorded_updates,
                elapsed_seconds=summary["elapsed_seconds"], process=state,
                training_episodes=len(episodes), training_successes=sum(int(r["success"]) for r in episodes),
                training_safe_successes=sum(int(r["safe_success"]) for r in episodes),
                first_training_success=first_success, final_current=final_current, final_task=final_task,
                recent_training={key: statistics.mean(numeric(r, key) for r in recent) for key in
                                 ("success", "safe_success", "collision", "mean_return", "r_distance",
                                  "r_obstacle", "r_boundary", "r_teammate", "r_collision", "r_terminal")},
                summary=summary, tensorboard=tensorboard, checkpoints=checkpoints,
                final_policy_tensor_count=len(policy_tensors), final_policy_all_finite=True)
    return data, config, evaluation, updates, episodes


def plot_runs(out, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for data, _, evaluation, updates, _ in runs:
        label = f"{data['number']} critic LR={data['critic_learning_rate']:g}"
        step = [int(r["step"]) for r in updates]
        for axis, key, title in zip(axes[0], ("loss_actor_mean", "loss_critic_mean", "predictQ_mean"),
                                    ("Actor loss (update-window mean)", "Critic loss (update-window mean)",
                                     "Q on replay samples (update-window mean)")):
            axis.plot(step, [numeric(r, key) for r in updates], label=label)
            axis.set_title(title)
        current = [r for r in evaluation if r["phase"] == "current"]
        final = [r for r in evaluation if r["phase"] == "final_task"]
        for axis, key, title in zip(axes[1, :2], ("safe_success_rate", "collision_rate"),
                                   ("Current stage: safe capture", "Current stage: any collision")):
            axis.plot([int(r["step"]) for r in current], [numeric(r, key) for r in current], "o-", label=label)
            axis.set_title(title)
            axis.set_ylim(-0.02, 1.02)
        axes[1, 2].plot([int(r["step"]) for r in final], [numeric(r, "safe_success_rate") for r in final],
                        "o-", label=label)
    axes[1, 0].axhline(0.8, color="gray", linestyle="--", label="Promotion threshold")
    axes[1, 1].axhline(0.1, color="gray", linestyle="--", label="Promotion threshold")
    axes[1, 2].set_title("Fixed final task (100 buildings): safe capture")
    axes[1, 2].set_ylim(-0.005, max(0.1, max(numeric(r, "safe_success_rate") for run in runs
                                             for r in run[2] if r["phase"] == "final_task") * 1.2))
    for axis in axes.flat:
        axis.set_xlabel("Environment transitions")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Numbered baseline comparison; evaluation: 100 episodes, no exploration noise", fontsize=13)
    figure.savefig(out / "numbered_baseline_comparison.png", dpi=180)
    plt.close(figure)


def write_markdown(out, runs, report):
    text = ["# Baseline 编号训练结果（2026-10-13）", "",
            "服务器分支：`feat-coverage-reaward`。训练复查修复环境种子、评估种子和并行回合数问题；"
            "行为测试 28 passed，10.33 s。", "",
            "## 对照设置", "",
            "两次实验使用相同代码、固定地图、奖励、训练种子 1、评估环境种子 10001、"
            "16 个训练环境、512×512 网络、batch_size=1024 和 300000 个环境转移。"
            "每 50000 步做 100 回合无探索噪声评估，初始评估结果完全一致。"
            "actor 学习率 0.0001；唯一参数变化为 critic 学习率 0.001 → 0.0003。", "",
            "| 编号 | critic 学习率 | 训练捕获 / 回合 | 训练无碰撞捕获 | 最终当前级捕获率 | "
            "最终当前级安全成功率 | 碰撞率 | 第 5 级安全成功率 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for data, *_ in runs:
        current, final = data["final_current"], data["final_task"]
        text.append(f"| {data['number']} | {data['critic_learning_rate']:g} | "
                    f"{data['training_successes']} / {data['training_episodes']} | "
                    f"{data['training_safe_successes']} | {numeric(current, 'success_rate'):.0%} | "
                    f"{numeric(current, 'safe_success_rate'):.0%} | {numeric(current, 'collision_rate'):.0%} | "
                    f"{numeric(final, 'safe_success_rate'):.0%} |")
    text += ["", "安全成功要求整回合所有追捕机均无建筑、边界或队友接触。"
             "训练回合含探索噪声，其统计不等同于评估成功率。", "", "## 参数选择", "",
             f"按最终当前级的安全成功率、碰撞率、捕获率和回报排序，选择实验 **{report['selected_experiment']}**，"
             f"critic 学习率 **{report['selected_critic_learning_rate']:g}**。"
             "该选择来自一次相同种子的对照，不代表跨种子的普遍结论。", "",
             "降低学习率的假设来自 001 后半程 critic loss 回升、Q 上升及安全成功率低。"
             "loss 较低不能单独证明策略更好；是否保留参数以安全成功率和碰撞率为主要依据。", "",
             "## 验证与模型", ""]
    for data, *_ in runs:
        success = data["first_training_success"]
        first = f"第 {success['step']} 步，{success['capture_type']}" if success else "无"
        stage = data["summary"]["curriculum_level"]
        best = data["summary"]["best_by_stage"].get(str(stage), {})
        text.append(f"- {data['number']}：退出码 0，{data['environment_steps']} 环境转移，"
                    f"{data['learning_updates']} 次学习更新；首次训练捕获：{first}。"
                    "全部 actor loss、critic loss、Q 的 TensorBoard 更新记录均有限，"
                    "记录条数与学习更新数一致；逐回合奖励合计与总回报一致。")
        text.append(f"- {data['number']} 当前阶段模型："
                    f"`outputs/results/maddpg/baseline_exp{data['number']}/best_model/best_stage_{stage}.pth`，"
                    f"对应 step={best.get('step')}，安全成功率={float(best.get('safe_success_rate', 0)):.0%}。")
        final_best = data["summary"]["best_final_task"]
        text.append(f"- {data['number']} 的 `best_model.pth` 是按固定第 5 级评估挑选的，"
                    f"对应 step={final_best['step']}、安全成功率={final_best['safe_success_rate']:.0%}；"
                    "其选择口径与当前阶段模型不同。")
    text += ["", "最终训练模型在 `outputs/models/maddpg/baseline_expNNN/seed_*/final_train_model.pth`。"
             "每次实验的冻结配置、源代码 ZIP、hash、进程退出记录和训练日志均已保存。", "",
             "## 当前限制", "",
             "升级需要至少 100000 步、连续两次安全成功率 ≥80% 且碰撞率 ≤10%。"
             "当前结果及课程转换见比较 JSON；未达标不能强制升级。"
             "这次对照不能证明完整五级课程已学会或 loss 已收敛。", "",
             "## 附件", "",
             "- [指标与核验 JSON](comparison.json)", "- [实验对照 CSV](comparison.csv)",
             "- [训练与评估曲线](numbered_baseline_comparison.png)", ""]
    (out / "comparison.md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbers", nargs="+", default=["001", "002"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runs = [analyze(root, number) for number in args.numbers]
    if len(runs) == 2:
        reference, candidate = runs
        delta = {k: [reference[1].get(k), candidate[1].get(k)] for k in set(reference[1]) | set(candidate[1])
                 if reference[1].get(k) != candidate[1].get(k)}
        relevant = {k: value for k, value in delta.items()
                    if k not in ("env_id", "log_dir", "model_dir", "result_dir")}
        assert relevant == {"learning_rate_critic": [0.001, 0.0003]}, relevant
        assert [r for r in reference[2] if int(r["step"]) == 0] == [r for r in candidate[2] if int(r["step"]) == 0]
        snapshots = [json.loads((root / "outputs/experiments" / f"exp{r[0]['number']}" / "metadata.json").read_text())
                     for r in runs]
        hashes = [{k: v for k, v in s["source_hashes"].items() if not k.startswith("experiments/")}
                  for s in snapshots]
        assert hashes[0] == hashes[1], "Unexpected source difference in paired comparison"
    else:
        relevant = {}
    def quality(run):
        row = run[0]["final_current"]
        return (numeric(row, "safe_success_rate"), -numeric(row, "collision_rate"),
                numeric(row, "success_rate"), numeric(row, "mean_return"))
    selected = max(runs, key=quality)[0]
    report = dict(experiments=[r[0] for r in runs], parameter_delta=relevant,
                  selection_criterion="Final current-stage safe success, then fewer collisions, success, return",
                  selected_experiment=selected["number"], selected_critic_learning_rate=selected["critic_learning_rate"],
                  limitations=["One training seed, 100 evaluation episodes per checkpoint",
                               "Finite loss/Q does not prove policy convergence",
                               "Selection is for the trained current stage, not final-task competence"])
    out = root / "docs/baseline_training_results_2026-10-13"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["number", "critic_learning_rate", "environment_steps", "learning_updates", "training_successes",
              "training_safe_successes", "success_rate", "safe_success_rate", "collision_rate", "mean_return",
              "final_task_safe_success_rate", "curriculum_level"]
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for data, *_ in runs:
            row = {k: data[k] for k in fields if k in data}
            row.update({k: data["final_current"][k] for k in
                        ("success_rate", "safe_success_rate", "collision_rate", "mean_return")})
            row.update(final_task_safe_success_rate=data["final_task"]["safe_success_rate"],
                       curriculum_level=data["summary"]["curriculum_level"])
            writer.writerow(row)
    plot_runs(out, runs)
    write_markdown(out, runs, report)
    print(json.dumps({"selected_experiment": selected["number"], "selected_critic_learning_rate":
                      selected["critic_learning_rate"], "directory": str(out), "checks": "passed"}))


if __name__ == "__main__":
    main()
