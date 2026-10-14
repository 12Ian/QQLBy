"""Combine audited run history with frozen final-model evaluations at every course."""
import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.analyze_level5_baseline import analyze


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def link(path, label):
    return f"[{label}](<{path.resolve().as_posix()}>)"


def course_steps(root, data):
    result = root / f"outputs/results/{data['method']}/baseline_{data['name']}"
    config = read_json(result / "config.json")
    completed = defaultdict(lambda: dict(episodes=0, transitions=0, safe_successes=0, collisions=0))
    with (result / "training_episodes.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            item = completed[int(row["level"])]
            item["episodes"] += 1
            item["transitions"] += int(row["episode_steps"])
            item["safe_successes"] += int(row["safe_success"])
            item["collisions"] += int(row["collision"])
    summary = data["summary"]
    total = int(summary["environment_steps"])
    unfinished = total - sum(item["transitions"] for item in completed.values())
    bound = int(config["parallels"]) * int(config["episode_length"])
    assert 0 <= unfinished <= bound
    final_level = int(summary["curriculum_level"])
    last_transition = summary["transitions"][-1]["step"] if summary["transitions"] else 0
    # After this many transitions, every worker has finished any older course
    # episode. The residual belongs to unfinished episodes in the final course.
    assert total - int(last_transition) > bound
    completed[final_level]["unfinished_transitions"] = unfinished
    for item in completed.values():
        item.setdefault("unfinished_transitions", 0)
        item["total_transitions"] = item["transitions"] + item["unfinished_transitions"]
    assert sum(item["total_transitions"] for item in completed.values()) == total
    return dict(levels=dict(completed), total_transitions=total,
                unfinished_transitions=unfinished, reset_bound_transitions=bound,
                transitions=summary["transitions"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--review", type=Path, default=Path("outputs/reviews/exp007_all_levels_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    review = args.review if args.review.is_absolute() else root / args.review
    state = read_json(review / "process.json")
    assert state["status"] == "completed" and len(state["jobs"]) == 10
    assert all(job["exit_code"] == 0 for job in state["jobs"])
    provenance = read_json(review / "provenance.json")
    final_validation = read_json(root / "outputs/acceptance/exp007_final_validation_v1/audit.json")
    final_acceptance = read_json(root / "outputs/acceptance/exp007_final_acceptance_v1/audit.json")
    for audit in (final_validation, final_acceptance):
        assert audit["checkpoint_sha256"] == provenance["models"]["final"]["sha256"]
        assert audit["process_completed"] and audit["full_task_verified"]
        assert audit["episodes"] == 200 and audit["full_episode_length"] == 400
        assert not audit["exploration_noise"] and not audit["collision_termination"]
    assert not final_validation["fresh_seeds"] and final_acceptance["fresh_seeds"]
    assert [item["seed"] for item in final_acceptance["reports"]] == [40001, 50001]
    evaluations = []
    config = read_json(root / "outputs/results/matd3/baseline_exp007/config.json")
    for role in ("final", "best"):
        expected_sha = provenance["models"][role]["sha256"]
        for level in range(5):
            row = read_json(review / f"{role}_level{level + 1}.json")
            assert row["checkpoint_sha256"] == expected_sha and row["level"] == level
            assert row["metrics"]["episodes"] == 100
            assert not row["exploration_noise"] and not row["collision_termination"]
            expected = config["curriculum_schedule"][level]
            for env in row["environment"]:
                assert env["curriculum_level"] == level
                assert env["building_count"] == 20 * (level + 1)
                for key in ("capture_radius", "target_min_speed", "target_max_speed"):
                    assert env[key] == expected[key]
                assert not any(env[key] for key in (
                    "terminate_on_collision", "terminate_on_boundary_collision", "terminate_on_building_collision"))
            row["model_role"] = role
            evaluations.append(row)
    runs = []
    for number in range(1, 8):
        data, _ = analyze(root, f"exp{number:03}")
        assert data["process"]["status"] == "completed" and data["process"]["exit_code"] == 0
        assert not data["live"] and data["summary"] is not None
        data["course_accounting"] = course_steps(root, data)
        runs.append(data)
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    assert branch == "feat-coverage-reaward"
    subprocess.run(["git", "diff", "--check"], cwd=root, check=True, capture_output=True)
    source_status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True)
    source_diff = subprocess.check_output(["git", "diff", "--stat"], cwd=root, text=True)
    timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    report = dict(timestamp_china=timestamp, runs=runs, current_model_evaluations=evaluations,
                  current_models=provenance, git_branch=branch, source_status=source_status,
                  source_diff_stat=source_diff,
                  final_validation=final_validation, final_acceptance=final_acceptance,
                  heldout_candidate_v1=read_json(root / "outputs/acceptance/exp007_stage5_candidate_v1/audit.json"))
    out = root / "docs/baseline_final_summary_2026-10-13.md"
    history = (root / "docs/baseline_change_history_and_videos_2026-10-13.md").read_text(encoding="utf-8")
    lines = ["# Baseline 本轮训练结束总结", "", f"核验时间（北京时间）：{timestamp}。", "",
             "本轮实验 007 已正常结束，完成 3,000,000 个新增环境转移。"
             + ("同一冻结最终模型通过两组新种子的第五级验收。" if final_acceptance["passed"]
                else "同一冻结最终模型未通过全部新种子的第五级验收。"), "",
             f"新种子 40001/50001 各 100 回合，合计安全捕获 {final_acceptance['aggregate_safe_success_rate']:.0%}、"
             f"碰撞 {final_acceptance['aggregate_collision_rate']:.0%}；原门槛为安全 ≥80%、碰撞 ≤10%。"
             "最终任务仍为 100 栋建筑、逃逸机 3–9 m/s、保持捕获半径 15 m、最多 400 步。", "",
             "最终模型的早期课程能力有所退化：第1/2级安全捕获为69%/76%；第4级碰撞15%也超过安全门槛。"
             "第五级通过验收不表示该模型在所有等级均达标。逐级复评与历史升级模型分别列出。", "",
             "## A. 每轮实际执行步数", "",
             "步数为全部并行环境合计的新增决策转移，非单个环境步数，也非梯度更新次数。", "",
             "| 实验 | 算法 | 新增环境转移 | 已完成训练回合 | 安全训练捕获 | Critic 更新 | Actor 更新 | 最终课程 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for data in runs:
        lines.append(f"| {data['name']} | {data['method']} | {data['summary']['environment_steps']:,} | "
                     f"{data['training_episodes']:,} | {data['training_safe_captures']:,} | "
                     f"{data['critic_updates']:,} | {data['actor_updates']:,} | {data['summary']['curriculum_level'] + 1} |")
    lines += ["", "以上各轮包含对照和未成功的尝试，不能把总和作为某个最终模型的累计训练步数。"
              "接续模型的来源、保存时刻和课程资格见下方修改记录及模型来源文件。", "",
              "## B. 每级实际训练步数", "",
              "完整回合的步数由原始 `episode_steps` 按课程累加。全局预算减去已完成回合转移的差值，"
              "是结束时未完成回合的步数；全部进程在最后一次升级后均已超过一个最大回合长度，"
              "因此差值可归入最终课程。此处已将差值单列并核对各级之和等于每轮总步数。", "",
              "升级按重置生效，部分环境会继续完成旧级别回合，因此准确步数与升级请求区间会略有差异。", "",
              "| 实验 | 课程 | 完整回合转移 | 未完成回合转移 | 实际合计 | 完成回合数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for data in runs:
        for level, item in sorted(data["course_accounting"]["levels"].items()):
            lines.append(f"| {data['name']} | {level + 1} | {item['transitions']:,} | "
                         f"{item['unfinished_transitions']:,} | {item['total_transitions']:,} | {item['episodes']:,} |")
    lines += ["", "## C. 当前冻结模型在每一级的效果", "",
              "重新评估本轮 300 万步最终模型，以及按固定第五级测试选择的最佳保存模型。"
              "每个模型每级 100 回合；比较种子 10001，无探索、最多 400 步，碰撞不提前终止。"
              "这与历史各级升级模型的成绩是两组不同结果，不混用。", "",
              "| 模型 | 课程 | 建筑数 | 捕获率 | 安全捕获率 | 任意机碰撞率 | 平均决策步数 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in evaluations:
        m = row["metrics"]
        role = "300 万步最终模型" if row["model_role"] == "final" else "最佳第五级保存模型"
        lines.append(f"| {role} | {row['level'] + 1} | {20 * (row['level'] + 1)} | "
                     f"{m['success_rate']:.0%} | {m['safe_success_rate']:.0%} | {m['collision_rate']:.0%} | "
                     f"{m['mean_episode_steps']:.2f} |")
    lines += ["", link(review / "provenance.json", "两份冻结模型的来源与哈希"), "",
              "### 最终模型的验证与全新种子验收", "",
              "模型在查看 40001/50001 结果前已冻结；这两组种子未用于训练、升级门槛或选取该模型。"
              "20001/30001 已用于此前95万步模型的失败验收，此次仅作为验证种子。", "",
              "| 用途 | 种子 | 回合数 | 捕获率 | 安全捕获率 | 任意机碰撞率 | 平均决策步数 | 达到门槛 |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for audit, label in ((final_validation, "已使用的验证种子"), (final_acceptance, "全新验收种子")):
        for item in audit["reports"]:
            m = item["metrics"]
            lines.append(f"| {label} | {item['seed']} | {m['episodes']} | {m['success_rate']:.0%} | "
                         f"{m['safe_success_rate']:.0%} | {m['collision_rate']:.0%} | "
                         f"{m['mean_episode_steps']:.2f} | {'是' if item['passed'] else '否'} |")
    lines += ["", f"最终模型 SHA256：`{final_acceptance['checkpoint_sha256']}`。", "",
              link(root / "outputs/acceptance/exp007_final_acceptance_v1/audit.json", "全新种子完整验收核验"), "",
              "## D. 代码、参数、奖励、课程和视频修改记录", "",
              history.partition("\n")[2], "",
              "### 当前奖励函数（实验007）", "",
              "对追捕机 i，实际奖励为下列八项之和，所有项统一乘0.1。距离 d 为到逃逸机的三维距离，"
              "Δt 为本次动作实际执行时长（通常1秒，提前捕获时更短）。", "",
              "```text",
              "r_i = 0.1 * [ -d_i/1000 * Δt + (0.99*Φ_i(next)-Φ_i(now))",
              "              -5*max(0,1-c_building_eff/50)^2 * Δt",
              "              -5*max(0,1-c_boundary_eff/50)^2 * Δt",
              "              -max(0,1-c_teammate/10)^2 * Δt",
              "              + collision_penalty_i + 300*I(capture_without_terminal_failure)",
              "              -300*I(training_collision_failure) ]",
              "Φ_i = -d_i/10",
              "c_eff = clearance - (1.5*v_closing + v_closing^2/(2*3))",
              "```", "",
              "制动公式中的接近速度取非负部分。007保留负有效间隙，因此制动不足越严重、惩罚越大。"
              "队友间隙是两机距离减去碰撞距离后截到非负值。首次建筑/边界/队友接触的原始扣分为80/100/80，"
              "持续接触另扣20/秒。真实终止时下一势函数为0；400步截断保留下一势函数并bootstrap。"
              "训练中首次任意追捕机碰撞触发全队失败；评估关闭这项提前终止，统计整个回合碰撞。", "",
              "评估中的捕获奖励不检查此前是否曾碰撞；整回合零碰撞的安全捕获率由独立指标计算。", "",
              "捕获沿用单机5米直接拦截，或同一架在课程半径内连续保持2秒；它尚未要求四架同时构成几何合围。", "",
              "### 主要源码入口", "",
              "| 内容 | 文件 |", "|---|---|",
              f"| 环境、奖励、人工势场、课程与碰撞 | {link(root / 'xuance/environment/multi_agent_env/uav_pursuit_coverage_3d.py', '环境源码')} |",
              f"| 最终实验的完整参数 | {link(root / 'experiments/baseline_exp007.yaml', '实验007配置')} |",
              f"| 完整任务冻结评估 | {link(root / 'examples/evaluate_baseline_checkpoint.py', '评估器')} |",
              f"| 视频记录与原环境重放核验 | {link(root / 'tools/render_baseline_success_videos.py', '视频工具')} |",
              f"| 训练、奖励与源快照审计 | {link(root / 'tools/analyze_level5_baseline.py', '日志审计')} |",
              "",
              "## E. 核验证据与文件改动", "",
              "七轮源码快照哈希、完整训练回合奖励分项之和、更新日志有限性、实际结束状态和逐级步数已核验。"
              "日志有限不代表 loss 收敛；固定城市上的不同初始条件测试不代表对新城市或多个训练种子的结论。", "",
              "### 当前 Git 文件状态", "", "```text", source_status.rstrip(), "```", "",
              "### 已跟踪文件改动统计", "", "```text", source_diff.rstrip(), "```", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(report=str(out), runs=len(runs), current_evaluations=len(evaluations),
                          exp007_transitions=runs[-1]["summary"]["environment_steps"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
