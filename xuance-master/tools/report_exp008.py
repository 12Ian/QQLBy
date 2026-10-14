"""Summarize actual exp008 progress without modifying training or historical experiments."""
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path):
    return list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []


def main():
    directory = ROOT / "outputs/experiments/exp008"
    result = ROOT / "outputs/results/matd3/baseline_exp008"
    metadata = read_json(directory / "metadata.json")
    process = read_json(directory / "process.json")
    episodes = read_csv(result / "training_episodes.csv")
    updates = read_csv(result / "training_updates.csv")
    evaluations = read_csv(result / "curriculum_eval.csv")
    summary = read_json(result / "baseline_summary.json")
    status = read_json(result / "training_status.json")
    per_level = []
    for level in range(5):
        selected = [row for row in episodes if int(row["level"]) == level]
        tested = [row for row in evaluations if int(row["level"]) == level and row["phase"] == "current"]
        latest = tested[-1] if tested else None
        per_level.append(dict(level=level + 1, completed_episodes=len(selected),
                              completed_episode_transitions=sum(int(row["episode_steps"]) for row in selected),
                              training_successes=sum(int(row["success"]) for row in selected),
                              training_safe_successes=sum(int(row["safe_success"]) for row in selected),
                              latest_evaluation=latest))
    observed_step = max([0, int(status.get("step", 0))] + [int(row["step"]) for row in evaluations + updates])
    report = dict(timestamp_china=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat(),
                  process=process, metadata=metadata, observed_step=observed_step, summary=summary,
                  levels=per_level, latest_update=updates[-1] if updates else {}, evaluations=evaluations,
                  logs=[str(path) for path in (ROOT / "outputs/logs/matd3/baseline_exp008").glob("seed_*")],
                  holdout_evaluations=[read_json(path) for path in sorted((ROOT / "outputs/reviews/exp008_final_v1").glob("*_level*.json"))])
    output = ROOT / "docs/baseline_exp008_progress_2026-10-13"
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = ["# 实验008：单侧远距离出生与详细日志", "",
            f"更新时间：{report['timestamp_china']}", "",
            f"状态：{process.get('status', '未启动')}；已观测转移数：{observed_step:,}；预算：3,000,000。", "",
            "## 修改与实验口径", "",
            "- 在 feat-coverage-reaward 分支修改。沿用007最终模型初始化，回放池重新建立，从第1级训练；不是从随机权重开始。",
            "- r_distance、r_teammate 权重为0；机间接触不计惩罚、终止或碰撞率。仍记录零值列用于比较。",
            "- 保留 r_progress、建筑/边界避障、真实碰撞、捕获奖励、全队碰撞失败惩罚；所有奖励仍乘0.1。",
            "- 追捕机加速度3 m/s²、逃逸机1 m/s²；jerk分别6和8 m/s³；加速度滞后0.5 s；决策1 s、物理0.1 s。课程会覆盖逃逸机加速度，5级均为1。",
            "- 逃逸机感知半径仍为60 m。按当前低加速度，建议下一次单独比较150/200 m；本轮没有同时改变感知或加速度。",
            "- r_progress=0.1[0.99Φ(s_next)−Φ(s)]，Φ=−d/10；真正终止时下一状态势值为0，400步超时保留下一势值。",
            "- r_failure 从实验003加入，训练因建筑/边界碰撞终止时全队每架扣30分一次；400步超时不触发。碰撞机另有 r_collision。",
            "- 四架追捕机期望出生：(100,350,180)、(100,450,200)、(100,550,220)、(100,650,200) m；建筑安全修正后记录实际位置。",
            "- 逃逸机在 x=650–750、y=350–650、z=150–250 m 的自由空间随机出生；实际每架初距至少450 m。初速1 m/s的追捕机朝目标出发。固定城市不变。",
            "- 每级至少100,000转移；每50,000步无探索评估100回合，连续两次安全捕获≥80%、碰撞≤10%才升级。最长400步，固定评估流为10001。",
            "- 安全捕获与碰撞率统计全回合；评估关闭碰撞提前终止，机间碰撞本轮不计。007的94%不能代替新布局成绩。", "",
            "## TensorBoard", "",
            "旧启动脚本在 seed 根目录中再次选择最新子目录，实际只加载006的 uav_3/r_failure/env-8；已修复这一选择逻辑。", "",
            "新日志全部写入同一 seed 根事件文件：", "",
            "| 标签 | 含义 |", "|---|---|",
            "| Train-Rewards/uav_0–3/* | 各机每回合奖励分项，含禁用项的零值 |",
            "| Train-Rewards/team/* | 四机奖励分项平均 |",
            "| Train-Rewards-Per-Step/* | 回合奖励分项除以决策步数 |",
            "| Train-Rewards-Rolling100/team/* | 当前级别最近100回合分项均值 |",
            "| Train-Returns/* | 各机总回报及全队均值 |",
            "| Train-Outcomes-Rolling100/* | 当前级别成功、安全成功、碰撞、超时率 |",
            "| Train-Distance/* | 出生距离、最终距离、回合最小距离 |",
            "| Train-Evader/* | 逃逸机速度、感知决策步比例、撞边界次数 |",
            "| Train-UAV/* | 各机回合末速度、加速度范数 |",
            "| Train-Learning/* | 每1000次更新的 loss/Q/噪声、真实更新计数 |",
            "| uav_0/loss_actor、loss_critic、predictQ_A/B | 原始每次优化日志，共享网络 |",
            "| Test-Results/*、Test-SubRewards-uav_0–3/* | 无探索的评估结果与分项 |", ""]
    for path in report["logs"]:
        text += [f"日志目录：`{path}`", ""]
    text += ["启动：`python examples/run_tensorboard.py --base-dir outputs/logs/matd3/baseline_exp008`。", "",
             "## 已执行验证", "",
             "47项环境/算法/日志测试通过（12.83秒），含5张地图共50次安全出生以及根事件文件奖励值核对。",
             "GPU集成测试正常完成2,048转移，Critic更新1,008次、Actor更新504次；根目录338个标量标签、没有分项子目录、所有标量有限。小样本冒烟测试只验证执行链路，不代表训练效果。", "",
             "## 本轮实际记录", "", "完整回合转移只统计已结束回合；进行中的回合尚未计入，因此不能替代总环境步数。", "",
             "| 级别 | 完整回合数 | 完整回合转移 | 训练安全捕获数 | 最新当前级评估步数 | 安全捕获率 | 碰撞率 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in per_level:
        test = row["latest_evaluation"]
        metrics = f"{int(test['step']):,} | {float(test['safe_success_rate']):.0%} | {float(test['collision_rate']):.0%}" if test else "尚无 | — | —"
        text.append(f"| {row['level']} | {row['completed_episodes']:,} | {row['completed_episode_transitions']:,} | {row['training_safe_successes']:,} | {metrics} |")
    if updates:
        text += ["", f"最新优化记录：Critic {updates[-1]['updates']} 次、Actor {updates[-1]['actor_updates']} 次。"]
    if report["holdout_evaluations"]:
        text += ["", "## 训练结束后复评", "", "| 模型 | 级别 | 种子 | 安全捕获 | 碰撞 |", "|---|---:|---:|---:|---:|"]
        for evaluation in report["holdout_evaluations"]:
            metrics = evaluation["metrics"]
            text.append(f"| {Path(evaluation['checkpoint']).stem} | {evaluation['level']+1} | {evaluation['env_seed']} | {metrics['safe_success_rate']:.0%} | {metrics['collision_rate']:.0%} |")
    text += ["", "来源：process.json、metadata.json、原始 training_episodes.csv、training_updates.csv、curriculum_eval.csv；未执行完的评估不推测结果。", ""]
    output.with_suffix(".md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps(dict(state=process.get("status"), observed_step=observed_step, output=str(output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
