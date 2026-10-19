# 实验014训练与评估结果

更新时间：2026-10-14T10:45:42.225230+00:00。

已观测环境转移：6,000,000/6,000,000；记录状态：completed；服务器实际训练进程：False。

007最终模型全策略迁移初始化，新优化器、新回放池；从第一级开始。夹角奖励0，逃逸机感知200米，闭合速度权重1.0，总缩放0.1；物理0.1秒、决策0.5秒，其他任务参数沿用013。

捕获条件为同一追捕机连续保持2秒，不要求全队此前无碰撞。捕获率与安全成功率分别统计；正式评估无探索、100回合、最长800步，并统计整个回合的碰撞。007原任务94%不是本轮成绩。

| 级别 | 完整训练回合 | 完整回合转移 | 训练捕获/安全捕获 | 最近当前级评估捕获/安全/碰撞 |
|---|---:|---:|---:|---|
| 1 | 13,580 | 1,001,272 | 3268/3265 | 100%/98%/2%（1,000,000步） |
| 2 | 36,002 | 1,050,078 | 2021/2009 | 100%/91%/9%（2,050,000步） |
| 3 | 23,587 | 3,947,284 | 14145/14039 | 89%/78%/22%（6,000,000步） |
| 4 | 0 | 0 | 0/0 | 未进入 |
| 5 | 0 | 0 | 0/0 | 未进入 |

完整回合转移合计5,998,634，进行中的回合未计入。最近记录的Critic/Actor累计更新：374,744/187,372；训练进行中，计数可能落后于真实执行。

## 最终单一模型五级复评

训练正常结束后核验训练源哈希、冻结最终模型并记录SHA256；每级100回合、8并行、种子71001、无探索、800步上限、完整回合碰撞统计。这些是复评设置；结果出现前不能称为已完成。

| 级别 | 捕获率 | 安全成功率 | 碰撞率 |
|---|---:|---:|---:|
| 1 | 100% | 61% | 39% |
| 2 | 100% | 88% | 12% |
| 3 | 90% | 71% | 29% |
| 4 | 44% | 4% | 96% |
| 5 | 1% | 0% | 100% |

## 原始记录

- [修改说明](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/docs/baseline_exp014_changes_2026-10-14.md)
- [真实配置](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/results/matd3/baseline_exp014/config.json)
- [初始化审计](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/results/matd3/baseline_exp014/initialization_audit.json)
- [课程评估与旧级复测](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/results/matd3/baseline_exp014/curriculum_eval.csv)
- [训练回合](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/results/matd3/baseline_exp014/training_episodes.csv)
- [优化更新](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/results/matd3/baseline_exp014/training_updates.csv)
