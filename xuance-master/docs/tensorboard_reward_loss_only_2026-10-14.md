# TensorBoard：仅保留 reward 与 loss

用户于2026-10-14明确表示TensorBoard主要需要reward和loss，其余指标不太需要。本次精简后续baseline实验的TensorBoard写入，并从014原始记录派生了一份精简查看日志；没有启动新的正式训练。

## 当前页面内容

| 图表 | 含义 |
| --- | --- |
| `Train-Returns/uav_0`至`uav_3` | 每架追捕机完整回合的总奖励 |
| `Train-Returns/team_mean` | 四架追捕机回合总奖励的平均值 |
| `Train-Rewards/.../r_purs` | 闭合速度奖励 |
| `Train-Rewards/.../r_obstacle` | 建筑稠密避障罚分 |
| `Train-Rewards/.../r_boundary` | 边界稠密避障罚分 |
| `Train-Rewards/.../r_collision` | 实际接触罚分 |
| `Train-Rewards/.../r_terminal` | 捕获奖励 |
| `Train-Rewards/.../r_failure` | 训练碰撞终止的全队失败罚分 |
| `Loss/loss_actor` | 策略网络优化目标，当前MATD3使用负的第一Critic估值 |
| `Loss/loss_critic` | 双Critic的TD目标预测误差 |
| `Eval-Rewards/Level-*` | 原训练中各级无探索评估的平均回报 |

奖励分项保留四架机和team各一套，均已包含原奖励权重与总缩放。当前不用的distance、progress、teammate、formation奖励标签不再写入这份精简页面。每个reward数据点是20个完整回合的回合累计奖励均值，按实际结束顺序统计；遇到课程切换或训练结束时补记不足20回合的尾批，不将不同级别的回合混为一组。横轴是原环境转移步数。

loss使用现有更新统计窗口的均值，每1000次实际Critic更新记录一次，并在评估区间末尾和结束时补记尾批；Actor均值只计算实际发生Actor更新的值。Actor loss并不以接近0为目标，Critic loss反映价值预测误差。未写入TensorBoard的Q、学习率、运动学、成功率等仍可从原始CSV/JSON进行必要诊断，课程升级和正式成绩的统计保持。

## 014派生查看日志

目录为本机镜像及服务器独立工作树各自的`outputs/reviews/exp014_reward_loss_view_v1/logs/seed_derived`。这是原014数据的筛选和均值汇总，没有重训或更换模型。原来的1,198个标量标签、22,373,424条标量记录被缩减为reward/loss相关曲线；新事件文件为8,839,541字节，约8.84MB。原始完整日志为1,647,313,578字节。

`manifest.json`记录原事件文件SHA256、所用模型和导出代码；`verification.json`记录与原始CSV的逐项核验结果。横轴保留原训练步数，派生日志的wall-time时间戳是导出时间，所以观察曲线应使用Step横轴。原训练中第4级未被单独评估，因此这份原日志导出没有第4级训练评估reward曲线；最终模型第1–5级正式复评继续见014冻结归档。

## 代码和基线保存边界

后续baseline TensorBoard通过`xuance/engine/uav_baseline_logging.py`中的`RewardLossTensorBoard`路由，以及`xuance/torch/agents/base/agents_marl.py`中的baseline专用入口执行该规则。改动仅影响TensorBoard写入，网络、动作、奖励公式、环境、课程判据和CSV/JSON统计逻辑没有修改。针对性测试位于`tests/test_uav_reward_loss_tensorboard.py`及既有baseline集成测试，6项通过。

服务器改动前的源码和改动后的哈希保存在`outputs/reviews/reward_loss_logging_update_v1`；仅限`feat-coverage-reaward`独立工作树，没有提交或推送。014只读归档`outputs/acceptance/exp014_baseline_v1`、模型、配置与原始完整日志均保留，归档有效载荷哈希已重新核验。

当前活动源码的日志实现已经有这项后续修改，故不能再要求它与014训练开始时的所有源码哈希逐项相同；差异为两份日志相关运行源码和一份相应测试。复现014原始代码应使用只读归档的`code/`并显式设置PYTHONPATH；开展后续实验则使用当前独立工作树、新实验名和新快照。
