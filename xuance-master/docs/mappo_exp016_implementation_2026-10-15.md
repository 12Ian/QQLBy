# MAPPO 五级课程接入与 exp016 验证

2026-10-15。工作分支：`feat-coverage-reaward`；实验位于服务器工作树
`/home/ryy/UAV_Mine_baseline_coverage/xuance-master`。

## 实现

- 复用现有四架无人机、三维建筑、基线奖励和预测制动安全过滤器。课程目标最高速度依次为 3、5、8、10、12 m/s。
- MAPPO 沿用 Xuance 的共享高斯 actor、联合观测 critic、PPO clip、GAE 与 ValueNorm。actor 训练时采样动作，评估时取分布的确定性动作。
- 课程评估记录每局成功与建筑、边界、机间碰撞，统计无碰撞成功率。晋级仍需当前级别至少 10 万环境步、100 局评估、无碰撞成功率不低于 0.8、碰撞率不高于 0.1，并连续两次通过。
- 多智能体检查点现在保存并恢复 ValueNorm 状态。训练日志将 PPO actor/critic loss 与 value 预测接入现有 CSV 和仅奖励/损失的 TensorBoard 视图。
- 完整运行在训练末尾对全部五级分别进行 100 局评估。`exp016` 从随机 MAPPO 权重开始；MATD3 的检查点与 MAPPO 架构不兼容。

正式参数在 `experiments/mappo_exp016_safety.yaml`；训练入口为 `examples/run_baseline.py`。源代码在服务器实验目录 `outputs/experiments/exp016_mappo_safety/code` 冻结，`source_manifest.json` 保存逐文件 SHA-256，监督进程写入 `process.json` 和最终 `verification.json`。

## 启动前验证

- CUDA 短训练：1,024 环境步，8 次完整 rollout 更新；五级各 2 局终评正常写出。此短训练没有学会围捕，不把 0 成功率解释为 MAPPO 最终效果。
- 全新 agent 装载检查点后，ValueNorm 张量逐项一致；同一观测改变随机种子，评估动作逐项一致；第五级评估指标计数为 2 局。
- 独立晋级探针使用仅供测试的宽松门槛，在 0 和 512 步请求升级，训练局实际进入第 2、3 级。
- 相关回归测试 65 项通过。旧测试中第五级速度仍写 9 m/s 的断言已同步改为 12 m/s。

## 正式运行

实验名：`exp016_mappo_safety`；随机种子 1；16 个训练环境；总预算 6,000,000 环境步；每 50,000 步做当前级别评估；每次 100 局；最终五级各 100 局。是否到达第五级由安全成功率门槛决定，不能从已启动或短训练推断。

服务器结果目录：`outputs/results/mappo/baseline_exp016_mappo_safety`。监督与完整日志：`outputs/experiments/exp016_mappo_safety`。实验完成后，以 `process.json` 的 `status=complete`、`verification.json` 的源代码哈希和 600 万步计数、五级终评的局数及安全成功率为验收依据。
