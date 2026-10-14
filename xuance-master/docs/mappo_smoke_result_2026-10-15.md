# 当前 MAPPO 运行检查与新目标速度

2026-10-15。结论：既有MAPPO的环境交互、PPO更新、模型保存/载入和CPU/CUDA运行路径可以跑通；当前MATD3专用baseline课程benchmark还不能直接用于MAPPO。

本次采用当前四机coverage环境及执行安全模块，原始观测79维、每机三维加速度建议动作。使用已有MAPPO_Clip_Learner、Basic_MLP表示层256、Actor/Critic各512/512、参数共享、GAE、ValueNorm和标准Huber，没有改写MAPPO优化算法，也没有接入MAT。

用户指定的后续五级目标最高速度 **3/5/8/10/12 m/s** 已在服务器和本机同步：更新环境缺省课程、无编号基础配置，并从015任务配置建立 `experiments/baseline_current.yaml`。该新任务配置保留015的四机、地图、出生、动力学、观测、捕获、奖励及安全模块，只调整第3/4级最高速度和任务标识。

原编号014/015配置与冻结归档保持原样。后台015仍按其冻结的3/5/7/8/12 m/s运行，不能把它的第3/4级结果当成新任务结果。已核验被冻结的环境、基础配置、测试和015配置哈希均未改变；本次检查时015仍在训练。

| 检查 | 实际执行结果 |
|---|---|
| 五级速度与空间 | reset后最高速度逐项为3/5/8/10/12；五级均79维obs、53维state、20/40/60/80/100栋建筑 |
| 已有课程回归 | 5项通过，验证课程只在reset生效、地图稳定、实际速度及上下文一致 |
| CPU主短训 | 2并行，512个环境转移，32次PPO optimizer更新，800步回合上限 |
| 截断/自动reset | 2并行，128个转移，4次更新；专门设16步上限，发生8次回合截断并正确继续采样 |
| CUDA短训 | cuda:0，2并行，128个转移，4次更新；实际参数位于CUDA设备 |
| 参数与数值 | Actor及其表示层、Critic及其表示层存在参数更新；记录的loss、输出、梯度及模型张量均有限 |
| 模型恢复 | 故意改动内存参数后载入保存文件，全部policy张量精确恢复；新建独立agent载入后，同输入动作均值精确一致 |
| 原生评估路径 | 第1/5级各2个完整回合，均到800步超时；native test_mode采用随机采样 |
| 安全执行 | 上述检查没有记录碰撞或未认证子步；动作输入没有被环境原地修改 |
| 最终进程退出 | 修正测试日志关闭后，v2完整重跑，退出码0 |

合计768个训练环境转移、40次PPO优化器更新。完整回合评估另计3200个转移，不作为训练预算。参数、epoch和rollout为短测设置，不是正式MAPPO训练方案。512个转移不足以评价捕获效果；四个评估回合均超时，不声称获得了学习后的成功率。

后续正式运行需要补齐以下接口：

1. `RunnerMARL._run_baseline_benchmark()`访问`agent.noise_scale`，实际MAPPO没有此属性，原入口抛出 `AttributeError`。这是UAV benchmark与on-policy agent的兼容问题；普通train路径和PPO更新已经通过。
2. 原生checkpoint保存policy/optimizer与部分RNG，没有保存ValueNorm统计。新建agent载入后，policy及动作均值正确，但ValueNorm与原agent不一致；不能将此文件称为完整的规范续训状态。
3. 原生MAPPO的`test_mode=True`仍随机采样。正式无探索评估需要明确选择均值，并接入与MATD3一致的安全成功/碰撞统计。当前reward/loss精简日志也需要PPO字段适配；本次训练指标已另存audit，不以TensorBoard显示证明运行。

首轮v1所有模型检查完成，但测试工具的Tee在日志文件关闭后仍被Python退出流程刷新，导致退出码120。已在最小子进程中复现：旧处理退出120，恢复原stdout/stderr后退出0。随后修正测试工具，在新的v2目录完整重跑通过。没有据此修改MAPPO算法；v1失败证据与v2结果分别保留。

可复核文件：

- [新后续任务配置](C:/Users/renyi/Documents/ChatGPT/服务器RL/baseline_work/experiments/baseline_current.yaml)
- [速度更新约定](C:/Users/renyi/Documents/ChatGPT/服务器RL/baseline_work/docs/target_speed_schedule_2026-10-15.md)
- [速度修改来源与哈希](C:/Users/renyi/Documents/ChatGPT/服务器RL/baseline_work/docs/target_speed_schedule_2026-10-15_manifest.json)
- [完整运行审计](C:/Users/renyi/Documents/ChatGPT/服务器RL/baseline_work/outputs/reviews/mappo_smoke_2026-10-15_v2/audit.json)
- [测试脚本](C:/Users/renyi/Documents/ChatGPT/服务器RL/baseline_work/outputs/reviews/mappo_smoke_2026-10-15_v2/smoke.py)
- [MAPPO短测配置](C:/Users/renyi/Documents/ChatGPT/服务器RL/baseline_work/outputs/reviews/mappo_smoke_2026-10-15_v2/mappo_smoke.yaml)
- [测试日志](C:/Users/renyi/Documents/ChatGPT/服务器RL/baseline_work/outputs/reviews/mappo_smoke_2026-10-15_v2/test.log)
- [回归日志](C:/Users/renyi/Documents/ChatGPT/服务器RL/baseline_work/outputs/reviews/mappo_smoke_2026-10-15_v2/speed_tests.log)
- [本机证据核验](C:/Users/renyi/Documents/ChatGPT/服务器RL/baseline_work/outputs/reviews/mappo_smoke_2026-10-15_v2/local_verification.json)

服务器独立测试目录为当前工作树下 `outputs/reviews/mappo_smoke_2026-10-15_v2`；只保存短测材料，不启动新的长训练。
