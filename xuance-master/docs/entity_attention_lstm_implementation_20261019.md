# 实体注意力＋冻结LSTM：实现与单种子训练

## 已实现

服务器分支：feat-coverage-reaward。

新增UAV_EntityAttention_MATD3_Policy，保留旧MATD3_Policy入口。Actor、Critic A、Critic B各有独立实体编码器和目标网络；在线双Critic分别初始化。优化器参数组与全部目标网络软更新均覆盖编码器、注意力、LayerNorm和决策头。

每机88维观测保持不变。自身40维加机号4维，经MLP编码为128维；三架队友和五个建筑分别经同类共享MLP编码为128维。自身查询9个实体，使用一层4头注意力、自身残差与128→256→128前馈层。

Actor头128→512→512→3→Tanh。每路Critic将四机编码512维、四机动作12维和查询机号4维拼成528维，再经512→512→1输出Q_i。Actor只使用本机观测，Critic使用集中信息。可训练参数共1,966,981个。

冻结LSTM仍为同一检查点，SHA256：703aa6009a128d8ae23ee41922fcdd696f6cdaaf77fa3df1ab0ff17a798359a7。不改其输入、预测方式、环境动力学、奖励、捕获规则、安全过滤或replay语义。

## 测试证据

- 16项新网络测试通过：切片、建筑填充掩码、实体置换、Actor局部性、双Critic独立性、联合动作梯度、四机顺序、真实Learner延迟更新、目标软更新、存取档、CPU/GPU一致性。
- 原LSTM集成17项回归测试通过。首次运行未设置UAV_LSTM_TEST_RUN而整模块跳过，补齐服务器测试数据目录后重跑通过。
- 4096步实际环境短程训练完成，240次学习更新，LSTM保持冻结，35.44秒（含小规模固定验证）。这不是正式600万步训练的效果结果。
- 正式batch 512、512×512决策头执行14次学习更新，其中后10次计时，每次约16.33毫秒；峰值张量显存561,073,664字节，输出与梯度均为有限值。这是合成样本更新测试，不含环境采样或正式验证耗时。
- 短程初始验证的4个场景与旧实验对应场景的seed和初态SHA逐项一致。正式运行结束时继续核对全部独立测试场景指纹。
- 隔离恢复父级源码归档后应用新覆盖层，698个Python源文件哈希验证通过。原实验冻结源码/配置指纹文件及被引用关键源码未改动。

## 本次训练设置

实验：/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/experiments/entity_attention_lstm_capture_v1_20261019

- 只运行seed 1，只运行实体注意力＋LSTM。
- 每回合最多600步，决策间隔0.5秒，即300秒。
- 固定第五级，16个并行环境，600万环境步；fresh replay及优化器。
- 沿用旧实验学习率、TD3延迟更新、目标平滑、奖励、安全过滤和冻结预测器。
- 每20万步，在相同的100个第五级验证场景选模；最新权重每5万步保存。
- 完成后自动对最佳权重做每级200个场景测试，对最终权重做第五级200个场景测试。
- 自动生成comparison.json和comparison.csv，与旧实验jobs/lstm_seed_1的best/final分别对比；核对场景指纹，并计算逐场配对的捕获率差值及场景bootstrap区间。

对照：/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/experiments/lstm_matd3_capture_long_v1_20261019/jobs/lstm_seed_1

初始化差异必须保留：旧实验加载exp020的MATD3权重继续训练；本次实体Actor/双Critic随机初始化，LSTM仍加载同一预训练权重。因此这是初始化不同的效果比较，不能独立归因于注意力，也不能凭单个训练种子推断训练种子间稳定性。

## 代码与追踪

服务器xuance-master下主要新增：

- xuance/torch/policies/uav_entity_attention.py
- xuance/torch/policies/uav_attention_matd3.py
- tools/run_entity_attention_capture.py
- tools/check_entity_attention_preflight.py
- tests/test_uav_entity_attention.py
- tests/test_uav_attention_matd3.py
- experiments/matd3_entity_attention_lstm.yaml

接线改动限于策略构建/注册和预测环境policy白名单。保持原布局与时序校验，并继续拒绝未支持的其他策略。

训练用独立冻结code目录；源码清单、配置清单、LSTM SHA和对照配置差异均已保存。分支内experiments/source_archives/entity_attention_lstm_capture_v1保存可恢复源码覆盖层、配置及预检报告，父级为lstm_matd3_capture_long_v1。

实验status.json指向当前job状态。训练日志在jobs/attention_lstm_seed_1/process.log；结果在该job的results和test目录。本说明记录启动时的验证情况，不等同于长程训练成功或捕获率改善。
