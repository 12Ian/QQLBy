# MATD3 实体注意力接入审查与实现草案

日期：2026-10-19。核对对象：feat-coverage-reaward，提交 8ed57eef96152bfa283ba8b74206c009420354ae，以及随提交归档的 lstm_matd3_capture_long_v1 配置。

## 结论与验证边界

当前提交尚无本次讨论的实体注意力策略。以下是对实际接口与拟采用结构的静态审查，不是新网络已经通过运行或长程稳定性测试的报告。本轮不修改训练运行时、不启动训练。

原网络的两个512单元隐藏层并非容量很小。此前训练退化不能单独证明欠拟合；实体注意力的主要假设是按对象编码、学习关系可以改善归纳偏置。必须通过同条件实验判断捕获收益。

首版采用一层局部实体注意力。Actor 使用本机编码；每路 Critic 独立编码四机观测，再按固定机号顺序拼接所有编码、动作和查询机号。暂不叠加第二层机间注意力，不同时缩减现有512×512决策头，以减少实验变量。

## 已核实的观测契约

当前LSTM包装器输出88维观测，算法额外提供4维机号。原79维输入与新增9维预测位移的切片如下：

| 内容 | 切片 | 维数 |
|---|---|---:|
| 自身位置、速度、加速度、捕获保持进度 | 0:11 | 11 |
| 三架队友相对位置和速度 | 11:29 | 3×6 |
| 六个边界距离 | 29:35 | 6 |
| 目标当前相对位置、速度等 | 35:45 | 10 |
| 课程上下文 | 45:49 | 4 |
| 目标未来1/2/3秒位移 | 49:58 | 9 |
| 五个建筑槽位 | 58:88 | 5×6 |

自身分支拼接11+6+10+4+9=40维，再加入4维机号，得到44维。队友分支为3个6维实体。建筑分支为5个6维实体。

课程上下文实际为：零基等级/(等级数-1)、建筑总数/100、目标最高速度/追捕机最高速度、捕获半径/100。速度比可能超过1，不再次裁剪原始输入。

建筑每行是相对x下界、x上界、y下界、y上界、顶部高度及有效标志。不是中心位置加三维尺寸。直接沿用既有几何表达，不擅自改写其物理意义。

环境现有 _nearest_obstacle_features 从建筑列表中选择最近K栋；此处没有新增视线遮挡判断。注意力掩码只屏蔽填充槽位，不能据此宣称实现了遮挡感知。

建筑有效标志是既有观测的一部分，与此前排除的“历史/预测有效标志”不是同一概念，不增加观测维度。

## 编码器具体计算

UAVEntityAttentionEncoder(obs, ids) -> h，h宽度128。

- self_mlp：44→128→128，隐藏激活沿用当前配置。
- teammate_mlp：6→128→128，三架队友共享此MLP。
- building_mlp：6→128→128，五个建筑共享此MLP。
- 三类MLP参数互不共享。不同实体类别由各自MLP区分。
- 一个自身token、三个队友token、五个建筑token，组成 [N,9,128]。
- 使用 nn.MultiheadAttention(128, 4, dropout=0, batch_first=True)。每头32维。
- 只查询自身：Q=LN(tokens)[:,0:1]，K=V=LN(tokens)，输出 [N,1,128]。
- u=h_self+attention_output；h=u+FFN(LN(u))，FFN为128→256→128。
- 残差保存未归一化的自身路径；首版不用BatchNorm或Dropout，不叠加观测running normalization。
- 这是面向自身的多头注意力，借鉴论文的编码和残差思路，不声称逐算子复现论文全网络。
- 队友和建筑内部不添加槽位位置编码。替换同类实体的行顺序时，其注意力聚合结果应保持一致。

建筑mask从原始第6维读取：invalid = building[...,5] < 0.5。先将无效建筑行全部置零，再做MLP，避免填充坐标或异常值进入映射；再将其在key_padding_mask中设为True。自身和三架队友设为False，故始终有合法key。

必须先生成valid，再将无效行用masked_fill清零，不能用“原始值×0”处理NaN。在线所有合法输入仍须做有限值检查。屏蔽行发生变化不能改变输出。

这里的9个token是同一时刻的不同对象，不是9帧历史，不使用时序因果三角mask。历史因果性由冻结LSTM的输入构建保证。

## Actor与集中式双Critic

Actor：每机(obs_i, ID_i)→Actor专用编码器→128→512→512→3→Tanh。一次批处理 [B×4,88]，四机共享同一Actor参数。机号已进自身分支，Actor头不再重复追加。

Critic A：四机观测分别经A专用编码器，得到 [B,4,128]；按固定agent_keys顺序展平成512维；拼接四机动作12维和查询机号4维，得到528维；Q头528→512→512→1。每个场景输出四个Q_i。

Critic B：同结构、独立随机初始化、独立参数，不能从A复制出在线B。A、B和Actor之间也不共享编码器。每个在线网络分别深拷贝成自己的目标网络，目标参数不加入优化器并设置requires_grad_(False)。

Critic内部对四机使用的编码器权重共享；A和B之间不共享。编码时每机使用其自己的建筑槽位和预测值，不把建筑并集提供给Actor。

保留现有MATD3语义：Qpolicy返回(q_A,q_B,(q_A+q_B)/2)，Qtarget返回min(q_A_target,q_B_target)。当前长程配置td3_actor_use_first_critic=true，Actor实际由Q_A优化。不能无意改成MAPPO的V(s)，或丢掉联合动作。

联合输入必须先拆分 joint_observation→[B,4,88] 与 joint_actions→[B,4,3]，再分别处理。现有Learner传入的是观测整体和动作整体，不是四个(o_i,a_i)交错的序列。

返回Q_i必须按batch优先、机号其次展平到[B×4,1]，与reward、terminal、agent_mask、ID顺序相同。不得以仅shape相同作为顺序正确的证据。

## 代码落点

以下路径均相对服务器 xuance-master 根目录：

| 文件 | 责任 |
|---|---|
| xuance/torch/policies/uav_entity_attention.py（新增） | 严格切片、实体MLP、建筑mask、自身查询注意力 |
| xuance/torch/policies/uav_attention_matd3.py（新增） | Actor、独立双Q、目标网络、参数组、旧策略接口 |
| xuance/torch/agents/multi_agent_rl/matd3_agents.py | 新policy构建分支，验证共享参数、四机、88维及三维动作等契约 |
| xuance/torch/policies/__init__.py | 注册新策略，不替换旧MATD3_Policy |
| experiments/matd3_entity_attention_lstm.yaml（新增） | 独立配置，显式设置注意力128维、4头、1层 |
| tests/test_uav_entity_attention.py（新增） | mask、实体顺序、分支隔离、极端合法输入 |
| tests/test_uav_attention_matd3.py（新增） | 接口、梯度、参数组、目标更新与四机顺序 |

新策略命名UAV_EntityAttention_MATD3_Policy。接口维持forward、Atarget、Qpolicy、Qtarget、parameters_actor、parameters_critic、soft_update。优先复用现有MATD3_Learner，避免改变算法更新公式。

root工作树与归档运行时并非逐文件相同：部分历史HGAT文件仅在实验源码归档中，不能把本地baseline_work里存在的HGAT类当成当前分支已注册接口。新类应明确注册和接线，不盲目依赖这些历史类。

parameters_actor必须覆盖Actor编码器与头的所有可训练参数；parameters_critic覆盖A/B编码器和头，排除Actor与所有target。优化器参数集合按对象ID核对，无遗漏、无重复。

soft_update按对应名字更新所有编码器、注意力QKV、LayerNorm和决策头。tau=0、tau=1及中间值分别验证。不用只有头部更新、编码器目标固定的实现。

首版只处理已用到的单学习器、参数共享、feed-forward模式；未实现的RNN/分布式/不同观测布局在构建期明确报错。LSTM仍在环境包装器中冻结，不把MATD3设置为use_rnn。

## 稳定性判断与检查

静态检查已确认维度与现有Learner接口可相容。运行稳定性尚待以下实际测试：

1. 用哨兵值逐项验证88维切片和四机展平顺序；batch至少包含两个不同场景。
2. 建筑0/1/5个有效槽位，极近建筑、边界、重合队友及大于1的速度比，输出/梯度均有限；无效建筑注意力权重为零。
3. 任意置换队友行、建筑行及其mask，输出在浮点容差内不变。固定ID时不要求整队机号重排不变。
4. 修改同批其他飞机的私有建筑/预测，只能影响集中Critic，不能影响本机Actor（本机原始obs保持不变）。
5. A/B无参数对象共享，初始化后的输出不要求相同；目标初始化分别与对应在线网络一致。
6. Critic损失会训练其编码器和Q头；Actor损失通过动作保留dQ/da到Actor，不能把Critic前向整体包进no_grad。检查Actor步仅改变Actor参数、Critic步仅改变Critic参数。
7. 目标网络不接收梯度；软更新与存取档恢复一致；存档包含布局、policy名称、预测器SHA与配置。
8. 执行至少一次真实Learner完整延迟更新周期，核对梯度、优化器及target，而不只测试独立模块forward。
9. 新网络使用新空replay，不将旧MLP检查点做不严格加载后宣称继承原策略。旧79/88维MLP无法精确映射到新实体网络。
10. 隔离运行时内做短程环境训练，监控双Q/TD误差、梯度裁剪前范数、动作饱和率、碰撞率、吞吐及峰值显存，再决定长程实验。

注意力权重熵可作辅助诊断，但低熵或高熵本身都不能认定失败，注意力热图也不能证明捕获策略正确。

当前长程配置actor学习率3e-5、critic学习率1.5e-4、tau=0.001、梯度裁剪0.5、target噪声0.2、clip0.5、gamma约0.99499。先保留这些设置作为可追溯起点，不声称已适用于新网络；若要调参，按固定验证集做独立实验记录。

## 比较设计

先固定同一预测输入、奖励、课程调度规则、步数预算和初始化方式，比较MLP+LSTM与实体注意力+LSTM；二者都从随机强化学习权重开始，LSTM都加载同一冻结预测器。课程规则相同不代表自适应晋级时实际等级暴露相同，必须一并记录。

如需将收益进一步归因到注意力而非增加参数，再补同预算实体MLP池化消融。注意力的收益未成立前，不直接扩展到两层机间注意力或同时更改奖励。

历史exp020的42%使用800步评估。新对照采用同一测试场景与同一步数上限；600步和800步结果分开报告。对照使用配对训练种子与场景、多训练种子，并报告各级捕获、碰撞、捕获时间、训练后退化及置信区间。验证选模与最终测试分离。

## 参考依据

- OPEN，IV-B.2：三类观测MLP、128维注意力、self残差；原方法为MAPPO，Critic为状态价值，不能原样替代MATD3双Q：https://arxiv.org/html/2409.15866v3
- Actor-Attention-Critic：集中式注意力Critic与分散执行的参考；此处首版暂不复制其完整算法：https://proceedings.mlr.press/v97/iqbal19a.html
- PyTorch MultiheadAttention：batch_first张量约定及key_padding_mask=True表示忽略该key：https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
