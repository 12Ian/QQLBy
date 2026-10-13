# 论文通用算法对应与民用适配

来源为用户提供的 Yu 等2026年 TAES PDF，DOI `10.1109/TAES.2026.3677846`。本实现对应其通用学习结构、数学运算、优化和预测指标；输入图、教师因素和验证数据作民用适配，不能称为论文全部实验复现。v0.4新增已测邻居分支和双实体图，细节见 [INTERACTION_GUIDE.md](INTERACTION_GUIDE.md)。下表同时列出单轨迹与交互配置。

| 来源 | 已实现 | 代码 | 民用适配/明确差异 |
|---|---|---|---|
| 式5–7、图4 | 意图分类→后验条件预测；独立预训练、冻结与联合训练 | model.py/workflows.py | v0.4可输入同步民用邻居历史；仍只预测焦点未来 |
| 式12/14、IV-A | 6维观察历史BiLSTM、双向末hidden拼接、128 hidden、64投影、ReLU/Softmax；静态FC分支 | IntentionEncoder | 交互配置取邻居末个有效状态；单轨迹配置取自身末状态；增加LayerNorm |
| 式15–16 | 条件节点嵌入 `h^(0)=[g; [h_t; h_s; p̂]]`，融合历史、意图拼接 z 和后验 | Forecaster | 双实体2O历史节点、实体标签、时序边和民用空间邻近边；军用对抗边见式15 一行 |
| 式17–21、IV-B裁剪 | 多头GAT、边/节点sigmoid乘法门控、ELU、残差LayerNorm；边描述、硬掩码、渐进裁剪 | TemporalGraphAttention/history_graph.py/interaction_graph.py | 7维位置/速度/时间差；有效测量、历史一致性、跨实体同期距离半径、Top-K和self回退；不使用原文专用几何 |
| IV-B结构 | d=256，6图注意力头，各64维；2图层、2时间层、4时间头；Top-K8、温度.1 | configs/server.json/paper_forecast.json | 时间层用TransformerEncoder；6×64投影回256，避免错误要求256被6整除 |
| IV-B解码、式32–33 | 自回归GRU、观察记忆cross-attention、三维增量cumsum锚定末位置 | Forecaster | 恒速残差初始化；预测反馈使用自己的输出，不做未来真值teacher forcing |
| 式34–35 | 欧氏ADE/FDE，单位米 | displacement_errors | 严格区别于MSE；另给恒速基线和RMSE |
| 式36–37、41 | 三组互斥head、detached教师KL、熵下限、相邻时刻抖动、跨组cos²分离 | objective.py | 三教师为运动平滑、速度一致、时间邻近，只读观察段 |
| 式54–55 | 可学习同方差log-variance加权；回归.4ADE+.6FDE；分离.1 | TrajectoryObjective | 四任务为预测和三种民用教师；不确定性权重限制[-6,6] |
| IV-A数据/优化 | 70/15/15整独立分组、训练集标准化、Adam1e-3/70轮/256批、30轮StepLR×.5、clip1 | paper_intention.json/交互配置 | 单轨迹按轨迹、耦合实体按整场景；yaw共同坐标属于适配；StepLR每30轮衰减 |
| IV-B优化 | AdamW1e-3，weight decay1e-4；5轮warm-up、cosine、200轮上限、patience20 | learner.py/server.json | 同时监控ADE/FDE显著改善；验证加权ADE/FDE选模；低速min_delta=.001米，原文1米可配置 |
| IV-B采样 | dt=.1、O=80、P=60 | paper_forecast.json | 与IV-A的dt=.5/O=20分开；两阶段始终采用兼容的同一协议 |
| IV-B基线 | LSTM/GRU两层256宽；TCN四因果残差块,k=3,dilation1/2/4/8；Transformer2+2层,d256,H4,FF1024 | baselines.py | 同输入和训练划分；均以恒速残差输出增量；实际参数量记录 |
| IV-B四格消融 | +I+T、-I+T、+I-T、-I-T | workflows.py/compare | T为三组民用历史教师，不是原文E；同一初始化种子的图模型共享冻结分类器 |
| 民用交互因素消融 | +I+C、-I+C、+I-C、-I-C | workflows.py/compare | C为已测邻居信息，保留T；关闭C时仅焦点O节点、静态邻居输入为零，输出对邻居值不变 |
| 在线更新流程 | 更新已测历史、重估后验、重建图、刷新未来预测 | rolling.py | 可接收同步双方实测状态及缺测mask；不使用控制器，不将预测当成新测量 |
| 训练与实验图 | CE/准确率、预测训练曲线、混淆矩阵、ADE/FDE与e(t)、3D轨迹图 | reporting.py | 从保存的测试结果直接绘制；报告种子SD，四复合类首次测试窗口，无误差筛选 |

## 数学实现

历史状态 `s_t=[x,y,z,vx,vy,vz]`。位置减去最后观察位置，可选按末观察水平速度的yaw旋转；以训练历史拟合的均值/标准差归一化。网络输出真实观察历史的分类后验，不输入标签。BiLSTM中的反向方向仍全部处于观察窗口内。

GAT单头的门控注意力为：

`alpha_ij = softmax_j(e_ij / tau + r logsigmoid(f_edge(h_i,h_j,edge_ij)) + r logsigmoid(f_node(h_j)))`。

`e_ij` 是LeakyReLU加性注意力打分，softmax只对允许邻居归一化。用log-sigmoid保持极小门控数值稳定。Top-K始终预留self边，极端掩码不会产生全行负无穷。边线性变换按两个节点分解，避免分配B×O×O×2D的大张量。

民用边描述为7维：归一化的三维位置差、三维速度差、绝对时间差。自身历史两点的允许边需满足 `||p_i-p_j|| <= V|t_i-t_j| + eps_p(2-r)` 和 `||v_i-v_j|| <= A|t_i-t_j| + eps_v(2-r)`，再交时间半径和速度掩码。它是观测一致性检查，不保证未来动力学可达性，也不重现原文两实体可达约束。自环始终保留。

`r=min(1,更新次数/(每轮批次数*crop_warmup_epochs))`；软门控影响从0增至1，Top-K上限从O降到K，测量容差由2倍降到1倍。论文没有给出这个调度的具体解析式，此处为明确的民用实现选择；历史时间半径仍生效，因此初期也不是全连接图。`r`存为模型buffer，更新次数与优化器一同恢复，推理采用所选检查点的裁剪程度。旧配置默认关闭这些新增参数，以兼容既有检查点。

解码增量 `delta_p_t = v_last * dt + dt * residual_t`，绝对输出 `p_hat_t = p_last + sum_{k<=t} delta_p_k`。自回归分支反馈上一预测增量；同一观察记忆通过cross-attention供各未来步使用。可选民用速度/加速度投影在累加前执行。预测期间不接触未来位置、未来速度或真实意图。

民用教师分数（观察段i,j）：

- 平滑性：`-||a_i-a_j||² / A²`，`a`从观察速度差分得到。
- 速度一致性：`-||v_i-v_j||² / V²`。
- 时间邻近性：`-|i-j| / (O-1)`。

每组教师在学生有效邻居集合内按温度softmax，随后detach。head分成大小相等的三个互斥组；训练KL(teacher||student)，附加熵下限、对齐行/列后相邻时刻L1抖动和跨组注意力向量cos²分离。教师尺度取民用配置上限（关闭上限时采用明确默认尺度）。这些分数不是原文专用因子的近似替身，教师内容已经改变。

`L_geo=.4 ADE+.6 FDE`。令 `L_r` 为几何及三组加权教师损失，`s_r`为log variance：

`L = sum_r[.5 exp(-s_r)L_r + .5s_r] + lambda_sep L_sep + lambda_s mean(a_pred²)`。

联合模式另加分类CE；分类阶段只训练CE；固定权重模式可配置。`s_r`及optimizer/scaler都纳入检查点。loss值可能由于可学习log variance变为负数，评价准确性应看位移指标和分类指标，不能跨不同目标直接比较total loss。

## 军用拦截交战版对应（S1–S7）

| 来源 | 军用实现 | 代码 | 与原文的差异/工程决策 |
|---|---|---|---|
| 式1–2、Table IV/V | 目标/防御机质点运动学；r、ṙ、e_L、ė_L、V_L；r ≤ (2/3)r₀ 触发反LOS逃逸、min(ā_eva, a_T,max) 饱和、基元段内保持常值；9g/15g 与 60° FOV | `engagement_scenarios.py` |  trapezoidal 速度保持积分；速度为标称常值（过载只改方向） |
| IV-A 两套意图 | 9类语义意图 + 11档过载强度（ā_n 落在 (I_m−.5, I_m+.5]，tie 归较大档） | `data.py` `intention_task`/`overload_labels`、`engagement_overload.json` | 决策 D5：双任务共享 BiLSTM 主体，分类头独立；窗口标签按观察窗多数投票 |
| 式15、Table I 边 | 19维边：7维几何 + 9维交战描述符 + 3类边 one-hot（时序/空间/对抗） | `engagement_graph.py` | 对抗边 = 跨实体 ∧ 近距 ∧ (可见 ∨ 接近) |
| 式22–25、Table I 裁剪 | 候选分支展开 + `m(j)=Cesc·Cenv·Cene` 硬掩码 + `exp(−Λ_crop·P_ij)` 软惩罚（随 crop_progress 递增） | `candidate_pruning.py`、`model.py` | 决策 D1 锚定候选法：A 个锚点方向挂 O 大小候选块；Cesc 用 ṙ ≥ ṙ_min（FOV 不可见时旁路）；Cenv 简化为高度带；Cene 用时间平均加速度增量 v·θ²/T²（式24）；软惩罚 P_ij 用 a_req/a_max；集合裁空时保留最小代价分支 |
| 式42–47 能量教师 | a_req（盘旋）、c_r（制动）、jerk 的 LSE | `military_teachers.py` | 归一化用 9g 与 T_p；κ 可配置 |
| 式48–49 风险教师 | 突防风险 LSE(r, ṙ⁺, 可达域) 后，再按式49 对 τ∈[t,t+H] 做加权 log-sum-exp 聚合 | `military_teachers.py` | ṙ 取正部（逃逸率）；w_{j,τ} 为时间衰减；窗口内最危险时刻主导 |
| 式50–53 时效教师 | 走廊径向/法向偏差与终端收敛，μ_t=sigmoid(κ(2t/O−1)) 门控 | `military_teachers.py` | 偏差按式(50)(51) 取 δ=(dist−b)/b（走廊内为负）；半宽可配置，默认 0.2·(窗口平均速度·T_p) |
| 式54–55 总损失 | 结构与系数对齐（λ_sep=.1、ω=.4/.6、log_var clamp[−6,6]） | `objective.py` | 复用民用不确定性加权框架 |
| IV-A/IV-B 采样协议 | dt=.5/O=20（5:1 抽取）与 dt=.1/O=80/P=60；IV-B 80/20 无测试集 | `engagement_*_json`、`split_tracks(holdout_test=)` | 决策 D4：统一 0.1s 内部积分；测试集缺失时评估回落验证集 |
| IV-B 四格消融 | I × E（E = 交战边 + 候选裁剪 + E/R/T 教师） | `workflows.py compare` | 新增 `graph_no_engagement`、`graph_no_intention_engagement` |
| 逃逸前后统计 | 评估按触发状态分组报告 ADE/FDE | `engine.run_epoch` `evasion_phases` | 分组标志取观察窗末步的 evasion_mask |

`predict`、`rolling-evaluate`、`report`、`study --data-mode engagement` 均已接入军用配置；教师只在训练期使用（Remark 4，推理不输入）。

### 论文对齐补充（P0/P1 修复后，2026-10-11 第二轮）

| 来源 | 军用实现 | 代码 | 与原文的关系 |
|---|---|---|---|
| 式(8)(9)(13) Q_int/Q_tra | 意图静态分支输入 = 防御机状态 s_M（6维）+ 交战因子 s_L（9维：r/ṙ/e_L/ė_L/可见性）；同一描述符按时间广播进图节点特征 | `data.Track.engagement`、`model.forward(engagement=)` | 式13 `h_s=ReLU(W_s·s̃_M+b_s)`；Remark 2 要求 Q_int/Q_tra 由 `s_L=[r,V_L,e_L]` 及其时间变化表征 → 6+9 与之逐项对应 |
| 式(12)(14) 意图编码器 | BiLSTM 双向末隐状态拼接 → LayerNorm → Dropout → FC m=64 → ReLU → 与 h_s **拼接**为 z=[h_t;h_s] → 分类头(2m→m→C) → Softmax | `model.IntentionEncoder` | 式12/14 有两支（序列 + 防御机静态），故**保留**静态分支；式14 直接吃拼接 z，无额外 fusion 层 |
| IV-A 九类语义 | 基元**按语义类别抽样**（含"无机动"直飞段），窗口标签取观察窗内主导基元；`balance_intention_classes` 做窗口级类别平衡 | `engagement_scenarios.sample_semantic_class`/`_primitive_from_class`、`data.Windows._balance_classes` | 论文 "maneuver classes are sampled with balanced coverage"；重采样只在窗口层，轨迹物理不变 |
| IV-A 十一档过载 | 带符号法向过载（速度系横向/纵向主分量定号）→ 按 `(I_m−0.5, I_m+0.5]` 量化；`overload_gap_report` 报告论文规则留下的 1 g 空档 | `engagement_scenarios.signed_normal_overload`/`overload_label` | 式(4) 是模长（非负），但 I={−10…+10} 要求符号，故取带符号主分量；±10 档在 `a_T,max=9 g` 下仅在空档回退时可达 |
| 式(38)–(41) 教师正则 | `teacher_regularizers="paper"`：熵作用于组内头均值 β̄、抖动取 β̄ 相邻时刻差、下限用双边 `abs()` | `objective.attention_terms` | 逐式改写；分离项（式41）按三重求和口径实现（对跨组有序头对求和再对 batch 取均值），量级由 λ_sep 与不确定度加权吸收 |
| 式(55) 总损失 | `L_tra = L_attn + 1/(2σ̄²_sup)(ω1·ADE+ω2·FDE)`：几何项带 `1/(2σ̄²_sup)` 权重但**不**带 `log σ̄_sup`（该正则属式54，只作用于三组注意力教师）；军用 `smoothness_weight=0` | `objective.TrajectoryObjective.forward`、`engagement_*.json` | `log_variance` 索引 0–2 为 H_E/H_R/H_T，3 为 σ̄_sup；几何项进入不确定性加权 |
| IV-B FOV 掩码 | `fov_attention_mask=true`：目标不可见时删除跨实体边 → 退化为纯外推 | `engagement_graph.engagement_edges` | 论文 "the FOV mask is enabled during interaction" + 不可见时自然退化 |
| IV-A 无预测域 | `stage=intention` 允许 `prediction_steps=0`，窗口只取 O 行 | `config.validate`、`Windows`、`objective`、`metrics` | 论文 IV-A 是独立分类实验；无预测域时仍报告宏 F1/多数类基线/逐类召回 |
| Table VI / Fig.14 | `FIGURE14_SCENARIOS` 四个复合机动初始条件 + `simulate_engagement_batch` 确定性仿真 | `engagement_scenarios`、`scripts/reproduce_figure14.py` | Table VI 数值取自论文 PDF 图版（文本层缺失）：r0=5.3 km、e_L0=[0.98, ±0.15, ±0.09]、VT=300、VM=360、To=8 s、Tp=6 s；论文终端误差 150/202/81/173 m |
| 每标签 1000 次蒙特卡洛 | 按标签有放回独立重采样（各自种子）后统一评估 | `scripts/monte_carlo_intention.py` | 论文 IV-A "1000 independent Monte Carlo trials with distinct random seeds for each maneuver label" |
| 目标值核对 | 88% / 94% / 102 m / 四场景终端误差 四项对照（±15% 判据） | `scripts/check_paper_targets.py` | 复现进度量化 |
| 数据规模与几何覆盖 | `--target-windows` 自动求解 `--scenes` 至 5×10⁴ 窗口；`--initial-fov-coverage` 控制 FOV 内外比例（IV-A 0.5 / IV-B 1.0） | `cli.py`、`bash/run_engagement.sh`（`TRAJECTORY_TARGET_WINDOWS`） | 论文 "loop is repeated until the target dataset size and geometry coverage are achieved" |
| IV-B 逃逸强度有界采样（C6） | 触发逃逸后 ā_eva 每 ~4 s 段从 `evasion_overload_range`（默认 0.5–1.0 × aT_max）重采样，段内保持（`a_eva=min(ā_eva,aT_max)` 饱和） | `engagement_scenarios.simulate_engagements`、CLI `--evasion-overload-range` | 论文 "Segment duration and the nominal overload level ā_eva are sampled within predefined bounds … holding a_eva within each short primitive segment and updating it at segment transitions" |
| 交战终止（C9） | 拦截（r ≤ `intercept_radius_m`，默认 50 m）或确定错过（逃逸后分离连续增长 `miss_grow_steps` 步）即终止；终止后尾部状态冻结，定长数组/窗口协议不变；终止步与原因写入 diagnostics | `engagement_scenarios.simulate_engagements`、CLI `--intercept-radius-m/--miss-grow-steps` | 论文样本为 "a complete engagement trajectory segment"；终止防止拦截后穿越伪影进入训练分布 |
| 不可达链路硬阻断（A7） | 跨实体边 (i,j) 需 `a_req=\|v_rel\|·θ/lag ≤ aT_max`（θ 为相对速度与 LOS 夹角），否则删边；同实体时序边与自环不受影响；`overload_reachability` 开关，四份军用配置已显式开启，E-off 消融象限随边一同关闭 | `engagement_graph.engagement_edges`、`config.overload_reachability`、`workflows.compare` | 论文 "The interactive graph dynamically reconstructs itself and imposes hard blocking on links deemed unreachable"；默认 False 以兼容旧 checkpoint |
| 式(36) 分组边子集 𝟙(i→j∈E_k(i))（G1） | 三组教师各自在**自己的边子集**上归一化：H_E→temporal、H_R→spatial、H_T→countermeasure（决策 D8）；E_k 为空的行回退到无限制掩码；`teacher_edge_subsets` 开关，IV-B 两份配置已开启 | `objective._edge_subsets`、`objective.attention_terms`、`config.teacher_edge_subsets` | 式(36) 分母的指示函数；Definition 1 "… using its corresponding teacher score and its corresponding edge subset" |
| ϕedg 的 V_L 及其时间变化（G2） | 边描述符新增 2 维：V_L = −(v_T−v_M)·e_L（在目标–防御机配对上求值）与 V̇_L（实体块内后向差分）；`ENGAGEMENT_EDGE_DIM` 19 → 21 | `engagement_graph.engagement_edges` | Section III-B ϕedg："relative separation, LOS-relative speed and its temporal variation, LOS direction and its temporal variation, and a visibility indicator"；式(2) V_L = V_M e_M·e_L − V_T e_T·e_L |
| 意图阶段不启用图教师（G4） | 两阶段训练的意图阶段显式关闭 `military_teachers` / `teacher_edge_subsets`（外推阶段保持开启）；修复前 `pipeline` 在 IV-B 配置下直接抛错无法训练 | `workflows.classifier_config` | 教师监督属 III-C 的 GAT 注意力层（Definition 1 的头组划分）；IV-A 意图识别为 BiLSTM + CE（式 12–14），不使用图教师 |

## 复现边界与证据

v0.4已实现通用双实体历史图和外部测量分支；原文专用对抗边、专用因素和原始场景数据没有实现，原始1000次/类的场景实验和结果表格也没有重现。可核对的是通用网络、分组监督、优化、预测误差公式，以及民用历史/交互图的边特征、掩码、裁剪与工程流程。

民用消融 `graph_no_intention` 和 `graph_no_attention_supervision` 分别关闭条件后验/context及三个民用教师，`graph_no_both`同时关闭二者。教师关闭时包括KL、熵、抖动、分离全部关闭，但民用历史图和裁剪保留。这些消融不等同于原文特定交互因素的消融。比较脚本默认两阶段，分类器每个初始化种子只训练一次；`--graph-training joint`保留历史联合模式。

同一原始轨迹的重叠窗口、同一交互场景的不同参与者均有关联，95%区间按整独立场景重采样；无耦合实体的数据按整轨迹重采样。跨初始化种子的标准差单独报告。`study`另生成多个独立民用样本库，分别交叉多个初始化种子，报告库间SD和层次bootstrap区间；这不是论文原始场景的Monte Carlo统计。所有运行保存数据SHA256、模型配置、代码SHA256、分组划分、normalizer与选模规则。详见VALIDATION.md和COMPLETION_AUDIT.md。
