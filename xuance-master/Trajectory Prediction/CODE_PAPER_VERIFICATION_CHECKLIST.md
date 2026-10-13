# 代码 ↔ 论文核验清单（2026-10-11 第五轮全面核验）

**论文**：Yu 等 2026 IEEE TAES《Trajectory Prediction for Maneuverable Target Based on Intention Recognition and Intercept Engagement》
**核验范围**：`xuance-master/Trajectory Prediction` 全部军用模块 + 训练/数据/推理链路 + 服务器脚本
**测试状态**：100 passed（军用 34 + 民用回归 + 保真回归 8 条）
**端到端冒烟**：IV-A / IV-B 全链路（生成→训练→评估→滚动→predict→Fig.14→蒙特卡洛）本机 CPU 全部跑通

图例：✅ 一致 · ✅* 一致（含工程决策，论文未明确定义）· ⚠️ 需注意 · ❌ 不一致（已修或待修）

---

## 一、模型结构核验（论文 §III）

| # | 论文设计 | 代码位置 | 是否对应 | 备注 |
|---|---|---|---|---|
| 1 | 式(4) `a_n=‖V_T×a_T‖/‖V_T‖` 法向过载 | `engagement_scenarios.normal_overload` | ✅ | 十一档符号约定见 §四 U-8 |
| 2 | 式(5)–(8) 目标/防御机运动学（点质量、过载饱和） | `simulate_engagements`（梯形积分、速度恒模） | ✅ | 速度恒模=协同转弯假设，论文一致 |
| 3 | Table IV 初始几何采样（r0∈[4,8]km、qφ∈±180°、qθ∈±30°、dh0∈±1km、VT∈[250,350]、VM∈[380,450]） | `simulate_engagements` A1 段 + `table_iv` 元数据 | ✅ | VM=360 与 Table IV 冲突为论文自身矛盾（U-7），Fig.14 场景单独用 360 |
| 4 | 式(9)(10) LOS 状态 `[r, ṙ, e_L, ė_L, V_L]` | `engagement_scenarios.los_state` | ✅ | |
| 5 | 逃逸触发律 `r ≤ (2/3)r0`（IV-B） | `evasion_fraction=2/3` | ✅ | |
| 6 | 反 LOS 逃逸指令（−e_L 投影到速度法面，饱和 ā_eva） | `_evade_command` | ✅ | |
| 7 | 追踪法（防御机 pure-pursuit，15 g 饱和） | `_defender_command` | ✅* | 论文未给出防御机制导律细节，pure-pursuit 为合理工程决策 |
| 8 | 式(12)–(14) 意图编码器：BiLSTM(Nh=128)+LN+Dropout(0.2)+中间层 m=64+ReLU+分类头 Softmax；`z=[h_t;h_s]` | `model.IntentionEncoder`（lstm_hidden=128, projection_dim=64, dropout=0.2） | ✅ | 式(14) 拼接 z=[h_t;h_s] 已在 A-06 修复后对齐 |
| 9 | 式(13) Q_int=s_M（防御机状态进意图查询） | `static_features=[s_M(6)+s_L(9)]` | ✅ | A2 修复项 |
| 10 | 式(15) 节点特征 19 维边=7 几何+9 交战+3 类别 one-hot | `engagement_graph.engagement_edges` | ✅ | |
| 11 | 式(16) `h^(0)=[g;[h_t;h_s;p̂]]` | `model.node_embedding`（6+2m+C+9 维输入） | ✅ | 意图上下文随 A-06 翻倍为 2m |
| 12 | 式(17)–(22) GAT：空间层 d=256、H=6、dk=dv=64、层数 2 | `GraphAttentionNetwork`（graph_dim=256, heads=6, graph_head_dim=64, graph_layers=2） | ✅ | 式(21) 拼接打分与代码加法式数学等价（A-08 撤销） |
| 13 | 时间层 H=4、dk=dv=64、层数 2 | temporal_heads=4, temporal_layers=2 | ✅ | |
| 14 | 式(23)–(25) 候选裁剪（Cenv/Cene/CF 指示，软惩罚） | `candidate_pruning.candidate_branch` | ✅ | Cene 判据已按式(24) 修正为 v·θ²/T²（A-05） |
| 15 | 式(26)–(29) 自回归 GRU 解码 + 线性回归输出头 | `decoder=nn.Sequential(Linear(graph_dim,3))` + GRU | ✅ | A-13 修复：线性头为默认 |
| 16 | FOV 掩码（θ_FOV=60°，交互时启用） | `fov_visible` + `engagement_graph` 掩码 + `fov_attention_mask=true` | ✅ | |
| 17 | 渐进裁剪（训练迭代中强度递增） | `learner.crop_progress=(it+1)/(steps×warmup)` | ✅ | |

## 二、损失函数核验（论文 §III-E）

| # | 论文公式 | 代码位置 | 是否对应 | 备注 |
|---|---|---|---|---|
| 18 | 式(36) 教师 `π=exp(s/τ)/Σexp(s/τ)`，τ=0.1 | `military_teachers` + `objective` 除 τ 一次 | ✅ | A-02 修复（曾被双重除 τ） |
| 19 | 式(37) `L_attn_k` 对**时间维求和** | `objective._time_sum` | ✅ | A-12 修复；量级放大由 uncertainty σ̃_k 吸收 |
| 20 | 式(38)–(40) 熵/抖动正则（组内头均值 β̄） | `teacher_regularizers="paper"` 路径 | ✅ | B5 修复 |
| 21 | 式(41) 分离正则 cos²（跨组头对） | `objective` separation 项 | ✅* | 保留均值口径（严格三重求和会放大两个数量级，已登记 P2 重标定项） |
| 22 | 式(47)(48) E/R/T 三教师（能量/风险/走廊） | `military_teachers`（_pairwise_energy/_risk_score/_corridor_score） | ✅ | R 教师 LSE 窗口加权已按式(49) 修复（A-03） |
| 23 | 式(49) `s^(R)=(1/λ)log Σ w·exp(λr)` 窗口 LSE | `_risk_score` 重写 | ✅ | |
| 24 | 式(50)(51) 走廊偏差 δ=(dist−b)/b | `corridor_deviation` | ✅ | A-04 修复（半宽 0.2·V·T_p） |
| 25 | 式(52)(53) 几何项（任务目标点） | `objective` ADE/FDE | ✅* | Y 用末状态恒速外推点（U-3 决策） |
| 26 | 式(54) `L_attn=Σ_k(1/(2σ̃²)L_k+log σ̃_k)+λ_sep L_sep` | uncertainty 加权分支 | ✅ | log 正则仅作用于三组教师 |
| 27 | 式(55) `L_tra=L_attn+1/(2σ̄²_sup)(ω₁ADE+ω₂FDE)`，无 log 正则 | `log_variance[3]` 分支 | ✅ | A-01 修复（几何项带权、无 +log） |
| 28 | λ_sep=0.1, ω₁=0.4, ω₂=0.6 | `separation_weight=0.1, ade_weight=0.4, fde_weight=0.6` | ✅ | 配置断言 |
| 29 | log-var clamp [−6,6] | `learner.log_variance.clamp_(-6,6)` | ✅ | |

## 三、训练协议核验（论文 IV-A/IV-B）

| # | 论文协议 | 代码位置 | 是否对应 | 备注 |
|---|---|---|---|---|
| 30 | IV-A：Adam 1e-3、γ=0.5@30、batch 256、70 ep | `engagement_intention.json`/`engagement_overload.json` | ✅ | |
| 31 | IV-B：AdamW 1e-3、wd 1e-4、warmup 5+cosine、batch 256、200 ep、梯度裁剪 1.0 | `engagement_forecast.json` | ✅ | |
| 32 | IV-B 早停：ADE&FDE 20 轮无 ≥1 m 改善即停 | `early_stopping_mode="both", patience=20, min_delta_m=1.0` | ✅ | |
| 33 | IV-A 协议：dt=0.5s、To=10s(O=20)、stride 5、fov_coverage=0.5 | `run_engagement.sh` intention/overload 分支 | ✅ | |
| 34 | IV-B 协议：dt=0.1s、To=8s(O=80)、Tp=6s(P=60)、fov_coverage=1.0 | `run_engagement.sh` forecast 分支 | ✅ | |
| 35 | 数据规模 5×10⁴ 窗口（IV-A/IV-B） | `TRAJECTORY_TARGET_WINDOWS=50000` | ✅ | 脚本自动扩 scenes；本机未跑满，待服务器 |
| 36 | 标准化仅用训练集 | `fit_normalizer(datasets["train"])` | ✅ | |
| 37 | 场景级划分防泄漏 | `split_tracks` + `validate_split` | ✅ | |
| 38 | 每类 1000 次蒙特卡洛（IV-A） | `scripts/monte_carlo_intention.py` | ✅ | 本机 50 次验证链路，1000 待服务器 |
| 39 | 基线：LSTM/GRU 2 层 256、TCN 4 块 D={1,2,4,8} k=3 c=256、Transformer 2 层 d=256 h=4 ffn=1024、dropout 0.1 | `baselines.py` | ✅ | |
| 40 | 意图分类指标：准确率+宏F1+逐类 P/R/F1+多数类基线 | `metrics.classification_metrics` | ✅ | N4 修复 |

## 四、论文未明确事项的工程决策（U 类，仅凭论文正文可决断）

| ID | 事项 | 决策 | 风险 |
|---|---|---|---|
| U-1 | 式(21) 拼接打分 vs Fig.6 q/k/v 文字冲突 | 加法式（数学等价，A-08 撤销） | 无 |
| U-2 | Definition 1 的 E_k(i) 边子集未定义 | 三组教师共用全边集 | 低（教师只做监督，不改前向） |
| U-3 | 任务目标点 Y/T*、走廊半宽 b | Y=末状态恒速外推点；b=0.2·V·T_p | 中（影响走廊教师绝对位置，量级已校） |
| U-4 | 式(47) unitization 基准 | a_T,max=9g / a_max/dt | 低 |
| U-5 | 式(49) 窗口 H 与权重 w | H=预测时域、w=exp(−Δt/T_p) | 低 |
| U-6 | 地形/禁飞栅格 F 无数据 | terrain_enabled=false（Cenv 恒 1） | 中（式23 环境约束退化，需在论文报告口径中说明） |
| U-7 | Table IV VM∈[380,450] vs 正文 360 | 训练数据用 [380,450]，Fig.14 用 360 | 无（论文自身矛盾，分开配置） |
| U-8 | 十一档符号（式4 非负 vs 档位带符号） | 速度系主分量符号（左转/爬升为正） | 低（已声明约定） |
| U-9 | dropout/初始化/加载细节未给出 | PyTorch 默认初始化 + dropout 0.1 | 低 |
| U-10 | Table VI a_φ 符号与常规右手系相反 | 按场景名（Left/Right/Climb/Dive）定符号 | 低（镜像对称不改变误差范数） |

## 五、可运行性核验（本机 CPU 实测）

| # | 检查项 | 结果 | 备注 |
|---|---|---|---|
| R1 | 测试套件 | ✅ 100 passed | 含 8 条保真回归 |
| R2 | `check --backward`（xuance/standalone 双后端 × 三套军用配置） | ✅ | finite_loss 有限、backward 通过 |
| R3 | IV-A 端到端：simulate-engagement→train(2ep)→evaluate→monte_carlo | ✅ | 9 类/11 档全在场；2ep 准确率 0.148≈随机基线（预期） |
| R4 | IV-B 端到端：simulate-engagement→train(2ep)→evaluate | ✅ | ADE 403 m≈CV 基线（2ep 未收敛，预期） |
| R5 | rolling-evaluate（IV-B checkpoint，refresh=5） | ✅（已修） | **发现并修复**：holdout_test=false 时无 test 分区导致崩溃，已加 validation 回退（与 engine.evaluate 一致） |
| R6 | predict 命令（IV-B checkpoint） | ✅（已修） | **发现并修复**：CLI 漏传 engagement 描述符（静默零填充），已补 |
| R7 | reproduce_figure14（CV 基线） | ✅ | 四场景终端误差相同（1183 m）为镜像对称的正确结果；需训练后模型对照 150/202/81/173 m |
| R8 | check_paper_targets / monte_carlo_intention 脚本入口 | ✅ | 参数与输出正常 |
| R9 | requirements.txt 与实际 import 一致性 | ✅ | numpy/scipy/gymnasium/PyYAML/cloudpickle/tqdm/opencv/matplotlib/torch |
| R10 | bash/run_engagement.sh 四协议（intention/overload/forecast/ablation） | ✅ | O/P/stride/fov_coverage/协议参数全部接线正确 |

## 六、本轮发现并修复的问题（第五轮）

| ID | 问题 | 位置 | 修复 |
|---|---|---|---|
| V-1 | rolling-evaluate 在 holdout_test=false（IV-B）下默认 split=test 直接崩溃 `No complete windows in partition` | `rolling.py:84` | 加 `if not saved["splits"].get(split): split="validation"` 回退，镜像 engine.evaluate 行为；修复后 rolling ADE 403.544 与窗口评估完全一致 |
| V-2 | CLI predict 命令未传 engagement 描述符给 Predictor | `cli.py` | 补 `track.engagement if track.engagement is not None and cfg.engagement_edges`；交战几何不再静默丢失 |

## 七、上服务器前的准备事项

1. **数据规模**：`TRAJECTORY_TARGET_WINDOWS=50000 bash bash/run_engagement.sh outputs/military cuda:0 forecast`（intention/overload/forecast/ablation 四协议）。
2. **蒙特卡洛**：训练完成后 `python scripts/monte_carlo_intention.py --data <npz> --checkpoint <best.pt> --trials 1000`。
3. **Fig.14 复现**：`python scripts/reproduce_figure14.py --output outputs/figure14 --checkpoint <forecast/best.pt>`（对照 150/202/81/173 m）。
4. **锚点核对**：`python scripts/check_paper_targets.py --intention <mc.json> --overload <mc.json> --forecast <metrics.json> --figure14 <summary.json>`（88% / 94% / 102 m / 四场景）。
5. **环境**：`pip install -r requirements.txt`（torch CPU 版本本机用 `--extra-index-url https://download.pytorch.org/whl/cpu`；服务器 CUDA 版直接主索引）；Python ≥3.10，已实测 3.12/3.13。
6. **打包**：`python scripts/prepare_server.py --output server.tar.gz`（含 manifest SHA256）。
7. **回归保护**：服务器跑完后 `PYTHONPATH="src;.." python -m pytest tests -q` 应保持 100 passed。

## 八、遗留事项（不影响上服务器）

- **P2 重标定项**：式(41) 分离项严格求和口径（当前均值）+ `separation_weight` 同步重标定；A3/E1/D7/B4/B6/F6 开关化敏感性扫描（论文未给数值，属可选）。
- **U-6 地形约束**：式(23) Cenv 目前恒 1（论文无地形数据）；如需完整复现需自造地形栅格，建议在报告口径中说明。
- **Fig.14 需训练后模型**：CV 基线四场景误差相同（镜像对称），论文数值 150/202/81/173 m 必须用训练后的 IV-B checkpoint 才能对照。

## 九、第六轮迭代（2026-10-11 晚）：三项 P2 差距落地

本轮把上轮清单登记的三个有明确论文依据的 P2 项实现完毕，测试 100 → **103 passed**：

| ID | 修改内容 | 论文依据 | 位置 |
|---|---|---|---|
| **C6** | 逃逸强度有界采样：触发逃逸后 ā_eva 每 ~4 s 段从 `evasion_overload_range`（默认 U(0.5,1.0)×aT_max）重采样、段内保持、按 min(ā_eva, aT_max) 饱和；CLI `--evasion-overload-range` | IV-B："Segment duration and the nominal overload level ā_eva are sampled within predefined bounds … holding a_eva within each short primitive segment and updating it at segment transitions" | `engagement_scenarios.simulate_engagements`、`cli.py` |
| **C9** | 交战终止：拦截（r ≤ `intercept_radius_m`=50 m）或确定错过（逃逸后分离连续增长 `miss_grow_steps`=30 内部步）即终止，尾部状态冻结（防拦截后穿越伪影），diagnostics 记录 terminated_step/termination_reason 与 intercepts/misses/horizon_completes 统计；CLI `--intercept-radius-m/--miss-grow-steps` | 论文样本为 "a complete engagement trajectory segment"；仿真协议应为拦截交战（至拦截/错过为止） | `engagement_scenarios.simulate_engagements` |
| **A7** | 不可达链路过载硬阻断：跨实体边要求 a_req=|v_rel|·θ/lag ≤ aT_max（9g），否则删边；同实体时序边、lag=0 空间边与自环不受影响；`overload_reachability` 配置开关（默认 False 保持旧 checkpoint 兼容，四份军用配置显式 true；workflows 的 E-off 消融象限联动关闭） | IV-B："The interactive graph dynamically reconstructs itself and imposes hard blocking on links deemed unreachable" | `engagement_graph.engagement_edges`、`config.py`、`workflows.py` |

新增回归测试：`test_evasion_overload_is_resampled_per_segment`（C6）、`test_intercept_termination_fills_frozen_tail`（C9）、`test_overload_reachability_blocks_unreachable_cross_entity_links`（A7）。

实测证据：12 场景冒烟 = 2 拦截 + 3 错过 + 7 跑满，终止原因与最小间距自洽（拦截场景 min_sep=16 m）；A7 交叉几何下 60 → 12 条跨实体边（48 条不可达边被硬阻断，幸存者恰为 lag=0 空间边）；IV-B 配置（A7 开启）1 epoch 训练正常。

**注意**：本轮改动使数据生成器的输出分布发生变化（逃逸强度多样 + 终止冻结），上服务器前应**重新生成数据集**；旧数据集仍可训练（C6/C9 只影响生成端）。

## 十、第七轮迭代（2026-10-11 深夜）：公式级缺口补齐 + 一处真实缺陷

目标：只处理"论文里有、但代码没做或做错了"的项。测试 103 → **107 passed**，IV-B 端到端 pipeline（意图 1 epoch + 外推 2 epoch + 评估）实测通过。

### 10.1 已修复（有明确论文依据）

| ID | 问题（论文 → 代码） | 修改 | 论文依据 | 位置 |
|---|---|---|---|---|
| **G1** | 式(36) 教师分布 π^(k) 的分母含指示函数 **𝟙(i→j ∈ E_k(i))**，即每组只在**自己的边子集**上归一化；代码对所有三组共用同一个"学生注意力 > 0"掩码，完全没有 E_k 概念 | 新增 `_edge_subsets()` 按 Section III-B 重建三类边（temporal / spatial / countermeasure），组内 teacher 支持集 = E_k ∩ 学生掩码；E_k 为空的行回退到无限制掩码（避免 π 未定义）；开关 `teacher_edge_subsets`（默认 False，IV-B 两份配置显式 true） | 式(36) + Definition 1："supervision and normalization are applied to the attention weights produced by each group using its corresponding teacher score and its **corresponding edge subset**" | `objective._edge_subsets`、`objective.attention_terms`、`config.py` |
| **G2** | ϕedg 的交战描述符列出 **"LOS-relative speed and its temporal variation"**；代码只有 LOS 速率 rdot，**没有其时间变化**，且未使用论文符号的 V_L | 边描述符新增 2 维：V_L（按式(2) 在目标–防御机配对上求值，V_L = −(v_T−v_M)·e_L）与其时间变化 V̇_L（实体块内后向差分 /dt）；`ENGAGEMENT_EDGE_DIM` 19 → 21 | Section III-B："The edge feature mapping ϕedg assigns each edge an engagement descriptor composed of relative separation, **LOS-relative speed and its temporal variation**, LOS direction and its temporal variation, and a visibility indicator"；式(2) V_L = V_M e_M·e_L − V_T e_T·e_L | `engagement_graph.engagement_edges` |
| **G3** | 生成器注释写描述符为 10 维 `[r, rdot, eL, eLdot, V_L, visible]`，实际只写 9 维（V_L 被 `los_state` 算出后丢弃），注释与实现不一致 | 注释改为与实际一致的 9 维，并说明 V_L = −rdot 故未重复存储（边描述符已携带） | 式(2) | `engagement_scenarios.ENGAGEMENT_DIM` |
| **G4** | **真实缺陷**：`classifier_config`（IV-A 意图阶段）关闭 `attention_supervision` 却继承了 `military_teachers=True`，`validate` 直接抛错 → **IV-B 的 `pipeline` 两阶段训练根本跑不起来** | 意图阶段显式关闭 `military_teachers` 与 `teacher_edge_subsets`（外推阶段不受影响） | 教师监督属 III-C 的 **GAT 注意力层**（Definition 1 的头组划分针对 H=6 的图注意力）；IV-A 意图识别是 BiLSTM + CE（式 12–14），不使用图教师 | `workflows.classifier_config` |

新增回归测试：`test_teacher_distributions_use_their_own_edge_subset`、`test_edge_subsets_are_opt_in_and_preserve_the_unrestricted_mask`（G1）、`test_edge_descriptor_carries_los_relative_speed_and_its_rate`（G2，含既有索引断言后移 2 位）、`test_intention_stage_drops_engagement_teacher_supervision`（G4）。

### 10.2 已查证为一致（无需修改）

- 式(47) `s_E = −(1/κ_E) log Σ exp(κ_E·x̃²)`：代码 `-logsumexp(κ·indicators)/κ`，三项均为归一化平方（ã_req、j̃_t、c̃_r）——一致。
- 式(48)(49) 风险教师：`R = LSE_κR(r̃, ṙ̃, P̃_T,t)` 后 `(1/λ_R) log Σ_τ w exp(λ_R r)` 窗口聚合——与 `_threat_levels` / `_risk_score` 一致（w 用指数衰减为工程选择，已登记 B6）。
- 式(50)–(53) 时效教师与 μ_t 门控、式(54)(55) 总损失结构（三组教师 KL + λ_sep L_sep + 不确定度加权 ω₁ADE+ω₂FDE）——一致；式(55) 无独立的输出轨迹能量正则，故式(42)–(44) 的 V/a/j 仅用于构造教师分数。
- 式(19)(20) f_e/f_n 门控输入含交战描述符 u：f_e 经 `edge_context`（交战边特征 φ_edg 含 u 分量）读 u；f_n 经 `node_context`（node_u = 每步交战描述符，候选块取终端值）显式读 u——两路均与论文 "take the node hidden states together with u as inputs" 逐字对应。
- 逃逸指令 `ãeva = −e_L(I − e_T e_Tᵀ)`、饱和 `a_eva = min{ā_eva, a_T,max}`：`_evade_command` 逐字一致；逃逸触发 `r ≤ (2/3) r₀` 一致。
- 防御机导律：论文 Table IV 只列目标机动过载参数，**未给防御机制导律**，pure-pursuit 维持工程决策（清单第二节第 7 项 ✅*）。

### 10.3 登记为"论文有但当前架构无法逐式实现 / 论文自身未指定"

| ID | 事项 | 现状与判断 |
|---|---|---|
| **X1** | **教师分数的定义域**：式(42)–(44) 明确说 V/a/j 来自"式(32) 外推结果的增量序列"，式(45) 是"node Δy_i → neighboring node Δy_j"的转移，Remark 3 说 "nodes correspond to candidate future states" | 当前按 D1 决策（2t_o 历史节点 + 锚定候选块）把教师算在**观测历史**域。这是 A5（论文自身在"式15 的 2t_o 历史节点"与"Table I/Remark 3 的候选未来状态"之间矛盾）的下游后果；要逐式复现必须整体切换到候选图架构，本轮不做，维持 A5 登记并在报告中单列 |
| **X2** | ~~式(26)–(31) 在正文中不存在~~ **勘误（第八轮推翻）**：式(26)–(31) 实际存在于 **Table I 算法框内**（式26 初始兼容性分数 q_i^(0)、式27 Cesc 窗口平均、式28 Cenv forall-k、式29 Cene、式30 m(j)=Cesc·Cenv·Cene、式31 软裁剪 α←m·exp(−Λ_crop·P)/S_i）。先前 grep 正文文本流未见，是因为 PDF 文本层丢失了算法框内公式；MILITARY_PORT_PLAN.md §2 Table I OCR 恢复可证。Cesc 窗口平均与 Cenv 全弧点检查已按式(27)(28) 实现（`candidate_arc_samples`）；式(31) 软裁剪由 `candidate_crop_lambda`+crop_progress 承担 | 已按 Table I 实现；式(26) 初始分数仍按式(21) 加性注意力实现（OCR 噪声，见 §6 风险2） |
| **X3** | 式(45) `a_req = V/Δt · arccos(Δy_j·Δy_i / \|Δy_i\|)` 分母只归一化了 Δy_i，公式本身不对称 | 论文疑为笔误；代码用对称的速度夹角版本（量纲正确），维持现状并登记 |
| **X4** | 式(37) 的 ρ_k、ζ_k 与式(36) 的 τ_k 带分组下标 k | 当前三组共享同一 `entropy_weight` / `jitter_weight` / `teacher_temperature`；论文未给数值，合并进 B4/B6 敏感性扫描 |
| **X5** | 组 → 边子集映射 | 论文只说"corresponding edge subset"，未列表；本轮定为 H_E→temporal、H_R→spatial、H_T→countermeasure（**决策 D8**），建议消融验证 |
| **X6** | 式(52)(53) 的走廊中心路径 Y*_k、任务目标点 Y 与时间要求 T*；式(48) 的可达域 P_T,t 公式 | 论文均未给数值/公式；当前分别用"窗口锚点+航向外推线"与"v·T_p + 0.5·a_max·T_p²"近似，μ_t 用 σ(κ(2·progress−1)) 近似 (t−T*)/T，维持登记 |

### 10.4 上服务器前提醒

- G2 改变了边特征维度（19 → 21），**旧 checkpoint 不兼容**，须与 C6/C9 的新数据集一起重新训练。
- G1（默认在 IV-B 配置开启）会改变三组教师的归一化口径，属于对论文的收紧；如需对照旧行为，可用 `teacher_edge_subsets=false` 跑一组消融。

## 十一、服务器代码套整合（2026-10-11 末）

### 11.1 新增与修改

| 文件 | 说明 |
|---|---|
| `SERVER_RUNBOOK.md` | **训练方法主文档**：环境准备、四协议训练命令、数据集参数表、论文数值复现、耗时估算、故障排查、对照入口 |
| `bash/setup_server.sh` | 新建 venv → 装 torch（cpu/cu121/cu124）+ requirements → 跑 `check --backward` 自检 |
| `bash/run_paper_targets.sh` | 训练后跑论文数值：IV-A 蒙特卡洛（1000 次/标签）+ Fig.14 + `check_paper_targets.py` 锚点对照 |
| `bash/run_engagement.sh` | ① 输出改到 `run/<protocol>/`（与 run_paper_targets 路径约定一致）；② 新增 `TRAJECTORY_EPOCHS` 覆盖便于小规模自检；③ 解释器改 `PYTHON` 变量可覆盖 |
| `scripts/prepare_server.py` | 打包清单补三个军用脚本（原来只导出民用脚本，服务器上会没有军用入口）；排除 `outputs/`（原来会把训练产物打进包） |
| `scripts/reproduce_figure14.py` | **真实缺陷修复**：`Track.context` 是 `[T,N,6]`、`Predictor.predict` 要 `[N,T,6]`，脚本少一次转置 → **带 checkpoint 复现 Fig.14 必崩**（`Peer history must be [configured N,at least O,6]`）。无 checkpoint 时走恒速分支不触发，故此前未暴露 |

### 11.2 本机 CPU 冒烟（等价链路已在小数据上跑通）

```
TRAJECTORY_EPOCHS=1 TRAJECTORY_TARGET_WINDOWS=300 TRAJECTORY_ENGAGEMENT_STEPS=150 \
  bash bash/run_engagement.sh outputs/serverkit_smoke cpu intention   # → run/intention/{best.pt,test_metrics.json}
TRAJECTORY_EPOCHS=1 TRAJECTORY_TARGET_WINDOWS=200 TRAJECTORY_ENGAGEMENT_STEPS=150 \
  bash bash/run_engagement.sh outputs/serverkit_smoke cpu forecast    # → run/forecast/{best.pt,validation_metrics.json}
TRAJECTORY_MC_TRIALS=10 bash bash/run_paper_targets.sh outputs/serverkit_smoke cpu
  # 蒙特卡洛 → Fig.14 → 锚点报告（0/7 达标：1 epoch + 200 窗口的冒烟结果，非缺陷）
```

导出：`server_package.tar.gz`（938 文件，901 KB，sha256 `4d6b4aba…`），只含源码与配置，不含数据/环境/日志。
