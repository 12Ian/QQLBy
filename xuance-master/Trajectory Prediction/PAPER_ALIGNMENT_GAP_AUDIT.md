# 当前实现 ↔ 原论文仿真实验：逐条差异审计

审计日期：2026-10-11 · 对象：`src/trajectory_prediction`（S1–S8 完成后的军用版）+ `configs/engagement_*.json` + `bash/run_engagement.sh`
依据：论文全文 `tmp/paper_text.txt`（IV-A：行 1253–1400；IV-B：行 1440–1800；教师：行 940–1180），以及 OCR 恢复的 Table I/III/IV/V/VI（见 `MILITARY_PORT_PLAN.md`）
约定：**P0 = 不修就无法得到论文口径的数值；P1 = 影响数值可比性或论文要求的实验；P2 = 登记性差异/论文未给出唯一实现**

行号为审计时状态，可能随后续修改漂移；以函数/配置项名称为准。

---

## 0. 论文报告的对照目标值（复现验收锚点）

| 来源 | 论文数值 | 当前代码是否产出 |
|---|---|---|
| IV-A 九类意图，测试准确率 | ≈88%，混淆矩阵强对角占优 | 产出准确率与混淆矩阵，但**数据里缺"无机动"类**（C1），宏口径不可比 |
| IV-A 十一档过载，测试准确率 | ≈94%，行归一化混淆矩阵对角带 | 产出，但**负档永不使用**（C2），只有 6 档有效 |
| IV-B ADE/FDE | 收敛到 ≈102 m 量级 | 产出，但训练规模/损失项/早停阈值不同（B2/B8/C3） |
| Fig.14 四复合场景终端误差 | 150 / 202 / 81 / 173 m | **无复现脚本**（G4） |
| Fig.16/18 对比与消融 e(t)、ADE/FDE | 基线与 A1/A2/A4 单调变差 | 工具具备（S7），未跑完整预算 |
| 每标签 1000 次独立蒙特卡洛 | 每机动标签 1000 次、不同种子 | 默认 3 个数据种子（C4） |

---

## 1. 模型与算法结构

| ID | 论文要求（位置） | 当前实现（代码位置） | 对复现的影响 | 修改方案 | 优先级 |
|---|---|---|---|---|---|
| A1 | IV-A：意图输入为 6 维位置+速度序列；BiLSTM Nh=128 → LayerNorm → Dropout .2 → FC m=64 ReLU → Dropout .2 → 9 类 Softmax；CE 训练 | 多一个静态分支 + `fusion`（64+64→64）+ `64→64→C` 分类头；军用配置一律 `static_state_branch=true`（`model.py:13–35`；`configs/engagement_intention.json`） | IV-A 是**独立的分类实验**，论文输入只有 6 维序列；额外分支与额外一层使"逐层复现"不成立，且改变决策边界 → 88% 不可直接比对 | ~~IV-A 协议配置设 `static_state_branch=false`~~ **本条审计有误，已撤销**：论文式(12)–(14) 明确给出序列分支 `h_t=ReLU(W_t·BiLSTM(x)+b_t)` **与**防御机静态分支 `h_s=ReLU(W_s·s̃_M+b_s)`，静态分支是论文原文的一部分，不应关闭。正确做法是保留静态分支并把 `s̃_M` 补全为论文式(8)(9) 的 `Q_int/Q_tra` 交战因子（见 A2，已实现）。 | ~~P1~~ |
| A2 | 式10–11、图3–4：交战因子 Q_int/Q_tra（拦截交战状态）注入意图与外推两个模块 | npz 里存了 `engagement_states`（r/ṙ/e_L/ė_L/V_L/可见性），但 `data.py load_tracks` **未读取**；静态分支只有防御机 6 维状态（`model.py:165,177–178`）；交战描述符只用于**边特征**（`engagement_graph.py`） | 论文核心卖点"engagement-aware formulation"只实现了一半：节点/意图侧没有交战状态 → 消融 E 的增益被低估，与 88%/94%/102 m 的可比性受损 | `Track` 增 `engagement_states`；静态分支改为 `[防御机6维 + 交战9维]`（或拼接 LOS 描述符）；节点嵌入同步扩展；`engagement_intention/forecast.json` 加开关 | **P0** |
| A3 | IV-B："The linear regression is employed for the output head" | 解码头为 `Linear→ReLU→Linear(3)` 且末层零初始化（`model.py:137–141`），起点=恒速基线 | 零初始化使首步 ADE/FDE 对图参数的梯度为 0（已登记为工程选择），训练早期行为与论文不同；末层多一层非线性 | 增加 `linear_output_head` 配置；论文对照实验用线性头；零初始化保留但写入 METHOD_MAPPING | P2 |
| A4 | —（论文未提） | 实体嵌入 `entity_embedding`、时间位置嵌入 `history_embedding`、`time_embedding`（`model.py:111,122,127`） | 工程附加项，参数量 +~1%，无直接误差影响 | 登记为适配项 | P2 |
| A5 | 式22–31、Table I：候选未来状态的交互分布裁剪 | D1 锚定候选法：A=32 锚点挂 O 大小候选块 + Cesc/Cenv/Cene + Λ_crop 软惩罚（`candidate_pruning.py`、`model.py:196–205,236–247`） | 论文本身在"式15 的 2t_o 历史节点"与"Table I 的候选未来状态"之间自相矛盾，无唯一标准实现；当前实现可跑通并可消融，但**不能声称逐式复现** | 保留 D1；在论文对照里把"候选裁剪"单列一项消融（E 轴子集），用配对误差差说明增益 | P1（登记） |
| A6 | IV-B："The FOV mask is enabled during interaction"；且"防御机不可见时预测器应自然退化为无交战耦合的外推" | 可见性只作为边特征 1 位 + Cesc 旁路（`engagement_graph.py:67–74`；`candidate_pruning.py` Cesc 段）；**注意力上没有 FOV 硬掩码** | 论文明确要求的"不可见→退化为纯外推"行为没有在图层实现；同时数据集侧 C5 又没造出"不可见"样本 → 该行为既无数据也无机制 | `engagement_edges` 增加 `fov_mask` 开关：目标节点不可见时屏蔽其 countermeasure 类边；配合 C5 生成 50% 不可见场景 | P1 |
| A7 | "hard blocking on links deemed unreachable" | 军事路径 `reachable=None`（`model.py:209`），只有速度可行性 `allowed` | 缺失过载可达性的硬阻断，裁剪强度弱于论文 | 交战边增加"过载可达"硬阻断（用 9g 与 Δt 判据） | P2 |
| A8 | 空间层 d=256、H=6、d_k=d_v=64；时间层 H=4、同维；两层深；Top-K=8；τ=0.1；自回归 GRU + 时间记忆 cross-attention | 全部一致（`configs/engagement_forecast.json`：`graph_dim=256, heads=6, graph_head_dim=64, graph_top_k=8, temporal_heads=4, temporal_layers=2, graph_layers=2, attention_temperature=.1`；`model.py:129–139`） | 无差异 | — | 已对齐 |

---

## 2. 损失函数与优化器

| ID | 论文要求 | 当前实现（代码位置） | 影响 | 修改方案 | 优先级 |
|---|---|---|---|---|---|
| B1 | 式55 几何项 = ω1·ADE + ω2·FDE（0.4/0.6），无平滑项 | 额外 `smoothness_weight × acceleration_mse`，默认 **0.001 且军用配置未置零**（`objective.py:26–27,132`；`config.py:30`） | 多一个正则项 → 轨迹更平滑、终端误差偏小/偏大不确定；直接改变 102 m 这个数值 | 军用配置显式 `smoothness_weight: 0`；保留民用默认 | P1 |
| B2 | 式54 只对**三个注意力组**写对数正则；式55 几何项字面无 log σ 项 | 几何也被纳入 4 任务不确定性加权，同样加 `0.5·s`（`objective.py:125–131`） | 几何权重被乘上 `0.5·exp(−s)`，等效尺度随训练变化 → **对最终 ADE/FDE 影响可达数十米量级** | 增加 `geometry_log_regularization: false`（论文口径：几何固定权重 0.4/0.6，只有三组教师用 log-variance） | **P0** |
| B3 | 解耦训练：意图用 CE；联合总损失未给 CE 权重 | joint 阶段 `intention_weight=0.2 × CE`（`objective.py:134`） | 权重未定，影响意图分支与外推分支的梯度分配 | 登记；建议联合训练沿用解耦后的冻结或极小权重，并做敏感性 | P2 |
| B4 | 教师项经式54 的不确定性加权；温度未给 | 每组先乘 `attention_weight=0.1`，再除以 `teacher_temperature=0.1`（`objective.py:77,127`；`config.py:64,69`） | 温度 0.1 把教师分布显著锐化 → KL 目标更"硬"，注意力形态与论文不同，间接影响误差 | 登记为稳定化选择；扫描 τ_T∈{0.1,0.5,1.0} | P2 |
| B5 | 式38/40 熵下限：`|H_min − H(β̄)|`，β̄ 为**组内头分布均值**；式39 抖动作用于**相邻时刻的组平均分布**；式41 分离为跨组头对 cos² 求和 | 逐头熵 + 单边惩罚 `max(H_floor−H,0)`；抖动在同一注意力张量上平移索引、未先做组平均；分离取跨组头对**均值**（`objective.py:78–100`；已有 review §4.2–4.4） | KL 目标数值不同 → 注意力形态系统性偏差 → 最终 ADE/FDE 与论文不可逐项比对 | 按式38–41 改写：先组内平均再算熵、抖动作用于组平均、分离改求和；保留现有实现为 `legacy` 开关 | P1 |
| B6 | κ_E、κ_R、λ_R、w_{j,τ} 只作定性说明，**未给数值** | `teacher_kappa=1.0`、`teacher_risk_lambda=1.0`、时间权重 `exp(−lag/horizon)`（`config.py:92–93`；`military_teachers.py:94–103`） | 无法"对齐"，只能做敏感性 | 登记 + 扫描 κ∈{0.5,1,2}、λ_R∈{0.5,1,2} | P2 |
| B7 | IV-B：AdamW 1e-3、wd 1e-4、clip 1.0、batch 256、200 轮、warm-up 5 + cosine、80/20、最佳验证检查点 | 全部一致（`learner.py:14–30`；`configs/engagement_forecast.json`） | 无差异 | — | 已对齐 |
| B8 | 早停：验证 ADE 与 FDE **连续 20 轮改善不足 1 m** 即终止 | `early_stopping_min_delta_m=0.001`（**1 毫米**，沿用民用低速尺度）（`configs/engagement_*.json`、`config.py:61`；`engine.py:279–282`） | 军用误差在百米量级，1 mm 阈值≈"任意改善即重置" → 实际训练轮数与选模点完全不同，直接影响报告的 102 m | 军用配置设 `early_stopping_min_delta_m: 1.0` | **P0** |

---

## 3. 仿真环境与数据集参数

| ID | 论文要求 | 当前实现 | 影响 | 修改方案 | 优先级 |
|---|---|---|---|---|---|
| C1 | 九类意图空间含"no maneuver"；"maneuver classes are sampled with **balanced coverage**" | 基元过载 `U(0.35,0.95)×9g = 31–84 m/s²`，而直行判定阈值 10 m/s²：需要 yaw、pitch 分量**同时**<10，在总量 ≥31 时数学上不可能 → **类别 0 从不出现**；其余八类由随机 share 决定，非平衡（`engagement_scenarios.py:89–97,118–128`） | 直接决定 IV-A 能否复现：混淆矩阵少一类、宏 F1 上限被压、88% 不可比 | 改为**按语义类别平衡抽样**：先抽 9 类之一（含一定比例低过载直飞段），再在类内抽过载/时长；加"每类窗口数"断言并写入 metadata | **P0** |
| C2 | 十一档 I={−10,…,0,…,+10} g，统计量 ā_n 为窗口**均值** | `normal_overload = ‖v×a‖/‖v‖` ≥ 0（式4 定义如此），因此标签恒 ≥0 → **负六档永不使用**（`engagement_scenarios.py:131–143`） | 11 类实验实际只有 6 档 → 混淆矩阵只剩右半，94% 不可比（论文自身式4 与带符号等级也矛盾，见 §8） | 明确符号约定：取法向过载在"水平/垂直基"上的带符号分量（如按 pitch 指令符号）形成 −10…+10；或显式声明只复现非负 6 档并说明差异 | **P0** |
| C3 | IV-A ≈5×10⁴ 窗口；IV-B 50 000 实例 | `run_engagement.sh` 默认 `scenes=60, steps=400` → dt=.1、O+P=140、stride 5 时约 3×10³ 窗口（`bash/run_engagement.sh:44–48`；`data.py Windows`） | 少一个数量级以上 → 欠拟合、数值与 102 m 不可比 | 脚本默认改为窗口数达标（如 scenes=400–600 / steps=400，或按目标窗口数自动求解），并断言"窗口数 ≥ 5×10⁴" | P1 |
| C4 | 每个机动标签 **1000 次独立蒙特卡洛、不同随机种子**构成测试集 | `study` 默认 3 个数据种子 × 3 个初始化种子（`cli.py:85–86`） | 无法给出论文口径的稳定性结论与区间 | 增加 `--data-seeds` 规模；或单列"每类 1000 试验"脚本（可先做 100 次并标注） | P1（受算力约束） |
| C5 | IV-A 库：防御机初始**既在 FOV 内也在 FOV 外**；IV-B 库：初始化**在 FOV 内** | `initial_fov_coverage=1.0`（恒在内部），CLI 未暴露该参数（`engagement_scenarios.py:174,216–221`；`cli.py` simulate-engagement 段） | IV-A 数据缺"不可见"样本 → A6 的退化行为无数据、论文要求的覆盖不全 | CLI 暴露 `--initial-fov-coverage`；IV-A 用 0.5，IV-B 用 1.0 | P1 |
| C6 | "segment duration and the nominal overload ā_eva are sampled within predefined bounds" | `a_bar_eva` 默认 = a_T,max（恒饱和），未采样（`engagement_scenarios.py:187–188`） | 逃逸机动强度单一 → 行为多样性不足，影响泛化与误差分布 | 增加 `a_bar_eva` 采样区间参数（如 U(0.5,1.0)×a_T,max） | P2 |
| C7 | IV-B 演示场景：VT=300、VM=360 m/s，a_T,max=9g、a_M,max=15g、FOV=60°；Table VI 四场景初始条件（aϑ=±39、aϕ=∓58、r0=5.3 km、eL0=[0.98,±0.15,±0.09]） | 生成器支持 `vT_fixed/vM_fixed`（`engagement_scenarios.py:172,203–204`），但**配置与脚本未用**；Table VI 四场景无复现入口 | 无法复现 Fig.14 的 150/202/81/173 m | 新增 `scripts/reproduce_figure14.py`：固定 300/360 + Table VI 初始条件，输出四场景终端误差对照表 | P1 |
| C8 | Table IV 采样：r0∈[4,8] km、qϕ∈±180°、qϑ∈±30°、Δh0∈±1 km、VT∈[250,350]、VM∈[380,450]、aϕ0/aϑ0∈±88.3 m/s² | 与采样器默认值一致（`engagement_scenarios.py:167–174`） | 无差异（注：正文 360 m/s 与 Table IV 的 [380,450] 是论文自身不一致，见 §8） | — | 已对齐 |
| C9 | 拦截交战仿真（至拦截为止） | 固定步数传播，无拦截终止条件 | 样本包含拦截后段（r→0 甚至穿越）→ 分布与论文不同 | 增加终止条件（r ≤ r_min 或 r 由减转增后一段）并写入 metadata | P2 |
| C10 | 采样协议：IV-A dt=.5/O=20/stride 5；IV-B dt=.1/O=80/P=60 | 一致（`simulate_engagements(sample_stride=)`；`configs/engagement_*.json`） | 无差异 | — | 已对齐 |

---

## 4. 超参数与随机种子

| ID | 论文 | 当前 | 影响 | 方案 | 优先级 |
|---|---|---|---|---|---|
| D1 | 早停 1 m / 20 轮 | 0.001 m（见 B8） | 见 B8 | 军用配置改 1.0 | **P0** |
| D2 | IV-A：Adam 1e-3、step γ=0.5@30、batch 256、70 轮、clip 1.0、dropout .2、Nh 128、m 64、70/15/15、训练区标准化 | 全部一致（`configs/engagement_intention.json`、`learner.py:17–18`） | 无差异 | — | 已对齐 |
| D3 | IV-B：d=256/H=6/d_k=d_v=64、时间 H=4、深度 2、τ=.1、K=8、λ_sep=.1、ω=.4/.6、log-var[−6,6]、AdamW 1e-3/wd1e-4、batch 256、200 轮、warm-up 5+cosine、80/20 | 全部一致 | 无差异 | — | 已对齐 |
| D4 | 基线容量：LSTM/GRU 2 层 enc-dec、n_hid=256、p_do=.1；TCN 4 残差块 k=3 c=256 D={1,2,4,8} + 线性头；Transformer 2 层 d=256 h=4 d_ff=1024 p_do=.1 | `baselines.py` 一致（`baseline_width=256, baseline_layers=2, temporal_heads=4`，TCN 每块 2 卷积层略多；冒烟配置 `cpu_engagement.json` 的 `baseline_width=32` 只用于烟测） | 冒烟配置的基线容量与论文不符（仅烟测用） | 正式实验用 `engagement_forecast.json`；TCN 块内卷积数登记 | P2 |
| D5 | 随机种子：仅说"1000 次不同种子" | `seed=42 / split_seed=42`，study 多种子 | 见 C4 | 固定主实验种子 + 多种子报告 | P2 |
| D6 | 未给：图 dropout（基线给了 .1） | IV-B `dropout=0.1`、`intention_dropout=0.2` | 轻微 | 登记 | P2 |
| D7 | 未给：A（候选锚点数）、Λ_crop、ṙ_min、地形 | A=32、Λ=1、ṙ_min=0、地形关闭（D1/D2/D3 决策） | 非论文值，只能消融/敏感性 | 扫描 A∈{8,16,32}、ṙ_min∈{0,−50} | P2 |

---

## 5. 初始化方式

| ID | 论文 | 当前 | 影响 | 方案 | 优先级 |
|---|---|---|---|---|---|
| E1 | 未指定权重初始化（仅"linear regression output head"） | PyTorch 默认；解码器与基线输出头**零初始化**（恒速起点）（`model.py:140–141`、`baselines.py:51–52`） | 训练早期更稳但与论文不同；首步 ADE/FDE 对上游参数梯度为 0（已登记） | 增加 `zero_init_output_head` 开关，论文对照实验关闭并报告差异 | P2 |
| E2 | 未指定 | 可学习 log-variance 初始化 0（σ=1）；实体/位置嵌入 N(0,0.02) | 轻微 | 登记 | P2 |

---

## 6. 训练流程与迭代轮数

| ID | 论文 | 当前 | 影响 | 方案 | 优先级 |
|---|---|---|---|---|---|
| F1 | IV-B：多任务总损失**联合**训练（图13） | `pipeline` 固定两阶段（先分类 70 轮、冻结后训外推）；`compare` 才有 `graph_training=joint`（`workflows.py:46–64,72–143`） | 用 pipeline 跑 IV-B 会与论文流程不同（冻结的意图分支） | 军用 IV-B 明确走 `compare --graph-training joint`（脚本 `run_engagement.sh` 的 ablation 分支已是 joint；forecast 分支走 train+evaluate 的 joint 配置） | P1 |
| F2 | IV-A 是纯分类实验，无预测域 | `Windows` 一律要求 O+P 行（`data.py Windows.__init__`），IV-A 配置 P=12 | 每个窗口多要 12 个未来点 → 样本数减少、轨迹末尾被截断 | stage=intention 时允许 P=0 的窗口构造 | P1 |
| F3 | 每轮打乱、每轮末记录验证、按验证选最佳检查点、早停 | 一致（`engine.py:262–305`） | 无差异（阈值见 B8） | — | 已对齐 |
| F4 | "裁剪强度随训练迭代递增" | `crop_warmup_epochs=5` + `set_crop_progress` 按迭代递增（`learner.py:43–44`；`model.py:152–154`） | 无差异 | — | 已对齐 |
| F5 | Table III 在线迭代：刷新间隔短于水平长度 | `rolling-evaluate --refresh-steps 1`（dt=0.1 s ≪ 6 s） | 无差异 | — | 已对齐 |
| F6 | 单卡训练 | 支持 AMP 但未在本机验证；无 DDP | 不影响数值 | 服务器记录硬件/驱动 | P2 |

---

## 7. 评价指标与结果输出格式

| ID | 论文 | 当前 | 影响 | 方案 | 优先级 |
|---|---|---|---|---|---|
| G1 | ADE/FDE：欧氏距离、米、ADE 为逐步均值、FDE 为末端 | 一致（`objective.py:6–9`） | 无差异 | — | 已对齐 |
| G2 | e(t) 误差曲线（图16/18a） | `error_by_horizon`（`metrics.py:109–112`） | 无差异 | — | 已对齐 |
| G3 | 混淆矩阵（9 类）、**行归一化**混淆矩阵（11 档） | 二者均有（`metrics.py:126–137`） | 无差异 | — | 已对齐 |
| G4 | 图14 四复合场景 + 终端误差 150/202/81/173 m | **无**（缺 Table VI 场景复现入口） | 论文最具体的数值锚点无法核对 | 新增 `scripts/reproduce_figure14.py`（见 C7），输出"实测 vs 论文"表 | P1 |
| G5 | 论文口径目标值核对 | 无自动化 | 复现进度无法量化 | 新增 `scripts/check_paper_targets.py`：88%/94%/102 m/四场景终端误差 四项对照 | P1 |
| G6 | 图16/18 的基线/消融汇总图 | `report` 从 `comparison.json` 出图（`reporting.py`），`evasion_phases` 分组已有 | 基本可用；需确认用 256 宽度基线的配置 | 正式实验走 `engagement_forecast.json` | P2 |
| G7 | 逐类召回/宏 F1/多数类基线 | 有逐类支持与混淆矩阵（`metrics.py:116,135`），未直接给宏 F1 与多数类基线 | 类别覆盖（C1/C2）修好后需要这些量才能判定 88%/94% | 输出宏 F1、逐类召回与多数类基线 | P1 |

---

## 8. 论文自身未明确、不能算作实现差异的事项（必须登记，不可"补全成一致"）

1. **候选节点语义矛盾**：式15 的图是 2t_o 历史节点，Table I/Remark 3 又称节点为候选未来状态 → 采用 D1 锚定候选法。
2. **式4 定义非负法向过载，而十一档等级带符号** → C2 必须显式选边并声明。
3. **ṙ_min、P_T,t 可达域、κ_E/κ_R/λ_R、w_{j,τ} 均无数值** → 当前为 D3 + 默认 1.0 + 指数衰减，仅能做敏感性。
4. **式21 加性打分 vs Table I 式26 点积兼容度** → 采用前者。
5. **式23 环境约束集合与可行性指示的正反语义不完全一致** → 采用"越界=不可行"的乘法掩码。
6. **Table IV 的 VM∈[380,450] 与正文 IV-B 场景 VM=360 不一致**（论文自身）→ 库用 Table IV，图14 演示用 360，分开配置。
7. **"每标签 1000 次独立试验"的具体口径未定义**（是每类 1000 个样本还是 1000 次完整训练？）→ 当前按"多个独立数据种子 × 初始化种子"实现并标注。
8. **权重初始化、图 dropout、意图 CE 在联合训练中的权重均未给出** → 已登记为适配选择。

---

## 9. 优先级排序与执行顺序（目标：稳定复现对比/消融/图数值）

**第一批（P0，做完才有资格谈"论文口径数值"）**
1. **C1** 九类语义平衡抽样（含 no-maneuver 段）+ 每类覆盖断言 → 否则 88% 与九类混淆矩阵无从复现。
2. **C2** 十一档符号约定与实现（带符号法向过载）→ 否则 94% 与十一档混淆矩阵只有半边。
3. **B8/D1** 军用早停阈值改 1 m → 否则训练预算与选模点不同。
4. **A2** 交战描述符进入 Q_int/Q_tra（Track 读取 `engagement_states` + 静态分支扩展）→ 论文核心贡献，影响全部数值。
5. **B2** 几何项去掉 log-variance 正则（论文口径）→ 直接影响 ADE/FDE 量级。

**第二批（P1，使数值可比对、实验齐全）**
6. **C3** 数据规模到 5×10⁴ 窗口 + 断言。
7. **C5** + **A6** FOV 覆盖 0.5（IV-A）与注意力 FOV 掩码（IV-B 退化行为）。
8. **F1/F2** IV-B 走 joint；IV-A 分类窗口不再要求未来点。
9. **B5** 熵/抖动/分离按式38–41 逐式改写（保留 legacy 开关）。
10. **B1** 军用配置 `smoothness_weight=0`。
11. **G4/G5/G7** Table VI 四场景复现脚本、论文目标值对照脚本、宏 F1 输出。
12. **A1** IV-A 关闭静态分支（与 IV-B 分开报告）。
13. **C4** 蒙特卡洛规模（受算力，先做到 ≥100 并标注）。

**第三批（P2，登记与敏感性）**
14. A3 线性输出头开关、E1 零初始化开关对照；A7 可达硬阻断；C6 ā_eva 采样；C9 拦截终止；D7 锚点/ṙ_min 扫描；B4/B6 教师温度与 κ/λ 扫描；D4 TCN 块结构登记；F6 硬件与 AMP 记录。

**验收判据（建议写成脚本断言）**
- 九类：训练/验证/测试逐类支持数 > 0；测试准确率、宏 F1、多数类基线三者齐备 → 对照 88%。
- 十一档：11 档全部有样本；行归一化混淆矩阵对角带 → 对照 94%。
- IV-B：ADE/FDE 与论文 102 m 量级对照；早停日志显示阈值 1 m；e(t) 曲线与 ADE/FDE 汇总图（含四基线 + A1/A2/A4）。
- Fig.14：四场景终端误差与 150/202/81/173 m 对照。
- 全部结果：数据 SHA256、配置、代码哈希、划分、标准化、选模规则与硬件信息齐备。

---

## 10. 实施进展与勘误（2026-10-11 第二轮）

### 10.1 已完成并实测验证的条目

| 批次 | 条目 | 实现位置 | 实测证据 |
|---|---|---|---|
| P0 | **C1** 九类语义平衡抽样 | `engagement_scenarios.sample_semantic_class` / `_primitive_from_class`（按类采样基元 → 直飞类可达）+ `data.Windows._balance_classes`（窗口级重采样，分区内进行，不跨界泄漏） | 40 场景：`semantic_class_counts=[61,627,…]`，`semantic_classes_used=9`；`balance_intention_classes=true` 后每类窗口数完全相等（105/105/…） |
| P0 | **C2** 十一档符号约定 | `engagement_scenarios.signed_normal_overload`（法向过载在速度系(横向/纵向)取主分量符号）+ `overload_label` 严格按论文 `(I_m−0.5, I_m+0.5]` | 40 场景：`overload_levels_used=11`，含全部负档与 0 档 |
| P0 | **B8** 早停阈值 1 m | `config.early_stopping_min_delta_m` 默认 1.0；`configs/{engagement_forecast,paper_forecast,paper_interaction,server,server_interaction}.json` 同步 | 配置读取断言通过 |
| P0 | **A2** 交战描述符进入 Q_int/Q_tra | `data.Track.engagement` + `load_tracks` 读 `engagement_states` + `Windows` 输出 + `model.forward(engagement=…)`：静态分支 `[s_M(6) + s_L(9)]`、节点特征广播 9 维描述符 | `test_engagement_descriptors_reach_intention_and_temporal_queries`：意图 logits 与轨迹均随描述符变化；`node_embedding.in_features = 6+m+C+9` |
| P0 | **B2** 几何项不参与 log-variance | `objective.TrajectoryObjective.forward`（只有三组注意力教师做不确定性加权，ADE/FDE 保持单位权重） | `test_uncertainty_weighting_excludes_geometry` |
| P1 | **B1** 军用 `smoothness_weight=0` | `configs/engagement_*.json` | 配置断言 |
| P1 | **B5** 式38–41 逐式改写 | `config.teacher_regularizers="paper"`：熵用组内头均值 β̄、抖动作用于 β̄ 相邻时刻差异、熵下限双边 `abs()` | `test_paper_teacher_regularisers_differ_from_legacy` |
| P1 | **F2** IV-A 无预测域 | `config.validate` 允许 `stage=intention` 时 `prediction_steps=0`；`Windows`/`objective`/`engine`/`metrics` 全链路空 horizon 保护 | `test_intention_windows_accept_zero_prediction_steps` |
| P1 | **C5/A6** FOV 覆盖与注意力掩码 | CLI `--initial-fov-coverage`（脚本 IV-A 用 0.5、IV-B 用 1.0）；`engagement_graph` 新增 `fov_attention_mask`：不可见时删除跨实体边 → 退化为纯外推 | `test_fov_attention_mask_drops_invisible_engagement_links` |
| P1 | **C3** 数据规模到 5×10⁴ | CLI `--target-windows/--observation-steps/--prediction-steps/--window-stride`；`run_engagement.sh` 新增 `TRAJECTORY_TARGET_WINDOWS`（默认 50000）并按协议传入 O/P/stride | 参数求解逻辑与脚本接线完成（本机 CPU 未跑满预算，交服务器） |
| P1 | **C4** 每标签 1000 次蒙特卡洛 | 新脚本 `scripts/monte_carlo_intention.py`（按标签有放回独立重采样，各自种子；输出准确率/宏 F1/平衡准确率/多数类基线/混淆矩阵） | 端到端跑通（2 轮训练 + 50 次/标签：准确率 0.1489、宏 F1 0.0595、多数类基线 0.1111、9 类全在场） |
| P1 | **G4** Table VI / Fig.14 | `engagement_scenarios.simulate_engagement_batch` + `FIGURE14_SCENARIOS`（论文 Table VI 四行数值，从 PDF 图版读取）；新脚本 `scripts/reproduce_figure14.py` | 脚本产出四场景终端误差对照表（当前为恒速基线 1183 m，待训练模型） |
| P1 | **G5** 目标值核对 | 新脚本 `scripts/check_paper_targets.py`（88% / 94% / 102 m / 150-202-81-173 m，±15% 判据） | 输出对照表与 `n/7 anchors` 汇总 |
| P1 | **G7** 宏 F1 / 多数类基线 | `metrics.EvaluationDetails.classification_metrics`（新增 `intention_macro_f1`、`intention_balanced_accuracy`、`intention_majority_baseline`、`intention_classes_present`） | `test_classification_metrics_report_macro_f1_and_baseline` |
| P1 | **F1** IV-B 走 joint | `engagement_forecast.json: stage=joint`，`run_engagement.sh` forecast/ablation 均走 joint | 配置断言 |

### 10.2 本轮新发现并修复的实现缺陷（原审计未覆盖）

| ID | 缺陷 | 影响 | 修复 |
|---|---|---|---|
| **N1** | `_frame_from_velocity` 把"水平面内法向"与"竖直面内法向"互换：`horizontal` 实为竖直面方向、`vertical` 实为水平侧向（`engagement_scenarios.py`） | 九类标签与命令语义整体错位——"left turn"实际生成爬升/俯冲，`signed_normal_overload` 的左右手性也随之反转 | 重定义为 `lateral = up × e_T`（水平转向轴）、`vertical = e_T × lateral`（爬升轴），并同步 `_command_from_overloads`/`semantic_label`/`signed_normal_overload` 的变量名 |
| **N2** | 基元指令在生成时用当时的速度系固定为**世界向量** `a_prim`，而标签每步用**当前**速度系重算 | 长基元内标签会随姿态漂移，一个基元可能被判成多类 | 改为在速度系中恒定 `(a_yaw,a_pitch)`，每步用当前速度系重组指令 → 一个基元内标签稳定 |
| **N3** | `overload_label` 分箱半宽写死 0.5 g 而档间距为 2 g | 大量窗口均值落入区间外，被静默归到最近档，档位语义与论文不同 | 严格按论文 `(I_m−0.5, I_m+0.5]` 实现，并把"落空→最近档"显式化：新增 `overload_gap_report()` 诊断 + `censored_ends` 可选截尾 |
| **N4** | IV-A 无预测域时 `EvaluationDetails.finish()` 因无误差行提前返回 | **宏 F1、多数类基线、逐类精确率/召回率全部缺失** → 88% 无法按论文口径核对 | 抽出 `classification_metrics()` 并在提前返回前调用 |
| **N5** | 注意力教师分组隐含要求 `heads % 3 == 0`，但未校验 | 非法配置会在目标函数深处以 reshape 报错，或静默使用 0 宽分组 | `config.validate` 显式校验（论文 H=6 → 每组 2 头） |
| **N6** | 逃逸触发后目标恒保持反 LOS 机动 | 原始数据中逃逸段占 >90% 时间 → 九类中"无机动"样本天然稀少（C1 的根因） | 物理仿真不改（忠实论文触发律），改在窗口层按论文"balanced coverage"重采样（见 C1） |

### 10.3 论文原文核对后的口径澄清

1. **过载分箱**：论文原文为 `locating ā_n in the interval (I_m − 0.5, I_m + 0.5]`，边界归较大标号。该规则在 2 g 档间距下会留下 1 g 宽空档（论文未定义）；本实现按"落空归最近档"处理并计入诊断。因 `a_T,max = 9 g`，严格规则下 ±10 g 两档只在 `ā_n ∈ (9, 9.5]`（+）或 `[−9.5, −9)`（−）时可达；实测仍有样本落入，11 档全部被使用。
2. **静态分支**：式(12)–(14) 有序列分支与防御机静态分支两支，A1 的"关闭静态分支"建议撤销（见上表）。
3. **IV-A 输入含防御机状态**：`Q_int = s_M = [r_M, V_M, e_M]`（式13 直接用防御机运动学状态），Remark 2 进一步要求 Q_int/Q_tra 由 `s_L = [r, V_L, e_L]` 及其时间变化表征 → A2 的实现（`s_M` 6 维 + `s_L` 9 维）与原文一致。
4. **数据规模**：论文明确 "sampling–simulation–labeling loop is repeated until the target dataset size **and** geometry coverage are achieved" → 规模与几何覆盖都要达标，故 `--target-windows` 与 `--initial-fov-coverage` 并列使用。

### 10.4 仍未完成（下一批）

| 条目 | 状态 | 说明 |
|---|---|---|
| C3 实跑 5×10⁴ | 待服务器 | 脚本与参数已就绪（`TRAJECTORY_TARGET_WINDOWS=50000`） |
| C4 实跑 1000 次/标签 | 待服务器 | 脚本就绪，本机以 50 次验证链路 |
| G4 用**训练后模型**复现 Fig.14 | 待服务器 | 入口就绪，当前为恒速基线对照 |
| **C6 ā_eva 有界采样** | ✅ 已实现（第五轮） | `simulate_engagements` 逃逸触发后每 ~4 s 段从 `evasion_overload_range`(默认 0.5–1.0×aT_max) 重采样并段内保持；CLI `--evasion-overload-range`；回归测试 `test_evasion_overload_is_resampled_per_segment` |
| **C9 交战终止** | ✅ 已实现（第五轮） | 拦截（r≤50 m）或确定错过（分离连续增长 30 内部步）即终止，尾部冻结填充，diagnostics 记录原因；CLI `--intercept-radius-m/--miss-grow-steps`；回归测试 `test_intercept_termination_fills_frozen_tail`；实测 12 场景 = 2 拦截 + 3 错过 + 7 跑满 |
| **A7 不可达链路硬阻断** | ✅ 已实现（第五轮） | `engagement_graph` 跨实体边按 `a_req=\|v_rel\|·θ/lag ≤ 9g` 删边，`overload_reachability` 开关（默认 False 兼容旧 checkpoint；四份军用配置显式 true；E-off 消融联动关闭）；回归测试 `test_overload_reachability_blocks_unreachable_cross_entity_links` |
| A3 / E1 / D7 / B4 / B6 / F6 | P2 登记 | 见 §9 第三批；均为开关化或敏感性扫描 |
| B5 分离项（式41） | 已定为"均值" | 式41 为跨组头对 cos² **求和**，元素数达 `3·2·H²` 量级；直接求和会把损失放大约两个数量级并主导训练。本轮保留均值口径，如需严格求和须同步重标定 `separation_weight`，已登记为 P2 |
| 逐类"无机动 vs 轻微转弯"混淆 | 观察项 | 论文称主要混淆发生在语义相邻类；C1 修好后须在真实训练后核对混淆结构 |

### 10.5 第七轮（2026-10-11 深夜）：式(36) 边子集、ϕedg 的 V_L/V̇_L、意图阶段缺陷

| 条目 | 状态 | 说明 |
|---|---|---|
| **G1 式(36) 分组边子集 𝟙(i→j∈E_k(i))** | ✅ 已实现（第七轮） | `_edge_subsets` 重建 temporal/spatial/countermeasure 三类边，三组教师各自在自己的子集上归一化（H_E→temporal、H_R→spatial、H_T→countermeasure，**决策 D8**，论文未列表故登记）；E_k 为空的行回退到无限制掩码防 π 未定义；`teacher_edge_subsets` 默认 False，IV-B 两份配置显式 true；测试 `test_teacher_distributions_use_their_own_edge_subset` |
| **G2 ϕedg 的 V_L 与其时间变化** | ✅ 已实现（第七轮） | 边描述符补 V_L = −(v_T−v_M)·e_L（式(2)，目标–防御机配对求值）与 V̇_L（块内后向差分）；`ENGAGEMENT_EDGE_DIM` 19 → 21；测试 `test_edge_descriptor_carries_los_relative_speed_and_its_rate` |
| **G3 描述符注释与实现不一致** | ✅ 已修（第七轮） | 生成器注释写 10 维含 V_L、实际写 9 维（V_L 算出后被丢弃）；注释改为 9 维并说明 V_L = −rdot 故未重复存储 |
| **G4 意图阶段泄漏图教师** | ✅ 已修（第七轮，真实缺陷） | `classifier_config` 关 `attention_supervision` 却继承 `military_teachers=True` → `validate` 抛错、**IV-B 的 `pipeline` 两阶段训练完全跑不起来**；现显式关闭（依据：教师监督属 III-C 的 GAT 注意力层，IV-A 为 BiLSTM+CE）；测试 `test_intention_stage_drops_engagement_teacher_supervision` |
| **X1 教师分数定义域** | 登记（架构级） | 式(42)–(44) 的 V/a/j 来自"式(32) 外推结果的增量序列"、式(45) 是"node Δy_i→Δy_j"、Remark 3 "nodes correspond to candidate future states" ⇒ 教师本应定义在**候选未来状态/预测增量**域；当前按 D1 落在观测历史域。属 A5（论文在"式15 的 2t_o 历史节点"与"候选未来状态"之间自相矛盾）的下游，需整体切候选图架构才能逐式复现，本轮不做 |
| X2 式(26)–(31) | 登记 | 正文中不存在，编号从 (25) 直接跳到 (32)（论文排版问题，非实现缺口） |
| X3 式(45) 归一化 | 登记 | `arccos(Δy_j·Δy_i/\|Δy_i\|)` 缺 \|Δy_j\|，论文疑为笔误；代码用对称速度夹角版（量纲正确），维持 |
| X4 ρ_k / ζ_k / τ_k 分组参数 | 登记（并入 B4/B6） | 论文带下标 k，代码三组共享；数值未给，走敏感性扫描 |
| X5 组→边子集映射 | 登记（D8） | 见 G1；建议加一组消融验证映射选择 |
| X6 走廊中心 Y*_k / 任务点 Y,T* / 可达域 P_T,t | 登记 | 论文均未给数值或公式；当前用窗口锚点+航向外推、"v·T_p+0.5·a_max·T_p²"、μ_t=σ(κ(2·progress−1)) 近似 |

### 10.6 验证入口

### 10.8 第九轮（2026-10-13）：P2 清单复核勘误 + E1/B4 字面对齐包落地

对 §9 第三批 P2 逐行复核当前代码（以文件为准，审计表此前滞后），结果：

| 项 | 复核结论 | 证据 |
|---|---|---|
| A3 线性输出头 | ✅ **已对齐**（第五轮即落地） | `config.py: decoder_head="linear"` 为默认；`model.py` 按 IV-B 原句实现单线性头；engagement 配置未覆盖即生效。守护测试 `test_paper_fidelity.py::decoder_head mlp 对照` |
| B5 式(41) 求和口径 | ✅ **已对齐**（第五轮即落地，§10.4"保留均值"记载过时） | `objective.py:193-196` 跨组头对 cos² 求和（批均值归一）；守护测试 `test_separation_sums_cross_group_cosines`。**勘误**：论文 IV-B 原文明给 `λsep = 0.1`（连同 ω1=0.4、ω2=0.6），故 sum + separation_weight=0.1 即字面，无需重标定 |
| E1 零初始化开关 | ✅ **本轮落地** | `config.py: decoder_zero_init: bool = True`（默认=现状，字节不变）；`model.py` 条件初始化；关闭即 PyTorch 默认（论文字面）。守护测试 `test_literal_alignment.py`（4 项，含与工程默认的逐参数位相同断言） |
| B4 教师温度 | ✅ **本轮配置化** | 论文式(36)-(37) 无温度（"attention τ=0.1" 为注意力温度，另已对齐）；字面读法 `teacher_temperature=1.0`，见 `configs/engagement_forecast_literal.json`（与工程基线逐键 diff 恰 2 项，测试断言） |
| D7/B6/F6/D3 等 | 维持敏感性扫描项 | 论文未给数值，无对齐目标 |

部署：`server_package_v4.tar.gz`（942 文件，sha256 `66820cb71f3f72277020a992a0e6e36ee73d7417160e7a8dcb8e14a4d4ed8a52`）。字面包只作用于 IV-B（IV-A 两配置 military_teachers=False），对照=第三轮 forecast 数字。验收：全量 127 passed；`run.py check --config engagement_forecast_literal.json --backward` 本机通过。

```bash
# 单元/回归（本机 CPU 全绿）
PYTHONPATH="src;.." python -m pytest tests -q            # 107 passed（第七轮后）

# 军用端到端冒烟（第七轮修复后 pipeline 可跑通）
python run.py simulate-engagement --output outputs/smoke_eng.npz --scenes 8 --steps 200
python run.py pipeline --data outputs/smoke_eng.npz --config configs/cpu_engagement.json \
  --output outputs/smoke_train --intention-epochs 1 --epochs 2
python run.py check --backward --config configs/engagement_forecast.json

# IV-A 小规模端到端
python run.py simulate-engagement --output data/iva.npz --scenes 40 --steps 200 \
  --dt 0.5 --sample-stride 5 --initial-fov-coverage 0.5
python run.py train --data data/iva.npz --config configs/engagement_intention.json \
  --output outputs/iva --device cuda:0 --epochs 70
python scripts/monte_carlo_intention.py --data data/iva.npz \
  --checkpoint outputs/iva/best.pt --trials 1000 --output outputs/iva/mc.json

# Table VI / Fig.14 与论文目标值
python scripts/reproduce_figure14.py --output outputs/figure14 --checkpoint outputs/forecast/best.pt
python scripts/check_paper_targets.py --intention outputs/iva/mc.json \
  --overload outputs/ovl/mc.json --forecast outputs/forecast/validation_metrics.json \
  --figure14 outputs/figure14/figure14_summary.json

# 完整协议（含 5×10⁴ 窗口自动求解）
TRAJECTORY_TARGET_WINDOWS=50000 bash bash/run_engagement.sh outputs/military cuda:0 forecast
```

### 10.7 第八轮（2026-10-11）：交互式分布裁剪器（式15–31 / Table I）逐式一致性核对

范围＝交互图 G=(V,E,φ_edg)（式15）、门控 GAT 消息传递（式16–21）、裁剪约束（式22–31 / Table I 式26–30），
对应 `engagement_graph.py`、`candidate_pruning.py`、`model.py` 的 GAT 部分。原则：**论文明确写了而代码做错/漏做的，必须改；
论文确实没写清的，做合理化处理并登记，不得冒充逐式复现。** 公式原文以 `tmp/paper_text.txt` 与 PDF 第 5–7 页（Table I 放大）为准。

**A. 论文明确、代码确有错误/遗漏 → 已修复**

| ID | 公式 | 原代码问题 | 修复 | 守护测试 |
|---|---|---|---|---|
| **K1** | 式(20) m_j=σ(f_n(h_j,**u**)) | 节点门 `node_gate=Linear(D,heads)` 只吃 h_j，**未吃交战描述符 u**（边门 f_e 已通过 edge_context 吃 φ_edg 符合式19，唯独节点门漏 u） | `model.py`：`TemporalGraphAttention` 增 `node_context_dim` 参数与 `node_context=Linear(u_dim,heads,bias=False)`，forward 增 `node_u` 并把其输出加入节点门 logits；`Forecaster` 军用图层传 `node_context_dim=9`，forward 拼 `node_u`（历史两块用逐时刻 engagement，候选块用末端 engagement 扩 O 行）；民用路径 `node_u=None` 不变 | `test_node_gate_consumes_the_engagement_descriptor`（军用有投影且注意力监督产生非零梯度；民用为 None） |
| **K2** | Table I Step1 + D1（G=1，候选未来挂观测末端） | `engagement_edges` 把候选块当成"第三个按时间索引的实体"：候选分支 a 被时间/空间边接到观测时刻 a。IV-B 配置 O=80、A=32 时真候选（行0–31）全部接到**最老的 32 个观测节点**，观测末端节点（行79）只连填充行——候选未来没挂在 D1 要求的观测末端 | 重写 `engagement_graph.py`：通用时间/空间/对抗边仅限历史两实体（entity<2）；新增 `candidate_link`，把每个有效候选行（entity==2）**双向**只连到末端目标节点 O−1 与末端防御机节点 2O−1，类型标为 countermeasure；候选块 V_L 填末端配对值（未来 V̇_L 未知→0）；candidate_link 豁免过载可达/FOV/速度过滤（候选已过式24 能量门）但仍受 valid 掩码（裁剪/填充候选断开），自环恒保留 | `test_candidate_branches_converge_on_terminal_nodes`（候选只连两个末端节点、候选行间无边、末端节点聚合全部存活候选、非末端历史不接触候选、边类型 one-hot 正确） |
| **K3** | 式(27) cesc=1[(1/T)Σ_k ṙ_k≥ṙ_min]；式(28) cenv=1[g(ŷ)∉F, **∀k**] | Cesc 只算候选**末端点** ṙ、Cenv 只判末端高度，是端点判据而非 Table I 的"整个采样窗口时间平均 / 每个采样点" | `candidate_pruning.py` 新增 `_constant_turn_arc`（Rodrigues 恒速协调转弯，按 s=1/S..1 闭式积分位置/速度，S=`candidate_arc_samples` 默认 12）；Cesc 改为防御机匀速传播后**全弧段 ṙ_k 的时间平均**≥ṙ_min；Cenv 改为**全弧段每个采样点**高度都在高度带内；候选节点特征仍取弧末端（s=1，即弦）；`config.py` 新增 `candidate_arc_samples=12` 与正整数校验；返回字典额外暴露安全网之前的 `escape/environment/energy` 三因子（对应式30，便于独立审计） | `test_cesc_uses_window_average_range_rate_not_endpoint`（晚交会：端点已分离但窗口均值仍逼近→拒；早交会→收）、`test_cenv_checks_altitude_at_every_arc_point`（∀k 高度带） |

> 全量回归中另发现并修掉两个仅在 B>1 / A=32 才触发的广播 bug（小批量 B=1 测试恰好掩盖）：弧段速度/位移系数应为四维 `speed[:,None,None,None]`；candidate_link 的速度可行性掩码第二轴应为 `plausible[:,:,None]` 而非 `plausible[None]`。

**B. 论文明确、曾考虑改但核对后撤回（不改）**

| ID | 公式 | 拟改 | 撤回依据 |
|---|---|---|---|
| **K4** | 式(24)/(29) Cene | 曾拟给 Cene 增加瞬时过载硬门 a_req=vθ/T≤a_max | 式24/29 字面只约束**平均加速度增量** (1/T)Σ‖a_{T,k}−a_{T,k−1}‖（恒速协调转弯下＝vθ²/T²）；`test_paper_fidelity.py::test_candidate_energy_gate_uses_the_mean_acceleration_increment`（A-05）**刻意断言** speed=300、T=6s 反转机动（a_req≈157>a_max=88.3，但 mean_increment≈82.3<a_max）时候选必须保留。瞬时过载由软代价 P_ij=a_req/a_max 与能量教师式45 负责；加硬门反而违背式24，维持原样 |

**C. 论文未写清 → 合理化处理（登记，非逐式复现）**

1. **D1 锚定候选法**：式15（2t_o 历史节点）与 Table I/Remark 3（邻居＝候选未来状态）自相矛盾，沿用锚定候选张量接口；K2 只把其 G=1 的接线改对，未改变该架构取舍。
2. **式21 加性注意力 vs 式26 点积** q·k/√d_k+b(r_ij)：代码取加性式（以正文式21/Fig.5 为准）；式31 印刷的归一化分母 S_i 漏写 exp 项属笔误，代码 softmax 正确再归一化。
3. **式23 集合正反记号混乱**：代码按物理语义"可行=1、越界=0"的乘法掩码实现。
4. **盲视旁路范围**：防御机不在 FOV 时旁路 Cesc，但**保留 Cene**（Cene 与交战无关，比 Step4 字面"只留式23"更合原则），登记。
5. **防御机未来运动**：论文未给，窗口内按匀速（零加速度）传播；弧段采样数 S=12 为收敛性工程选择（配置暴露，非论文值）；候选块 V_L 取末端配对值、V̇_L=0。
6. 其余既有登记不变：时间边 `temporal_radius` vs "consecutive"、σ 取 ELU、多头拼接后 output 投影、软裁剪用 logits 减分+softmax、全裁空保留最低代价分支的安全网、地形 D2 默认关、ṙ 与 V_L 互为负值的冗余。
7. **X1（教师定义域）** 属架构级缺口（教师本应在候选未来域、D1 落在观测历史域），超出本次裁剪器范围，不动（见 §10.5）。

**验证**：新增 4 个用例后裁剪器/交互图两文件 17 passed；全量 `pytest tests` **111 passed**（第七轮基线 107 + 本轮 4），0 failed（本机 CPU，Python 3.12 / torch 2.7.1+cpu）。

### 10.9 梯度累积（batch 256 显存守恒，2026-10-13 追加）

**发现**：论文两处明文规定 batch size（paper_text.txt:1287 IV-A "batch size is set to 256"、:1580 IV-B "The batch size is set to 256"）。RTX 5080 16GB 无法一次前向 256 样本（实测训练进程达 14.31GiB 后 OOM），此前拟降 batch 128 属未披露偏差，撤销。

**方案**：等效 batch 256 = 物理 micro batch 128 × 梯度累积 2。模型无 BatchNorm 类批统计层，累积梯度与全批梯度严格相等。

**实现**（默认路径字节不变，micro_batch_size=0 时走原路径）：
- config.py：新字段 `micro_batch_size: int = 0` + 校验（0 或 batch_size 的正因子）
- learner.py：`_update_accumulated` 用样本数加权和反传（loss×count），组末 `grad.div_(组内总样本数)` 恢复组均值——对任意组构成（含 epoch 末尾悬空组）与 DataLoader 真批 256 的梯度严格相等；`flush()` 在 epoch 末补步尾巴组（engine.run_epoch 训练循环末调用），复现普通 DataLoader 部分批语义；`_group_pending/_micro_index` 入 checkpoint
- engine.py：`make_loader` 用 `micro_batch_size or batch_size` 做物理批；`steps_per_epoch` 不变（本就是等效批的优化器步数）

**验证**：全量 132 passed（新增 5 条守护：默认关闭、校验、全组等价、尾巴组等价、json 钉住 batch 256/micro 128）；CPU 双配置 `check --backward` 通过。

**口径**：`batch_size` 字段保持论文字面 256；`micro_batch_size` 属于部署层显存守恒机制，不改变优化语义，不计入"工程选择偏离"清单。

### 10.10 逃逸指令段内冻结（2026-10-14 凌晨追加，第八轮 T1 归因产物）

**发现**：T1 低准度归因链（纯度分桶 → 逃逸相位分层 → 交战时长扫描）最终定位到逃逸段实现与论文的一处语义偏差。论文 §IV-B 明文："The evasion maneuver is generated by **holding a_eva(t) within each short primitive segment and updating it at segment transitions**"——逃逸指令（方向+幅值）应段内冻结、段间更新。原实现在段间只重采幅值（C6），**方向却每步用实时 eL(t)、vT(t) 重投影**——等效连续转向，而论文是分段冻结的恒指令段。

**影响**：连续转向使逃逸指令成为几何的实时函数，语义标签随几何连续漂移，窗口识别难度系统性抬高（逃逸段准确率 53.1% vs 逃逸前 87.6%）；分段冻结后每段为恒加速度机动，与基元同构、可辨识性恢复。投影公式本身（ã=(I−e_T e_Tᵀ)(−e_L)、饱和 min(ā,a_T,max)、ε=1e-6）逐字一致，无偏差。

**修正**：`simulate_engagements` 与 `simulate_engagement_batch`（Fig.14）两处生成器均在段切换瞬间一次性计算 `_evade_command` 并冻结至下一段切换；batch 版新增 4 s 段长（与主生成器 `evasion_segment_seconds` 默认一致）。docstring 同步更新。

**背景测量**（审计证据链）：purity_buckets 分层（pre-evasion 322 窗 87.6% / evasion 3269 窗 53.1%，加权精确还原 overall 56.2%）；交战时长扫描 steps∈{120,200,400} → 逃逸窗占比恒 ~92%（触发 r≤2/3·r₀ 数秒内生效，时长不是混合比例的杠杆），overall 0.650/0.595/0.562 随逃逸段年龄单调。

**性质**：论文保真度修正（纠错），非对 88% 的调参——修正依据是论文明文句，不是结果导向。

**待验证**：服务器重生成 IV-A 数据并重训（约 10 分钟链），预期逃逸段准确率显著上行、整体向论文口径靠拢；IV-B/Fig.14（T3/T4）随之在 Round 5 全链重跑中统一更新。
