# 军用对抗版改造计划：demo 与论文逐模块对照（MILITARY_PORT_PLAN）

> 依据：Yu 等 2026 IEEE TAES《Trajectory Prediction for Maneuverable Target Based on Intention
> Recognition and Intercept Engagement》（DOI 10.1109/TAES.2026.3677846），全文精读 + Table I/II/III/IV/V/VI
> OCR 恢复（表格正文为矢量字形，文本层不可提取，已于 2026-10-11 用 RapidOCR 4× 渲染逐格恢复并复核）。
> 现状基线：本目录 v0.4 民用 demo（六类通行场景，2 实体交互，民用三教师）。

---

## 0. 一句话总结

demo 已搭好论文的**骨架**（解耦结构、2O 节点交互图、门控+Top-K+渐进裁剪、自回归 GRU、KL 教师、不确定性加权、两阶段训练、在线滚动），
但**血肉全是民用语义**：场景是低速通行规则、边特征是民用相对运动、教师是平滑/速度/时间、意图是通行机动。
军用版需要替换四层内容：**交战数据层、对抗边特征层、E/R/T 教师层、候选分支裁剪层**，并把协议对齐 IV-A / IV-B 两套实验设置。

---

## 1. 逐模块对照总表

| # | 模块 | 论文要求 | demo 现状 | 差距 | 优先级 |
|---|------|----------|-----------|------|--------|
| 1 | 场景/数据生成 | 3D 拦截交战：式(1)-(2) 质点运动学、r≤(2/3)r₀ 触发反 LOS 逃逸、速度法平面投影、9g/15g 饱和、FOV 60°、Table IV 参数采样、NATOPS 基元拼接、9 类语义标签 + 11 档过载标签 | `civilian_scenarios.py` 六类 2D 通行场景（0.9–1.5 m/s、右优先规则、九类通行机动标签） | **完全缺失**，需新建 `engagement_scenarios.py` | P0 |
| 2 | 意图空间 | 9 类 NATOPS 语义（no maneuver/left turn/right turn/climb/dive/left-climb/right-climb/left-dive/right-dive）+ 11 档法向过载 {−10,…,10}g 窗口均值归档 | 9 类标签语义是民用通行（直行/转弯/保持等，由 yaw_rate/vz 阈值判别） | 标签集语义不符，需按机动基元主导语义标注；11 档任务缺失 | P0 |
| 3 | 交互图边特征 φ_edg | 交战描述符：相对距离 r、LOS 相对速度 ṙ、LOS 方向 eL 及变化率、可见性指标（FOV 60°）——式(15) | `interaction_graph.py` 7 维民用边特征（归一化位置差/速度差/时间差） | 需新建交战描述符计算（r/ṙ/eL/ėL/可见性），维度扩展 | P0 |
| 4 | 对抗边（三类边） | 时序边（同实体相邻时刻）+ 空间边（跨实体同刻）+ **对抗边**（拦截交战与逃逸机制约束） | 时序边 + 空间边已有（`interaction_graph.py`），无对抗边 | 对抗边 = 由交战描述符派生的边类型嵌入，需加第三类边标记 | P1 |
| 5 | 候选未来分支裁剪（交互分布裁剪器） | Table I/式(22)-(25)：邻居 j = 候选轨迹运动学状态；Cesc（ṙ≥ṙ_min 逃逸约束）、Cenv（地形栅格 g(y)∉F）、Cene（Σ|a_k−a_{k−1}|≤a_max）；硬裁剪 m(j)=Cesc·Cenv·Cene；软裁剪 α_ij←m_ij·exp(−Λ_crop·P_ij)/S_i，Λ_crop 随训练递增；FOV 不可见时只用交战无关子集 | `model.py` 的门控+Top-K+渐进裁剪 crop_progress 只作用于**历史邻居消息**，无候选未来分支语义 | **最大工程难点**（论文式(15)说 |V|=2t₀ 历史节点，Table I 说邻居是候选未来状态——自相矛盾，见 §4 工程决策 D1） | P0 |
| 6 | 教师监督 | 三组注意力头：H_E 能量（式(45)-(47)：法向过载需求 a_req、制动损失 c_r、jerk，LSE 聚合）、H_R 风险（式(48)-(49)：距离+逃逸率+可达域，时间加权 LSE）、H_T 时效（式(50)-(53)：走廊径向/法向偏差 ↔ 终端收敛门控 μ_t） | `objective.py` 民用三教师：加速度平滑、速度一致、时间邻近 | **完全替换**（KL+熵下限+抖动+cos² 分离框架可复用，教师打分函数全部重写） | P0 |
| 7 | 损失权重 | 式(54)：注意力损失 = Σ_{k∈{E,R,T}} [1/(2σ̃_k²)·L_k + log σ̃_k] + λ_sep·L_sep；式(55)：L_tra = L_attn + 1/(2σ̄_sup²)(0.4·ADE+0.6·FDE) | 已有 uncertainty 加权 + log_variance clamp[−6,6] + λ_sep=0.1 + ω₁/ω₂ | 结构一致，系数对齐即可；教师组数 3 不变 | P1（改造成本低） |
| 8 | 意图识别网络 | BiLSTM(hidden=128) + 防御机运动学状态 sM 的静态分支 → 式(12)-(14) | `model.py` IntentionEncoder：BiLSTM(6→128) + 静态分支(6→64) 融合 | 结构基本一致；需把静态分支输入换成 sM=[r,V_M,eL] 类交战状态（或防御机 6 维状态），并支持 11 档头 | P1 |
| 9 | GAT 配置 | IV-B：空间层 d=256、H=6、d_k=d_v=64；时序层 H=4；两层深；温度 τ=0.1；FOV mask 启用；半径连接 + Top-K=8 | `config.py` graph_dim/heads/graph_top_k 已参数化（demo 用小维度） | 只需按协议配置；FOV mask 需新增 | P1 |
| 10 | 解码器 | 自回归 GRU + 时序记忆 cross-attention + 线性输出头；bound（速度/加速度）投影 | 已有 autoregressive_gru + cross-attention + bound_increments | 结构一致；bound 需换成过载语义（速度上限 + 法向过载上限） | P2 |
| 11 | 采样协议 | IV-A（意图）：dt=0.5s、T_o=10s(O=20)、stride=5、~5×10⁴ 样本、70/15/15 track-level 划分、Adam 1e-3、step γ=0.5@30、batch 256、70 epochs；IV-B（外推）：dt=0.1s、T_o=8s(O=80)、T_p=6s(P=60)、50k 样本 80/20、AdamW 1e-3/wd 1e-4、batch 256、200 epochs、warmup 5 + cosine、早停 ADE/FDE 1m/20 epochs | 单套协议（dt=0.5、O=10、P=4 量级） | 需两套 config（`engagement_intention.json` / `engagement_forecast.json`）+ CPU 冒烟版 `cpu_engagement.json` | P0 |
| 12 | 在线迭代预测 | Table III：Δ_Tp 间隔刷新观察窗、意图→图→裁剪→外推，更新频率高于水平长度 | `rolling.py` RollingPredictor 已实现逐帧刷新 | 协议对齐即可（refresh_steps 参数已有） | P2 |
| 13 | 基线对比 | LSTM/GRU/TCN/Transformer 匹配容量：2 层 enc-dec n_hid=256 / TCN 4 blocks k=3 c=256 D={1,2,4,8} / Transformer 2 层 d=256 h=4 dff=1024 | `baselines.py` 已有四基线（宽度小） | 宽度按协议配置 | P2 |
| 14 | 消融 | I（意图）/E（交战耦合）两因素：Proposed(+I,+E)、A1(−I,+E)、A4(+I,−E)、A2(−I,−E) | 已有 I/C/T 消融（C=民用邻居上下文） | C 语义升级为 E（交战耦合），四象限复用 compare 框架 | P1 |
| 15 | 评估指标 | ADE/FDE（e(t)=‖ŷ_t−y_t‖）、误差-水平曲线、终端误差 | `metrics.py` 已有 ADE/FDE/按水平/按场景/bootstrap | 复用；增加"逃逸触发前后"分组统计（对抗场景特有） | P2 |

---

## 2. 论文精确协议参数（含 OCR 恢复值）

### Table IV：场景采样参数（机动意图样本库）
| 参数 | 范围 |
|---|---|
| 初始距离 r₀ | [4, 8] km |
| 初始 LOS 方位角 q_ϕ,0 | [−180°, 180°] |
| 初始 LOS 俯仰角 q_ϑ,0 | [−30°, 30°] |
| 初始高度差 Δh₀ | [−1, 1] km |
| 初始目标速度 V_T | [250, 350] m/s |
| 初始防御机速度 V_M | [380, 450] m/s |
| 偏航机动过载幅值 a_ϕ,0 | [−88.3, 88.3] m/s²（= ±9g） |
| 俯仰机动过载幅值 a_ϑ,0 | [−88.3, 88.3] m/s²（= ±9g） |

### Table V：数据生成流程（A1–A6）
A1 采样初始条件与机动参数 → A2 实例化终端交战场景并初始化双机状态 →
A3 离散时间质点运动学传播并更新 LOS 交战状态 → A4 施加逃逸决策（过载限内生成 a_eva(t)）→
A5 滑窗提取 + 九类语义意图标注 → A6 追加入库，循环 A1–A6 至规模与覆盖达标。

### Table VI：四复合机动场景初始条件（IV-B 单次外推演示）
| 场景 | a_ϑ,0 (m/s²) | a_ϕ,0 (m/s²) | r₀ (km) | e_L,0 |
|---|---|---|---|---|
| Left-Climb | 39 | −58 | 5.3 | [0.98, −0.15, 0.09]ᵀ |
| Left-Dive | −39 | −58 | 5.3 | [0.98, −0.15, −0.09]ᵀ |
| Right-Climb | 39 | 58 | 5.3 | [0.98, 0.15, 0.09]ᵀ |
| Right-Dive | −39 | 58 | 5.3 | [0.98, 0.15, −0.09]ᵀ |

终端误差参考（Fig. 14）：150 / 202 / 81 / 173 m。

### Table I（Algorithm 1）：交互分布裁剪器（关键恢复）
- 输入：N(i)、r_min、F（自由栅格集）、a_T,max、T；
- Step 1：**候选目标轨迹运动学状态取自层 ℓ 中邻居 j 到节点 i，采样窗口 T**——证实"邻居=候选未来分支"；
- Step 2：初始兼容性分数 q_i^(0)（式 26）→ Softmax 得初始边权（式 21）；
- Step 3：约束检查——Cesc(j)=1[ṙ≥ṙ_min]（式 27）、Cenv（式 28，地形栅格）、Cene（式 29，Σ|a_T,k−a_T,k−1|≤a_T,max）；
  可行性指标 **m(j) = Cesc·Cenv·Cene ∈ [0,1]**（式 30）；
- Step 4：**FOV 指示器编码于 φ_edg；防御机不可见时仅用交战无关约束子集（式 23 地形约束）裁剪**；
- Step 5：裁剪后归一化 **α_ij = m_ij·exp(−Λ_crop·P_ij)/S_i**（式 31），S_i=Σ_{k∈N(i)}，Λ_crop 为裁剪强度（随训练迭代递增）；
- Step 6/7：信息聚合更新隐状态；监督学习更新参数。

### Table II（Algorithm 2）与 Table III（Algorithm 3）
- Algorithm 2（离线训练）：归一化→构图(15)→软掩码+注意力(19)-(21)→裁剪(22)-(25, Table I)→消息传播(17)-(18)→解码增量(32)积分(33)→总损失(55)反传；
- Algorithm 3（在线）：Δ_Tp 间隔刷新 X(t_k), s_M(t_k), s_L(t_k)→意图(12)-(14)→构图融合(15)-(16)→裁剪(22)-(25)→注意力(19)-(21)+隐状态(17)-(18)→解码积分(32)-(33)；推理只执行前向；t_{k+1}←t_k+Δ_Tp 循环。

### 训练超参（IV-A / IV-B）
| 项 | IV-A 意图 | IV-B 外推 |
|---|---|---|
| dt | 0.5 s | 0.1 s |
| 观察窗 T_o | 10 s (O=20) | 8 s (O=80) |
| 预测水平 T_p | — | 6 s (P=60) |
| stride | 5 | —（50k 实例） |
| 样本量 | ~5×10⁴（70/15/15 track-level） | 50 000（80/20 train/val） |
| 优化器 | Adam lr=1e-3 | AdamW lr=1e-3, wd=1e-4 |
| 调度 | step γ=0.5 @epoch30 | warmup 5 + cosine |
| batch | 256 | 256 |
| epochs | 70 | 200（早停 ADE&FDE 改善<1m 连续 20 epochs 终止） |
| 梯度裁剪 | 1.0 | 1.0 |
| 其他 | 测试精度 ~88%（9类）/ ~94%（11档） | λ_sep=0.1、ω₁=0.4、ω₂=0.6、log_var clamp[−6,6]、K=8、τ=0.1、FOV mask on |

### 逃逸机制（IV-B 正文）
- 触发：r ≤ (2/3)·r₀；
- 指令：ã_eva(t) = −e_L(t)·(I − e_T(t)e_Tᵀ(t))（反 LOS 方向投影到速度法平面）；
- 饱和：a_eva(t) = a_eva·ã_eva/max{‖ã_eva‖₂, ε}，a_eva=min{ā_eva, a_T,max}，ε=1e-6；
- 分段保持：a_eva(t) 在短基元段内保持常值、段切换时更新（连续性）；
- 交战常量：V_T=300、V_M=360 m/s、a_T,max=9g、a_M,max=15g、θ_FOV=60°。

### 11 档过载意图（IV-A 第二实验）
I={−10,−8,−6,−4,−2,0,2,4,6,8,10}g，g=9.81；统计量 ā_n=(1/t_o)Σ_{k=1..t_o} a_n(k)；
标签 Im 满足 ā_n∈(I_m−0.5, I_m+0.5]，边界 tie 归较大档。

### 教师监督三组（式 42–53）
- 能量 H_E：a_req（相邻增量向量夹角法向加速度，式 45）、c_r=a_r,t−a_r,min（式 46）、jerk j_t（式 44），
  归一化后 s^(E)=−(1/κ_E)·log(e^{κ_E ã²}+e^{κ_E j̃²}+e^{κ_E c̃²})（LSE，式 47）；
- 风险 H_R：R_j,τ=LSE_κR(r̃_t, ṙ̃_t, P̃_T,t)（式 48），s^(R)=(1/λ_R)·log Σ_τ w_j,τ·exp(λ_R·r_j,τ)（式 49）；
- 时效 H_T：走廊径向/法向偏差 δ_r,δ_n（式 50-51），s^(T)=−[(1−μ_t)·LSE_κT(δ_r,δ_n) + μ_t·LSE_κT(‖y_t−Y‖,|T−T*|)]（式 52），
  门控 μ_t=σ(κ_T(t−T*)/T)（式 53）——早期偏走廊修正，临拦截偏终端收敛；
- Remark 4：教师分数仅训练期使用，推理不输入（防监督信号泄漏）。

---

## 3. demo 代码资产复用映射

| 论文构件 | 复用 demo 模块 | 改动 |
|---|---|---|
| 式(1)-(2) 运动学 + 逃逸 | — | 新建 `engagement_scenarios.py`（替代 civilian_scenarios 的军用模式） |
| 式(12)-(14) BiLSTM 意图 | `model.py IntentionEncoder` | 静态分支输入改交战状态；新增 intensity 头（11 档） |
| 式(15) 图 G=(V,E,φ_edg) | `interaction_graph.py` | 边特征换交战描述符；加对抗边类型 |
| 式(16) 节点融合 (h_t; h_s; p̂) | `model.py Forecaster` node_embedding | 输入维度扩展（意图分布+防御机状态） |
| 式(17)-(21) 门控 GAT | `model.py TemporalGraphAttention` | 复用（门控=式 19-20；加性注意力=式 21 变体） |
| 式(22)-(25)+Table I 裁剪 | `model.py` Top-K+crop_progress | 新增候选分支展开 + Cesc/Cenv/Cene 硬掩码 + Λ_crop 软调度（见 §4 D1） |
| 式(26)-(31) 裁剪器细节 | — | 并入裁剪模块 |
| 式(32)-(33) 增量解码 | `model.py` 自回归 GRU | 复用；bound 换过载语义 |
| 式(34)-(35) ADE/FDE | `objective.py displacement_errors` | 原样复用 |
| 式(36)-(41) KL+熵+抖动+分离 | `objective.py attention_terms` 框架 | 框架复用，教师打分函数全换 E/R/T |
| 式(42)-(53) 教师 | — | 新建 `military_teachers.py`（或并入 objective） |
| 式(54)-(55) 总损失 | `objective.py TrajectoryObjective` | 结构一致，系数对齐（λ_sep=0.1, ω=0.4/0.6） |
| Table V 数据流 | `data.py Windows/load_tracks/split` | 复用（含 scene 分组防泄漏） |
| Table III 在线 | `rolling.py RollingPredictor` | 复用，对齐 Δ_Tp |
| 两套协议 | `config.py Config` | 新增 military 配置字段 + 三个 config JSON |
| 四基线 | `baselines.py` | 复用，宽度 256 |
| 消融 I/E | `workflows.py compare` | 四象限复用，C→E 语义 |
| 评估/报告 | `metrics.py/reporting.py` | 复用 + 逃逸触发前后分组 |

---

## 4. 关键工程决策记录

### D1：候选未来分支裁剪的张量接口（论文自相矛盾处的决策）
**矛盾**：式(15) 说节点集 V 是观察窗内 2t₀ 个历史节点（t₀ 目标 + t₀ 防御机）；Table I Step 1 却说"候选目标轨迹
运动学状态取自邻居 j"（邻居=候选未来状态）。直接按字面实现候选分支需为每条历史节点挂未来展开，内存与复杂度爆炸。

**决策（锚定候选法）**：
1. 保留式(15) 的 2t₀ 历史节点主干；
2. 从训练集未来增量构建**锚点池** anchor_pool ∈ R^{A×3}（对训练集所有窗口的未来位移增量做 k-means 或均匀球面采样，A≈16–32）；
3. 观察窗末端目标节点的候选分支 = 末位置 + anchor_pool（经速度/过载可行性预筛），作为**额外候选节点**参与注意力计算（节点数 2t₀+G·A，G 为挂载的末端节点数，默认 G=1 只挂末节点以控制复杂度）；
4. Cesc/Cenv/Cene 在（历史/候选 → 候选）边上计算硬掩码 m(j)；Λ_crop 软调度复用 crop_progress 机制（等价于论文"裁剪强度随训练迭代递增"）；
5. FOV 不可见（防御机出视野）时：Cesc 置 1（只用 Cenv/Cene 交战无关子集），交互边退化为纯外推——对应论文 Step 4 与 IV-B"自然退化"论述；
6. 该决策**保持 demo 现有 Top-K/渐进裁剪框架不变**，候选节点作为 feasible 掩码与附加 entity 的自然扩展。

**记录于代码注释与 README；论文原文表述不精确处以此决策为准。**

### D2：Cenv 地形约束的简化
论文 F 是地形/障碍/禁飞/净空栅格（式 23），但 IV-B 未给出具体地形图。决策：合成简单地形场（平地 + 若干禁飞长方体 +
高度带 [h_min, h_max]），g(y)∉F 判定用于训练期裁剪与数据生成（越界轨迹丢弃）；不追求真实地形。可配置 `terrain_enabled=False`
关闭（退化为只有高度带约束）。

### D3：ṙ_min 与 P_T,t 可达域
论文未给 ṙ_min 数值（Cesc 的逃逸约束阈值）与 P_T,t（机动可达域边界）的精确构造。决策：
- ṙ_min 默认 0。**Cesc 是"可行/逃逸"集合，指示为 1 的候选被保留**：ṙ 取 LOS 距离的时间变化率 ṙ=dr/dt（目标相对防御机），
  逃逸＝分离增大＝ṙ 为正。式(22)/式(27) 的指示 cesc=1[(1/T)Σ_k ṙ_k≥ṙ_min] 对**整个采样窗口的时间平均 ṙ** 判据（第八轮 K3
  已从早先"只判候选末端点"改为全弧段均值，防御机在窗口内按匀速传播）：窗口内平均逼近（ṙ<0）的候选剔除，平均脱离（ṙ≥ṙ_min）
  的保留。早期版本文档曾把方向写反（称"逃逸 ṙ 应为负、ṙ≥ṙ_min 被裁"），与代码及物理符号相反，以此处为准。默认 ṙ_min=0，配置暴露。
- P_T,t 用当前速度与剩余过载能力的时间积分球（V_T·(T−t) + 0.5·a_max·(T−t)² 的可达半径）近似。

### D4：两套协议的窗口不兼容处理
IV-A（dt=0.5, O=20）与 IV-B（dt=0.1, O=80）观察时长同为 10s/8s 量级但采样率不同。决策：数据生成统一用 dt=0.1s 仿真，
IV-A 配置下按 5:1 抽取；同一 npz 可服务两套协议（`Windows` 已支持按 cfg.dt 校验时间戳，需放宽为可抽取模式）。

### D5：11 档强度任务与 9 类语义任务共存
IV-A 两个实验是并列的两个意图空间。决策：`intention_task="semantic9" | "overload11"` 配置项，共享 BiLSTM 主体、
分类头独立；消融与报告分开出。

---

## 5. 实施顺序与逐步验证（每步一个冒烟检查）

按依赖排序，每步完成即跑冒烟，确保"实用性 + 论文贴切"（对应任务清单 Task 3–9）：

| 步 | 内容 | 新增/改动文件 | 冒烟检查 |
|---|---|---|---|
| S1 | 交战数据生成器（Table IV/V：运动学+逃逸+FOV+9类/11档标签） | `engagement_scenarios.py`、`config.py`（新字段） | 生成 100 场景：速度≤350+ε、过载≤9g+ε、r 单调段存在、逃逸触发率>0、标签分布直方图、可复现（同 seed 同 hash） |
| S2 | 交战描述符边特征 + 对抗边 | `engagement_graph.py`（或扩展 interaction_graph） | 单窗手算 r/ṙ/eL/FOV 与函数输出一致；边维度；三类边掩码正交 |
| S3 | E/R/T 教师 | `military_teachers.py`、`objective.py` | 平滑直线轨迹：s_E 高、s_R 低；逃逸段：s_R 上升；μ_t 单调性；教师与观测 detach（无梯度泄漏） |
| S4 | 候选分支裁剪（D1） | `model.py`（候选展开+硬掩码+Λ_crop） | 候选节点计数=A；Cesc/Cenv/Cene 单元测试（构造已知 ṙ/地形/过载）；FOV off 时 Cesc 旁路；注意力行归一化保持 |
| S5 | 意图 11 档任务 + 静态分支换 s_M | `model.py`、`config.py`、`engine.py` | 小数据训练损失下降、混淆矩阵对角带、accuracy>随机基线 |
| S6 | 两套协议 pipeline + xuance 集成 | `configs/engagement_*.json`、`workflows.py`、`cli.py`、`xuance/` 注册 | `run.py check`；`pipeline --config engagement_forecast.json --cpu-smoke`（缩样 epoch=1）端到端跑通；manifest 正确 |
| S7 | 消融 I/E 四象限 + 基线宽度对齐 | `workflows.py`、`baselines.py`、`reporting.py` | compare 8 模型 + 四象限跑通；报告图表生成；逃逸前后分组统计 |
| S8 | 文档更新 + 服务器脚本 | `README.md`、`METHOD_MAPPING.md`、`bash/run_engagement.sh` | 冒烟全绿后写服务器复现实验说明 |

**监测原则**：每步冒烟必须（a）物理量硬校验（速度/过载/FOV 数值断言）；（b）与论文公式手算样例对照至少 1 个；
（c）单元测试进 `tests/test_engagement_*.py`，民用测试保持全绿（回归不破坏）。

---

## 6. 风险与开放问题

1. **内存**：IV-B O=80 → 160 节点 + 候选 A=32 → 192 节点，注意力 192²×6 头×2 层；CPU 冒烟需缩样（cpu_engagement.json 用 O=20/P=10），
   全量跑服务器（bash/run_civilian_interaction.sh 模式改造）。
2. **式(26) 的 OCR 噪声**：初始兼容性分数 q_i^(0)=Id_k+b(rii) 无法完全还原，按式(21) 的加性注意力直接实现（demo 已有），
   差异记录在 METHOD_MAPPING.md。
3. **教师归一化**：ã_req/c̃_r/j̃ 的归一化方式论文只说 "unitization"，决策用训练集统计的 z-score 或 max-归一（配置暴露）。
4. **IV-B 的 80/20 无测试集**：外推实验只有 train/val，test 复用 val（论文报告口径按 val 选 best checkpoint）。已实现：`Config.holdout_test=False` + `split_tracks(holdout_test=)`（验证集取剩余全部场景），`evaluate` 在无测试划分时回落验证集。

---

## 7. 实施状态（2026-10-11，S1–S8 全部完成）

| 步 | 状态 | 主要文件 | 守护测试 |
|---|---|---|---|
| S1 交战数据生成器 | ✅ | `engagement_scenarios.py` | `tests/test_engagement_scenarios.py`（5） |
| S2 交战描述符边 + 对抗边 | ✅ | `engagement_graph.py`（19维） | `tests/test_engagement_graph.py`（4） |
| S3 E/R/T 教师（式42–53） | ✅ | `military_teachers.py`、`objective.py` | `tests/test_military_teachers.py`（5） |
| S4 候选分支裁剪（D1） | ✅ | `candidate_pruning.py`、`model.py` | `tests/test_candidate_pruning.py`（7） |
| S5 11档过载意图任务 | ✅ | `data.py`（`overload_labels` 路由）、`engagement_overload.json` | `tests/test_overload_task.py`（6） |
| S6 两套协议 pipeline | ✅ | `configs/engagement_*.json`、`cli.py simulate-engagement`、`sample_stride` 抽取 | `tests/test_engagement_protocol.py`（4） |
| S7 I/E 四象限消融 + 逃逸分组 | ✅ | `workflows.py compare`、`engine.run_epoch`、`study --data-mode engagement` | `tests/test_engagement_ablation.py`（3） |
| S8 文档与服务器脚本 | ✅ | `README.md`（军用章节）、`METHOD_MAPPING.md`（军用对应表）、`bash/run_engagement.sh` | `run.py check` 两套协议均通过 |

工程补充说明（S4 新增决策，记录在 `candidate_pruning.py` 文档串）：
- 候选块补齐为 O 大小（A 个锚点 + 掩码填充），与现有实体块结构、教师 reshape、解码记忆掩码保持一致；
- 锚点 0 固定为直飞方向，保证候选集非退化；
- 三约束全不满足时保留代价最小分支（恒速退化），避免候选块整体消失；
- 软裁剪以 destination 惩罚形式接入注意力 logits，与 `crop_progress` 共用渐进调度。

**验证口径**：本机 CPU 全量测试 74 passed（军用 34 + 民用 40 回归）；`run.py check --backward` 在 IV-A（dt=.5/O=20）与 IV-B（dt=.1/O=80/P=60，含候选裁剪）两套结构下均通过；完整规模实验需在服务器执行 `bash/run_engagement.sh`。
5. **ω₁ADE+ω₂FDE 的不确定性**：σ_sup 单独可学习（demo 已有 4 维 log_variance，第 4 维即轨迹项，语义对齐）。
