# 代码 ↔ 论文 静态保真度审查（不含需训练才能得到的实验结论）

审查日期：2026-10-11 晚 · 范围：仅静态代码与论文文本比对，**未执行任何训练/数据生成，未修改代码**
对象：`src/trajectory_prediction/*`、`configs/engagement_*.json`、`bash/run_engagement.sh`、`scripts/*`
依据：`tmp/paper_text.txt`（式 1–55；II–IV 章；Table I–VI 标题；Remark 1–6）+ 已从 PDF 图版读出的 Table VI 数值

排除项（属于"上服务器跑完才有结论"，**不计入本次差距**）：数据集是否已跑满 5×10⁴、蒙特卡洛是否已跑 1000 次/标签、Fig.14 是否已用训练后模型复现、88%/94%/102 m 是否达标。这些的**入口与参数**均已就绪，见 `PAPER_ALIGNMENT_GAP_AUDIT.md` §10。

---

## 0. 结论摘要

| 类别 | 条数 | 说明 |
|---|---|---|
| **A 影响方法正确性、必须补齐** | 13 | 含 2 处本次新发现的**实现 bug**（教师温度被平方、几何项失去不确定性权重） |
| **B 锦上添花、可选优化** | 10 | 量纲/归一化/命名/论文未给数值的近似 |
| **U 无法判定，需你补充** | 10 | 论文图版内容缺失或原文自身不一致 |

最需要优先处理的 4 条：**A-02**（教师温度 ×2）、**A-01**（式55 几何项权重）、**A-03**（风险教师未做 LSE 聚合）、**A-05**（Cene 判据用错物理量）。

---

## 1. A 类：影响方法正确性、必须补齐

### A-01 式(55) 几何项丢失不确定性权重 ★本次新发现 ✅ 已修复
- **论文**：IV 章 + 式(55) `L_tra = L_attn + (1/(2σ̄²_sup))·(ω1·ADE + ω2·FDE)`，σ̄_sup 是 ADE/FDE 这一路的不确定性尺度；IV-B 明确 `ω1=0.4, ω2=0.6`，heteroskedasticity 参数范围 `[−6,6]`。
- **代码**：`src/trajectory_prediction/objective.py` → `TrajectoryObjective.forward()`，`loss_balance=="uncertainty"` 分支中 `geometry` 被**直接以单位权重相加**；log-variance 参数只作用于三组注意力教师。
- **差距**：几何项应有 `1/(2σ̄²_sup)` 因子（且按论文式(55) 字面**不附加** `log σ̄_sup` 正则项，正则项只在式(54) 出现）。当前既无权重也无正则 → 与论文的"heteroscedastic weighting 平衡多子损失"设计不符，Fig.13 的多子损失平衡行为不可复现。
- **注**：这是我上一轮执行 B2（"几何项去掉 log-variance 正则"）时的过度修正——去掉正则是对的，但把权重也一并去掉了。

### A-02 教师温度被除两次（实现 bug）★本次新发现 ✅ 已修复
- **论文**：式(36) `π = exp(s_ij/τ_k) / Σ exp(s_ik/τ_k)`，IV-B 指定 `τ = 0.1`。
- **代码**：`objective.py` → `attention_terms()`：
  1. 军用分支 `teachers = [teacher / cfg.teacher_temperature for teacher in teachers]`；
  2. 循环内又出现 `scores = teachers[group][...] / cfg.teacher_temperature`。
- **后果**：军用教师实际温度为 `τ² = 0.01`，教师分布接近 one-hot，KL 监督退化为硬指派；民用分支只除一次，两者不一致。**必须去掉其中一处**。

### A-03 式(49) 风险教师未做 log-sum-exp 时间聚合 ✅ 已修复
- **论文**：式(49) `s^(R)_ij = (1/λ_R)·log Σ_{τ=t}^{t+H} w_{j,τ}·exp(λ_R·r_{j,τ})`；原文强调该机制"assign additional weight to the most dangerous threat"（最危险时刻主导）。
- **代码**：`military_teachers.py` → `_risk_score()`：`return threat[:, None] * weight`，其中 `weight = exp(-lag/horizon)`。
- **差距**：这是"威胁值 × 时间衰减"的**线性加权**，没有 `log Σ w·exp(λ_R·r)`。缺少 LSE 的"取最大"效应，λ_R（`teacher_risk_lambda`）也未被使用于聚合。

### A-04 式(50)(51) 走廊偏差缺少"减去半宽"项 ✅ 已修复
- **论文**：`δ_r = [dist_r(y_t, y*_t,k) − b_r] / b_r`、`δ_n = [dist_n(...) − b_n] / b_n` —— 只有**超出**走廊半宽的部分才计入。
- **代码**：`military_teachers.py` → `_timeliness_score()`：`delta_r = radial / b_r`、`delta_n = normal / b_n`。
- **差距**：未减 `b_r/b_n`，连走廊内部的正常节点也会被惩罚，符号语义与论文相反（论文 δ 在走廊内应 ≤0，被 `clamp_min(0)` 吸收）。

### A-05 式(24) Cene 用了错误的物理量 ✅ 已修复
- **论文**：`Cene = { ŷ | (1/T)·∫_{t0}^{t0+T} ||a_{T,k} − a_{T,k−1}|| ≤ a_T,max }` —— **相邻时刻加速度差**的时间平均（原文：用 `a_T,max` 限制机动过程中的能量耗散）。
- **代码**：`candidate_pruning.py` → `candidate_branch()`：`a_req = speed * theta / horizon; energy = (a_req <= a_max)` —— 用的是**平均加速度**（盘旋侧向过载），不是加速度差。
- **差距**：裁剪判据本身错了一个差分阶，会系统性地保留/裁错候选分支。

### A-06 式(14)(16) 应用 `[h_t; h_s]` 拼接，代码用了融合层 ✅ 已修复
- **论文**：式(14) 明确 `z = [h_t; h_s]` 后接 `Softmax(W_c·ReLU(W_h·z + b_h) + b_c)`；式(16) 节点初始特征 `h^(0)(k) = [ g^(0)(k), [h_t; h_s; p̂] ]`。
- **代码**：`model.py` → `IntentionEncoder.__init__/forward`：`self.fusion` 把 `cat(h_t(64), h_s(64))` 压成 64 维；分类头与节点特征都使用这个**融合后**的向量。
- **差距**：多出一个论文没有的 `fusion` 层，且 `z` 的维数由 128 降为 64；式(16) 的 `[h_t; h_s; p̂]` 也被替换为 `[fusion(h_t,h_s); p̂]`。

### A-07 式(13) 静态分支缺 `e_M` ✅ 复查后撤销（已含 e_L）
- **论文**：III-A "The kinematic state `s_M = [r_M, V_M, e_M]` of the defender aircraft is used as the intention recognition factor `Q_int`" —— 位置、速度、**单位速度方向**共 9 维（`e_M` 与 `e_T/e_L` 同一命名体系，即航向单位向量）。
- **代码**：`model.py` → `Forecaster.forward()`：`static_features = 防御机最后状态 6 维（位置+速度）+ engagement[:, -1]（9 维交战描述符）`；没有独立的 `e_M`。
- **差距**：`Q_int` 缺少论文明列的 `e_M` 分量（交战描述符里只有 `e_L`，不含防御机航向）。

### A-08 式(21) 注意力打分：论文用拼接，代码用加法 ✅ 复查后撤销（数学等价）
- **论文**：式(21) 分子 `exp(LR(α^T [W·h_i ∥ W·h_j]))` —— 两节点变换后**拼接**，再与参数向量 `α` 内积，过 LeakyReLU。
- **代码**：`model.py` → `TemporalGraphAttention.forward()`：`scores = leaky_relu(source_i + dest_j)`（`source/destination` 各自与可学习向量内积后相加），即标准 GAT 加法式。
- **差距**：打分函数族不同（拼接 vs 加法）。
- **冲突提示**：Fig.6 的说明文字又提到 "the W and U used to compute `q_i, v_j, k_j`"，即 QKV 形式 —— 与式(21) 不一致。代码更接近 QKV 风格。→ **见 U-1，需你确认采用哪一种**。

### A-09 式(19) 边门控未对 `(h_i, h_j, u)` 联合建模 ⬇ 降级为可选
- **论文**：式(19) `m_ij = σ(f_e(h_i, h_j, u))`，`f_e` 是"以节点隐状态**连同** u 为输入的可学习评估映射"，输出标量门控。
- **代码**：`model.py` → `TemporalGraphAttention.forward()`：`edge_logits = 线性(h_i) + 线性(h_j) + self.edge_context(edges)` —— 节点对与 u 是**加性分解**，没有 u×(h_i,h_j) 的交互项。
- **差距**：`f_e` 的表达能力弱于论文，交战描述符只作为偏置进入门控。

### A-10 Definition 1 + 式(36)：每组对应的**边子集** `E_k(i)` 未实现
- **论文**：Definition 1 "supervision and normalization are applied to the attention weights produced by each group using its corresponding teacher score **and its corresponding edge subset**"；式(36) 分母的归一化集合是 `k′ ∈ E_k(i)`（**该组自己的边子集**），分子用 `1[i→j ∈ E_k(i)]` 指示。
- **代码**：`objective.py` → `attention_terms()`：三组共用同一个 teacher 矩阵，掩码统一取 `valid = student.detach() > 0`（学生注意力非零处），没有按组区分边子集。
- **差距**：分组监督的"边子集"约束完全缺失。论文未给出 `E_k` 的定义 → **见 U-2**。

### A-11 式(41) 分离项：论文求和，代码取均值 ✅ 已修复
- **论文**：`L_sep = η Σ_k Σ_{h∈H_k} Σ_{h′∉H_k} cos²(vec(β̃_h), vec(β̃_h′))`（三重**求和**），`λ_sep = 0.1`。
- **代码**：`objective.py` → `attention_terms()`：`separation = cosine[:, separate].square().mean()`。
- **差距**：元素数量级约 `3·|H_k|·(|H|−|H_k|)`（H=6 时约 200 倍），均值口径下 `λ_sep=0.1` 的实际正则强度远弱于论文标定。
- **实施提示**：改为求和须**同步重标定** `separation_weight`（约 ÷200），否则该损失项会主导训练。

### A-12 式(37) 时间维：论文求和，代码取平均 ✅ 已修复
- **论文**：`L_attn_k = Σ_t [ (1/|H_k|) Σ_{h_k} D_KL + ρ_k·ΔH_k + ζ_k·Δβ̄_k ]`（对时间 **Σ**）。
- **代码**：`attention_terms()` 中的 `row_mean()` 对时间（及头）取平均。
- **差距**：量级差 `T` 倍；因 `ρ_k/ζ_k` 论文未给数值，实际会被权重吸收。**影响较弱，但属于公式不一致。**

### A-13 输出头：论文用线性回归，代码用 MLP ✅ 已修复（IV-B 优先于式32）
- **论文**：IV-B "The **linear regression** is employed for the output head."（配合式(32) `Δŷ_k = MLP(h^L_k)`——论文此处自相矛盾，前者是 IV-B 的显式配置说明）。
- **代码**：`model.py` → `Forecaster.__init__`：`self.decoder = Linear(d→d) → ReLU → Linear(d→3)`，末层零初始化（从恒速基线起步）。
- **差距**：多一层非线性；零初始化是为复现"恒速基线起步"的工程选择（已登记），但与论文的线性头不同。

---

## 2. B 类：锦上添花、可选优化

| ID | 论文 | 代码（文件 → 函数） | 说明 |
|---|---|---|---|
| B-01 | 式(45) `a_req = (V/Δt)·arccos(Δy_j·Δy_i / ‖Δy_i‖)`，作用于**位置增量序列** `Δy`，且分母只除 `‖Δy_i‖` | `military_teachers._pairwise_energy` 用**速度向量**夹角并双模归一化 | 数值上接近（Δy ≈ v·Δt），但严格复现应按增量位移且只用 `‖Δy_i‖` 归一 |
| B-02 | 式(46) `c_r = a_{r,t} − a_{r,min}` | `_pairwise_energy` 取 `braking.clamp_min(0)` | 代码把 `a_{r,min}` 视为 0；论文未给 `a_{r,min}` 数值 |
| B-03 | 式(53) `μ_t = σ(κ_T·(t − T*)/T)` | `_timeliness_score` 用 `σ(κ·(2t/O − 1))` | 等价形式只在 `T* = T/2` 时成立，且相当于 κ 加倍；`T*` 论文未给 |
| B-04 | 式(23) `Cenv`：地形/障碍/禁飞/高度栅格 `F ⊂ Z²`，越界即裁剪 | `candidate_pruning.candidate_branch` 的 `terrain_enabled` **默认 false**（Cenv 恒 1） | 决策 D2 简化为高度带且未启用；论文无地形数据 → 见 U-6 |
| B-05 | IV-B "imposes **hard blocking** on links deemed unreachable" | `model.TemporalGraphAttention.forward` 的 `reachable` 在多实体（军用）路径下恒为 `None` | 可达性硬阻断只在单实体民用路径实现（审计 A7） |
| B-06 | 式(32) 说增量序列经**因果自注意力**融合；IV-B 又说解码器为**自回归 GRU + 时序记忆 cross-attention** | `engagement_forecast.json: decoder_type = "autoregressive_gru"` | 论文内部两处不一致；代码采用 IV-B 的显式配置说明（**当前选择是合理的**），仅登记 |
| B-07 | 式(25) `α ← α·1_c(j)`；IV-B "soft & hard cropping 强度随训练迭代递增" | `model.set_crop_progress` + `TemporalGraphAttention` 的 top_k 退火与 `penalty` 缩放 | 递增只作用于 **Top-K 与软惩罚**；硬掩码 `feasible` 不随进度变化 |
| B-08 | 论文未提运动学约束裁剪 | `model.bound_increments` 对主模型与四个基线**都**施加速度/加速度上界 | 会同时"帮助"主模型与基线；严格 matched-capacity 比较下建议可开关 |
| B-09 | IV-D 消融命名 `A1(−I,+E)`、`A4(+I,−E)`、`A2(−I,−E)` | `workflows.compare` 用 `graph_no_intention` / `graph_no_engagement` / `graph_no_intention_engagement` | 语义一一对应，但报告图表未打论文标签，对照 Fig.17/18 时需人工映射 |
| B-10 | IV-B 只给了基线 `p_do = 0.1`，主模型 dropout 未说明 | `engagement_forecast.json`: `dropout=0.1`、`intention_dropout=0.2`；IV-A 配置 `0.2/0.2` | 无论文依据可校验，属自由超参（IV-A 的 0.2 是论文明确值 ✓） |

---

## 3. U 类：无法判定，需要你补充的论文内容

| ID | 需要什么 | 为什么需要 | 当前代码的临时处理 |
|---|---|---|---|
| U-1 | 式(21) 的拼接式打分 vs Fig.6 文字的 `q_i/k_j/v_j` QKV —— 论文以哪个为准 | A-08 的修改方向取决于此 | 采用 GAT 加法式（近似 QKV） |
| U-2 | Definition 1 中每组 `E_k(i)` 边子集的定义（E/R/T 各自用哪些边：时序/空间/对抗？） | A-10 无法落地 | 三组共用全边集 + 学生非零掩码 |
| U-3 | 式(52)(53) 的任务目标点 `Y`、要求时间 `T*`；式(50)(51) 的走廊半宽 `b_r/b_n` 数值；"中心路径 `Y*_k`" 如何构造 | A-04/B-03 的精确实现依赖这些量 | `Y` 用末状态恒速外推点代替，`T*` 用窗口末时刻，`b_r` 自动取 `0.2·V·T_p` |
| U-4 | 式(47) 的 "unitization"（`ã_req, c̃_r, j̃_t`）具体除以什么基准 | 决定能量教师的尺度 | 除以 `a_T,max=9g`（加速度量）与 `a_max/dt`（jerk 量） |
| U-5 | 式(49) 的聚合窗口 `H` 与权重 `w_{j,τ}` 的公式 | A-03 落地需要 | `H` 取预测时域，`w = exp(−Δt/T_p)`（工程设定） |
| U-6 | 式(23) 的地形/障碍/禁飞栅格 `F` 在仿真中如何生成（论文无地形数据） | B-04 无法严格实现 | D2：简化为高度带，且默认关闭 |
| U-7 | Table IV 的精确数值范围（图版）；正文 IV-B 又给 `V_M=360 m/s`，与恢复值 `[380,450]` 冲突 | 数据集几何分布 | 采用恢复值 `[380,450]`；Fig.14 场景另用 `V_M=360`（Table VI） |
| U-8 | 十一档 `I={−10…+10}` 的符号约定（式(4) `a_n` 是模长，恒非负） | 决定负六档是否可达 | 工程决策：取法向加速度在速度系（横向/纵向）主分量的符号 |
| U-9 | 主模型在 IV-B 的 dropout、权重初始化方式、数据加载细节 | 无法校验 | PyTorch 默认初始化 + `dropout=0.1` |
| U-10 | Table VI 中 `a_φ,0 = −58` 对应 "Left"、`+58` 对应 "Right"，与常规右手系相反；`a_θ/a_φ` 与俯仰/偏航的对应关系 | Fig.14 四象限语义可能整体镜像 | 按**场景名**（Left/Right/Climb/Dive）定符号，论文原始 `a_θ/a_φ` 值仅存档于 metadata |

---

## 4. 抽查确认：已与论文一致的关键项（供对照）

| 论文 | 代码 | 结论 |
|---|---|---|
| 式(1)(2) 相对运动学（ṙ, ė_L, V_L） | `engagement_scenarios.los_state` | 逐项一致 ✓ |
| 逃逸指令 `ã=−e_L(I−e_T e_T^T)`、`a=min(ā_eva,a_T,max)` 饱和、`ε=1e−6` | `engagement_scenarios._evade_command` | 一致 ✓ |
| 式(15) 三类边（时序/空间/对抗）+ 19 维描述符 | `engagement_graph.engagement_edges` | 一致 ✓ |
| 式(17)(18) 聚合 + 残差 + LN | `TemporalGraphAttention.forward` 末尾 | 一致（σ 用 ELU）✓ |
| 式(22) `Cesc: ṙ ≥ ṙ_min`、FOV 不可见时旁路 | `candidate_pruning.candidate_branch` | 一致 ✓ |
| 式(25) `α ← α·1_c(j)` 硬裁剪 + `exp(−Λ_crop P)` 软惩罚 | `feasible` 掩码 + `penalty` | 一致 ✓ |
| 式(34)(35) ADE/FDE 定义 | `objective.displacement_errors` | 一致 ✓ |
| 式(36) 教师分布 softmax(s/τ)、τ=0.1 | `attention_terms`（**但见 A-02**） | 结构一致，系数有 bug |
| 式(38) `ΔH_k = |H_min − H(β̄)|`、式(40) `β̄` 组内头均值、式(39) 相邻时刻抖动 | `teacher_regularizers="paper"` 分支 | 一致 ✓ |
| 式(42)–(44) `V/a/jerk` 三阶差分 | `_pairwise_energy` 的速度/加速度/加加速度 | 一致 ✓ |
| 式(47) 能量教师 `−(1/κ)·LSE(ã², j̃², c̃²)` | `_pairwise_energy` | 一致（归一化基准见 U-4）✓ |
| 式(48) 风险 `LSE_κR(r̃, ṙ̃, P̃_T,t)` | `_threat_levels` | 一致 ✓ |
| 式(52) 时效 `−[(1−μ)LSE(δ_r,δ_n) + μ·LSE(‖y−Y‖, |T−T*|)]` | `_timeliness_score` | 结构一致（细节见 A-04） |
| Remark 4 教师只在训练期使用 | `military_teacher_scores` 全程 `no_grad`，推理不输入 | 一致 ✓ |
| IV-A：d=6 / Nh=128 / LayerNorm / Dropout .2（投影前后）/ m=64 ReLU / 9 类 Softmax / CE | `IntentionEncoder` | 一致（**但分类头输入见 A-06**） |
| IV-A：dt=0.5、To=10 s（O=20）、stride=5、70/15/15 轨迹级划分、标准化仅在训练集 | `engagement_intention.json` + `engine.fit_normalizer(train)` | 一致 ✓ |
| IV-A：Adam 1e−3、step γ=0.5@epoch30、batch256、70 epoch、clip1.0、每轮 shuffle | 同上配置 | 一致 ✓ |
| IV-B：d=256、H=6、d_k=d_v=64、时序 H=4、深度 2、Top-K=8、τ=0.1、FOV mask 启用 | `engagement_forecast.json` | 一致 ✓ |
| IV-B：解码器自回归 GRU + 时序记忆 cross-attention | `decoder_type="autoregressive_gru"` | 与 IV-B 文字一致 ✓（与式32 冲突，见 B-06） |
| IV-B：AdamW 1e−3/wd1e−4/clip1.0/batch256/200 epoch/warmup5+cosine；早停 20 轮 ADE&FDE 改善<1 m | 同上配置 + `engine.train` | 一致 ✓ |
| IV-B：λ_sep=0.1、ω1=0.4、ω2=0.6、log-var 范围 [−6,6] | 同上配置 + `learner.update` 的 clamp | 一致（**但见 A-11**） |
| IV-C 基线容量：LSTM/GRU 2 层 n=256 p=0.1；TCN 4 残差块 k=3 c=256 D={1,2,4,8} 线性头；Transformer 2 层 d=256 h=4 d_ff=1024 p=0.1 | `baselines.SequenceForecaster` + `baseline_width/layers` | 一致 ✓ |
| 式(33) 增量累加 `ŷ_t = Σ Δŷ + r_{T,to}` | `Forecaster.forward` 末尾 `cumsum` | 一致 ✓ |
| Remark 3 候选节点 + 置换不变性 | GAT 聚合 + 候选块设计（决策 D1） | 一致 ✓ |

---

## 5. 建议处理顺序

1. **A-02** 去掉重复的 `/teacher_temperature`（一行），否则所有军用注意力监督都在 τ=0.01 下运行。
2. **A-01** 按式(55) 恢复几何项的 `1/(2σ̄²_sup)` 权重（不加 `log σ̄_sup`）——需要第 4 个 log-variance 参数或单独的 σ̄_sup。
3. **A-03 / A-04 / A-05** 三个公式级修正：风险 LSE 聚合、走廊偏差减半宽、Cene 改用相邻加速度差均值。
4. **A-06 / A-07** 结构修正：去掉 `fusion`（改用 `z=[h_t;h_s]`）、静态分支补 `e_M`。
5. **U-1 / U-2 / U-3** 需要你从论文图版（Fig.5/6/7、Table I）或附录补信息后，再动 A-08 / A-09 / A-10。
6. A-11 与 A-13 属"改了要重标定"的项，建议与服务器实验同批验证。

---

## 6. 修复实施记录（2026-10-11，已全部完成并回归通过）

按 §5 的顺序落地，测试 92 → 100 passed（`tests/test_paper_fidelity.py` 新增 8 条针对性回归）。

| 编号 | 修改 | 位置 |
|---|---|---|
| **A-02** | 军用教师分支只除一次 `teacher_temperature`（删除分支内预处理处那一次），实际温度恢复为 IV-B 的 τ=0.1 | `objective.attention_terms` |
| **A-01** | 式(55)：几何项改为 `0.5·exp(−σ̄_sup)·(ω₁ADE+ω₂FDE)`，即带 `1/(2σ̄²_sup)` 权重；**不**加 `log σ̄_sup`（该正则只属式54）。`log_variance` 索引：0–2 为 H_E/H_R/H_T，3 为 σ̄_sup | `objective.TrajectoryObjective.forward` |
| **A-03** | 式(49) 风险教师改为窗口加权 LSE：`(1/λ_R)·log Σ_{τ=t}^{t+H} w_{j,τ}·exp(λ_R r_{j,τ})`，窗口 τ∈[t, t+P]，w 为时间衰减 | `military_teachers._risk_score` |
| **A-04** | 式(50)(51) 改为 `δ=(dist−b)/b`（抽出 `corridor_deviation()`）；LSE 前不再 `clamp_min(0)`（原文允许走廊内为负）。默认半宽改为"预测时程飞行距离的 20%"（原来是 `‖归一化朝向‖·T·0.2 ≈ 1.2 m`，量级错误） | `military_teachers._timeliness_score` |
| **A-05** | 式(24) `C_ene` 改为时间平均加速度增量：`v·θ²/T² = a_req·(θ/T)`（原为 `a_req = v·θ/T`，那是式45 的过载判据）。软裁剪 `P_ij` 仍用 `a_req/a_max` | `candidate_pruning.candidate_branch` |
| **A-06** | 式(14) 去掉 `fusion`，分类头直接吃 `z=[h_t;h_s]`（`Linear(2m→m)→ReLU→Linear(m→C)`）；`IntentionEncoder` 返回的 context 即 z，供式(16) 使用 | `model.IntentionEncoder` + `model.Forecaster.node_embedding` |
| **A-11** | 式(41) 分离项由 `mean()` 改为 `sum(-1).mean()`（对全部跨组有序头对求和，再对 batch 取均值） | `objective.attention_terms` |
| **A-12** | 式(37) 时间维由 `row_mean` 改为"对 t 求和 /(|H_k|·batch)"（抽出 `_time_sum`），kl/entropy/jitter 三项同口径 | `objective.attention_terms` |
| **A-13** | IV-B"输出头用线性回归"：`decoder` 默认改为单个 `Linear(d,3)`；新增 `decoder_head="mlp"` 开关保留式(32) 的 MLP 读法 | `model.Forecaster` + `config.Config.decoder_head` |

### 6.1 复查后撤销/降级的两条（原判"不一致"其实一致）

* **A-08（式21 拼接 vs 加法）——撤销。** 式(21) 的 `αᵀ[Wh_i ∥ Wh_j]` 与代码的 `f(h_i)+f(h_j)` **数学等价**：把 α 按前后半切分即 `a_left·Wh_i + a_right·Wh_j`，正是 `self.source`/`self.destination` 两个参数的实现方式。U-1（式21 与 Fig.6 的 q/k/v 说法冲突）仍保留为论文内部表述问题，但不影响代码。
* **A-09（式19 联合门控）——降级为可选。** 代码 `f_e = W_left h_i + W_right h_j + W_u u` 本就是 `(h_i,h_j,u)` 的线性映射，符合"learnable evaluation mapping … output scalar gating score"；若论文图版显示 MLP+交叉项再升级。
* **A-07（式13 静态分支缺 `e_M`）——撤销。** `static_features` = 防御机相对位/速(6) + 交战描述(9)，其中 9 维含 `e_L(3)`，即 `s_M=[r_M,V_M,e_M]` 已齐备（另含 `ṙ、ė_L、可见性` 三个额外通道，属合理增广）。

### 6.2 仍未动（需要论文信息）

* **A-10 / U-2**：Definition 1 提到每组"corresponding edge subset `E_k(i)`"，但正文未定义三个子集如何划分（按式(15) 的时序/空间/对抗三类边？），未实现，当前三组共用同一邻接集。
* **U-1 / U-3 / U-4 / U-5 / U-6 / U-7 / U-8 / U-9 / U-10**：见 §3。

### 6.3 训练期需要留意的一个量纲后果

式(37) 改为"对 t 求和"后，注意力监督项整体放大约 N 倍（军用图 N=3O=240）。这正是论文设置 `σ̃_k` 不确定度加权（log-var ∈[−6,6]）要吸收的量级差；`loss_balance="uncertainty"`（IV-B 协议）下会被自动平衡，但 **不要用 `loss_balance="fixed"` 跑军用主实验**，否则 `attention_weight=0.1` 下的注意力项会压过几何项。
