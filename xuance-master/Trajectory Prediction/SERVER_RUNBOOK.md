# 服务器训练手册：军用拦截交战版（IV-A / IV-B）

本手册给出**可直接上服务器执行**的完整流程。所有命令都在仓库根 `xuance-master/` 下运行。

---

## 0. 交付的代码套

```
xuance-master/
├─ Trajectory Prediction/
│  ├─ run.py                     # 唯一入口（generate/simulate/train/pipeline/evaluate/study/predict/report/check）
│  ├─ configs/                   # 四份军用配置
│  │   ├─ engagement_intention.json   # IV-A 九类语义意图
│  │   ├─ engagement_overload.json    # IV-A 十一档过载强度
│  │   ├─ engagement_forecast.json    # IV-B 轨迹外推（主力）
│  │   └─ cpu_engagement.json         # 冒烟自检（小容量，勿用于正式实验）
│  ├─ scripts/
│  │   ├─ prepare_server.py           # 打包导出（源码，不含数据/环境）
│  │   ├─ monte_carlo_intention.py    # IV-A 每标签 N 次独立蒙特卡洛
│  │   ├─ reproduce_figure14.py       # Table VI 四组合场景
│  │   └─ check_paper_targets.py      # 四项论文锚点对照
│  └─ src/trajectory_prediction/      # 实现
├─ bash/
│  ├─ setup_server.sh            # 建环境 + 自检
│  ├─ run_engagement.sh          # 主实验入口（4 个协议）
│  └─ run_paper_targets.sh       # 论文数值复现（训练后）
```

---

## 1. 环境准备（约 10 分钟）

### AutoDL（推荐路径）

AutoDL 实例的系统盘很小，**把代码和输出都放到数据盘**（`/root/autodl-tmp`），否则 5×10⁴ 窗口的数据集 + checkpoint 会撑爆系统盘：

```bash
cd /root/autodl-tmp && tar xzf server_package.tar.gz && cd xuance-master
bash bash/setup_server.sh .venv auto       # 自动探测预装 torch 的 CUDA 版本
source .venv/bin/activate
```

`auto` 会读取镜像里预装 torch 的版本选 cu121/cu124/cu126；裸镜像上有 `nvidia-smi` 时默认 cu124。显式指定同理：`bash bash/setup_server.sh .venv cu124`。

### 通用服务器

```bash
cd xuance-master
bash bash/setup_server.sh .venv cu121        # GPU（CUDA 12.1）；CPU 机器用 cpu
source .venv/bin/activate
```

脚本会：建 venv → 装 `torch==2.7.1+cu121` → 装 `requirements.txt` → 跑 `run.py check --backward` 冒烟 → 打印磁盘空间与数据盘提示。

手工等价步骤（若不用脚本）：

```bash
python -m venv .venv && source .venv/bin/activate
pip install "torch==2.7.1+cu121" --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r "Trajectory Prediction/requirements.txt"
export PYTHONPATH="$PWD/Trajectory Prediction/src:$PWD"
```

> `PYTHONPATH` 必须同时含 `src`（模块）和仓库根（xuance 包）。`run.py` 会自动补，但直接调 `python -m` 时要手动设。
>
> AutoDL 学术加速：下载 torch 前可 `source /etc/network_turbo`，装完 `unset http_proxy https_proxy` 再训练，避免训练流量走代理限速。

---

## 2. 训练总览

四个协议，共用同一份生成器（内部按 `dt/stride` 网格传播，故 IV-A 的 0.5 s 与 IV-B 的 0.1 s 同源）：

| 协议 | 论文 | 任务 | 配置 | 关键超参 | 验收锚点 |
|---|---|---|---|---|---|
| `intention` | IV-A | 九类语义意图识别 | `engagement_intention.json` | BiLSTM Nh=128, m=64, dropout 0.2, Adam 1e-3, γ=0.5@30 epoch, **70 epoch**, batch 256, clip 1.0 | **≈88%**（test） |
| `overload` | IV-A | 十一档过载强度 | `engagement_overload.json` | 同上，输出 11 类 | **≈94%**（test） |
| `forecast` | IV-B | 轨迹外推 | `engagement_forecast.json` | GAT d=256/H=6/dk=dv=64/2 层，时间层 H=4，τ=0.1，λsep=0.1，ω1=0.4，ω2=0.6，AdamW 1e-3/wd 1e-4，warmup 5 + cosine，**200 epoch**，早停 both 1 m×20 | **ADE/FDE ≈102 m** |
| `ablation` | IV-B | I/E 四象限 + 基线 | 同上 | 3 数据种子 × 3 模型种子 | 消融增益方向 |

统一入口：

```bash
bash bash/run_engagement.sh OUTPUT_DIR DEVICE PROTOCOL [--resume]
```

`--resume` 可断点续跑。每次运行自动记录：控制台全文日志（`logs/<protocol>_<时间戳>.log`，SSH 断线也不丢）与 `run_manifest.json`（每个协议的开始/结束时间、所用配置、日志路径、状态）。

### OUTPUT_DIR 输出目录结构（复盘分析入口）

```
OUTPUT_DIR/
├─ run_manifest.json              # 全部运行的时间/配置/日志/状态索引
├─ data/
│  ├─ engagement_<protocol>.npz          # 数据集（可复用，重跑训练不必重生成）
│  └─ engagement_<protocol>.metadata.json # 生成参数、窗口数、标签分布
├─ logs/
│  └─ <protocol>_<YYYYmmdd_HHMMSS>.log   # 该次运行的控制台全文（含逐 epoch JSON）
├─ run/<protocol>/
│  ├─ best.pt                     # 验证最优 checkpoint（评估/复现都用它）
│  ├─ last.pt                     # 最新 epoch（断点续跑用）
│  ├─ config.json                 # 本次训练的完整生效配置
│  ├─ environment.json            # torch/CUDA/设备信息
│  ├─ manifest.json               # 数据集指纹与划分
│  ├─ history.csv                 # 逐 epoch 训练/验证指标（画学习曲线直接用）
│  ├─ initial_validation.json     # 训练前基线
│  ├─ training_summary.json       # 结束 epoch/早停/最优分数/总耗时
│  └─ <split>_metrics.json        # 评估指标（ADE/FDE/分组/混淆矩阵）
│                                 #  + <split>_metrics_classes.csv（分类协议）
├─ ablation/                      # 消融协议：每个模型×种子一个子目录，同上结构
└─ paper_targets/                 # run_paper_targets.sh 产物
   ├─ intention_mc.json / overload_mc.json   # IV-A 蒙特卡洛准确率
   ├─ figure14/                              # 四场景轨迹图 + figure14_summary.json
   └─ paper_targets_report.json              # 四项锚点 vs 论文值对照（复盘主入口）
```

> **复盘建议顺序**：`run_manifest.json`（跑了什么、何时）→ `run/<p>/training_summary.json` + `history.csv`（收敛与早停）→ `<split>_metrics.json`（分组误差）→ `paper_targets/paper_targets_report.json`（与论文对照）。

---

## 3. 数据集生成

脚本自动完成；如需单独生成（例如换种子重造）：

```bash
# IV-B（dt=0.1 s，O=80，P=60，5×10^4 窗口）
python "Trajectory Prediction/run.py" simulate-engagement \
  --output data/forecast.npz --scenes 60 --steps 400 \
  --dt 0.1 --sample-stride 1 --aT-max-g 9 --aM-max-g 15 --fov-deg 60 \
  --initial-fov-coverage 1.0 --target-windows 50000 \
  --observation-steps 80 --prediction-steps 60 --window-stride 5
```

关键参数与论文依据：

| 参数 | 值 | 论文依据 |
|---|---|---|
| `--aT-max-g` / `--aM-max-g` | 9 / 15 | Table IV 过载限制 |
| `--fov-deg` | 60 | 目标 FOV 半角 |
| `--initial-fov-coverage` | IV-A 0.5，IV-B 1.0 | IV-A 要求 FOV 内外都覆盖；IV-B 初始化在 FOV 内 |
| `--target-windows` | 50000 | "dataset comprises 50 000 instances" |
| `--evasion-overload-range` | 默认 `0.5,1.0` | IV-B ā_eva 段内有界采样（已实现，无需显式传） |
| `--intercept-radius-m` / `--miss-grow-steps` | 默认 50 / 30 | 拦截或确定错过即终止（完整交战段） |

> **重要**：第六/七轮改了生成器（逃逸强度段内重采样、交战终止、边特征维度 19→21），**必须重新生成数据集并从头训练**，旧 `.npz` 与旧 checkpoint 不兼容。

---

## 4. 逐协议训练命令

```bash
# ---------- IV-A 九类语义意图（70 epoch，最快）----------
bash bash/run_engagement.sh outputs/military cuda:0 intention

# ---------- IV-A 十一档过载强度（70 epoch）----------
bash bash/run_engagement.sh outputs/military cuda:0 overload

# ---------- IV-B 轨迹外推（200 epoch，主力实验）----------
bash bash/run_engagement.sh outputs/military cuda:0 forecast

# ---------- IV-B 消融（I 轴 × E 轴四象限 + 基线）----------
bash bash/run_engagement.sh outputs/military cuda:0 ablation
```

可调环境变量（不传则用默认）：

```bash
TRAJECTORY_TARGET_WINDOWS=50000   # 目标窗口数（默认 5e4）
TRAJECTORY_ENGAGEMENT_SCENES=60   # 场景数下限
TRAJECTORY_ENGAGEMENT_STEPS=400   # 每场景步数
TRAJECTORY_ENGAGEMENT_SEED=42     # 数据种子
TRAJECTORY_DATA_SEEDS=101,102,103 # 消融：数据种子
TRAJECTORY_MODEL_SEEDS=42,43,44   # 消融：模型种子
TRAJECTORY_INTENTION_EPOCHS=70    # 消融：意图阶段 epoch
TRAJECTORY_MC_TRIALS=1000         # 蒙特卡洛次数/标签
```

单卡显存不够时优先调小 `batch_size`（配置里 256）与 `TRAJECTORY_TARGET_WINDOWS`；**不要**改 `observation_steps`/`prediction_steps`，那会破坏与论文协议的一致性。

---

## 5. 论文数值复现（训练后）

```bash
bash bash/run_paper_targets.sh outputs/military cuda:0
```

依次产出：

1. IV-A 九类 / 十一档：每标签 1000 次独立蒙特卡洛（论文口径，非单次确定性推理）
2. Fig. 14：Table VI 四组合场景（VT=300、VM=360、r0=5.3 km、To=8 s、Tp=6 s）的预测误差
3. `check_paper_targets.py`：四项锚点对照报告 → `outputs/military/paper_targets/paper_targets_report.json`

| 锚点 | 论文值 | 来源 |
|---|---|---|
| IV-A 九类准确率 | ≈88% | 论文 IV-A |
| IV-A 十一档准确率 | ≈94% | 论文 IV-A（含混淆矩阵描述） |
| IV-B ADE/FDE | ≈102 m | 论文 IV-B |
| Fig.14 四场景 | 150 / 202 / 81 / 173 m | Table VI + Fig.14 |

> Fig.14 必须用**训练后的** IV-B checkpoint；未训练时用恒速基线，四场景误差会相同（镜像对称，属正常现象）。

---

## 6. 建议执行顺序与耗时（单卡 A100 量级估算）

```bash
bash bash/setup_server.sh .venv auto && source .venv/bin/activate

# 先跑最短的，确认链路与显存（AutoDL 把输出放数据盘：/root/autodl-tmp/smoke）
bash bash/run_engagement.sh outputs/smoke cuda:0 forecast      # 建议先小数据：TRAJECTORY_TARGET_WINDOWS=2000
# 再跑正式
bash bash/run_engagement.sh outputs/military cuda:0 intention  # ~0.5-1 h
bash bash/run_engagement.sh outputs/military cuda:0 overload   # ~0.5-1 h
bash bash/run_engagement.sh outputs/military cuda:0 forecast   # ~4-8 h（200 epoch，早停可能提前）
bash bash/run_engagement.sh outputs/military cuda:0 ablation   # ~10-20 h（9 次训练 ×2）
bash bash/run_paper_targets.sh outputs/military cuda:0         # ~1-2 h（1000 次/标签蒙特卡洛）
```

先用一个小规模确认链路：

```bash
TRAJECTORY_TARGET_WINDOWS=2000 bash bash/run_engagement.sh outputs/smoke cuda:0 forecast
```

---

## 7. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: trajectory_prediction` / `xuance` | 缺 `PYTHONPATH` | `export PYTHONPATH="$PWD/Trajectory Prediction/src:$PWD"` |
| `military_teachers requires attention_supervision` | 配置自相矛盾 | 已修（`classifier_config` 意图阶段自动关闭）；若手改配置请注意两者联动 |
| `Engagement graph expects whole entity blocks` | `observation_steps` 与节点数不匹配 | 不要手改 `observation_steps`；改了必须重生成数据 |
| 边特征维度不匹配（21 vs 19） | 旧 checkpoint | 第七轮起边特征为 21 维，必须重训 |
| `No complete windows in partition` | 划分集为空 | IV-B 无 test 集，用 `validation`（脚本已处理；rolling 也有回退） |
| 显存 OOM | batch 256 + O=80/P=60 + 候选块 A=32 | 先降 `--target-windows` 与 `batch_size`；再考虑 `candidate_anchors` 32→16 |
| 想关闭第七轮的新边子集掩码做对照 | — | 配置里 `teacher_edge_subsets` 改 `false` 再跑一组 |
| RTX 5090/50xx（Blackwell）报 `no kernel image is available` | cu124/cu126 wheel 不含 sm_120 内核 | 装 cu128：`bash bash/setup_server.sh .venv cu128`（`auto` 已会按 compute capability 自动选 cu128） |

---

## 8. 与论文的对照入口

| 文档 | 内容 |
|---|---|
| `CODE_PAPER_VERIFICATION_CHECKLIST.md` | 40 项逐条对照；§十为第七轮（式36 边子集、ϕedg 的 V_L/V̇_L、意图阶段缺陷修复） |
| `PAPER_ALIGNMENT_GAP_AUDIT.md` | A–G 七类差异、U 类工程决策、§10.5 第七轮登记 X1–X6 |
| `METHOD_MAPPING.md` | 论文公式 → 代码位置逐行映射 |
| `MILITARY_PORT_PLAN.md` | 军用改造的 15 模块对照与 D1–D8 决策 |

已知**无法逐式复现**并已在报告中单列的事项（不是 bug）：
- **X1 教师分数定义域**：论文式(42)–(45) 与 Remark 3 把教师定义在「候选未来状态/预测增量」域，而式(15) 的图是 2t_o 历史节点——论文自相矛盾，当前按决策 D1 落在观测历史域。
- ~~式(26)–(31) 在正文中不存在~~ **勘误**：式(26)–(31) 实际在 Table I 算法框内（PDF 文本层未提取到框内公式）；Cesc 窗口平均（式27）、Cenv 全弧点（式28）、m(j) 合成（式30）、软裁剪（式31）均已按此实现。
- 式(45) 的 `arccos` 缺 |Δy_j| 归一化（论文疑笔误，代码用对称版本）。
