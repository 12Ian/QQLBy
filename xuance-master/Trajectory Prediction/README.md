# 民用三维轨迹预测：Xuance 原生接入

已实现离线监督训练的完整工程流程：历史数据 → BiLSTM 意图识别 → 门控时序图注意力 → 时间记忆 → 自回归 GRU 增量预测，以及两阶段训练、验证选模、测试评估、基线比较、断点恢复和服务器打包。

v0.4新增民用双实体交互预测：固定路线交叉、共用通道、合流、坡道、通行优先级变化和分离对照；同期邻居历史进入静态FC与2O节点交互图，按完整场景划分和计算区间。详细参数、输入格式、I/C消融、本机与服务器命令见 [INTERACTION_GUIDE.md](INTERACTION_GUIDE.md)。旧单轨迹配置和检查点保持兼容。

入口默认使用 **Xuance 1.4.4 原生 get_runner、RunnerBase、Policy/Learner 注册表和 Learner 基类**。缺失的框架源码已从官方发布包恢复，校验和见 `../UPSTREAM_XUANCE.json`；本地框架改动见 `../LOCAL_XUANCE_CHANGES.md`。这是民用三维预测任务的监督学习接入，学习目标、数据和评价均围绕预测。

算法和论文通用部分的逐项对应见 [METHOD_MAPPING.md](METHOD_MAPPING.md)，补全及剩余差异见 [COMPLETION_AUDIT.md](COMPLETION_AUDIT.md)，实际运行证据见 [VALIDATION.md](VALIDATION.md)。原始论文数据和实验结果没有在此重现。

## 本机运行

本项目已经有可用的 `.venv`：Python 3.12.14、PyTorch 2.7.1+cpu。以下命令从仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe "xuance-master/Trajectory Prediction/run.py" check --device cpu
.\.venv\Scripts\python.exe -m pytest "xuance-master/Trajectory Prediction"
powershell -ExecutionPolicy Bypass -File "xuance-master/Trajectory Prediction/scripts/cpu_smoke.ps1"
```

冒烟脚本使用 `cpu_completion.json` 的缩小网络，生成90条独立低速轨迹，完成3轮分类预训练、4轮预测训练、测试集报告、预测CSV与逐步滚动评估。输出默认放到带时间戳的新目录。短轮次仅验证工程闭环。

如果需要新建环境，建议 Python 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r "xuance-master/Trajectory Prediction/requirements-dev.txt"
# 可选：注册 trajectory-prediction 命令；保持整个 Xuance 源码目录
.\.venv\Scripts\python.exe -m pip install -e "xuance-master/Trajectory Prediction"
```

精确 CPU 版本在 `requirements-tested-cpu.txt`。`run.py` 可直接使用源码；可选 `--backend standalone` 使用同一算法核心。服务器也必须携带已恢复的 Xuance 目录。其他在线强化学习算法的可选依赖不在这个最小环境的验证范围内。

## 数据与配置

输入必须采用米、秒、米/秒。CSV 必填 `track_id,time,x,y,z,vx,vy,vz`，可选整数 `label`。NPZ 使用以下数组，不允许 pickle：

| 键 | 形状 | 含义 |
|---|---|---|
| track_id | 字符串 [S] | 独立原始轨迹 ID |
| time | float64 [S] | 等间隔时间戳 |
| states | float32 [S,6] | x,y,z,vx,vy,vz |
| labels | 可选 int64 [S] | 0–8；未知为 -1 |

默认九类顺序为直行、左转、右转、上行、下行、左转上行、右转上行、左转下行、右转下行。窗口标签只在观察段多数表决，不读取未来标签。双向 LSTM 只使用已经观察到的历史。

按完整独立轨迹划分 70/15/15 后，分别切滑动窗口。不能将同一原始轨迹改成多个 ID 来随机划分。归一化只拟合训练观察段。`split_seed` 固定划分，`seed` 控制模型初始化和训练随机性；基线的不同种子复用相同划分。时间戳不一致、缺测、重复、非有限数值和长度不足直接报错，预处理时需明确重采样方案。

| 配置 | 用途 |
|---|---|
| configs/cpu_completion.json | CPU轻量网络；含历史边/可达性、渐进裁剪、双指标早停 |
| configs/cpu_interaction.json | CPU民用双实体交互；6轮预测、1个邻居 |
| configs/server_interaction.json | 完整交互结构；dt=.5/O20/P12 |
| configs/paper_interaction.json | 完整交互结构；dt=.1/O80/P60，低速民用运动上限 |
| configs/cpu_native.json | 保留上一版CPU验证配置，便于复用历史结果 |
| configs/server.json | 完整网络；dt=0.5，观察20步、预测12步 |
| configs/paper_forecast.json | 完整网络；dt=0.1，观察80步、预测60步，对齐论文 IV-B 时间协议 |
| configs/paper_intention.json | 对齐 IV-A 分类协议：dt=0.5、观察20步、Adam/70轮 |
| configs/cpu_smoke.json | 保留早期平行解码轻量配置，便于历史检查点复用 |

论文分类和预测实验的采样协议不同。`pipeline` 使用同一份数据、采样间隔、观察长度、划分与归一化进行两阶段训练，防止不兼容的权重迁移。选择 `paper_forecast.json` 时，分类预训练也观察80步；新配置使用独立的 `intention_dropout=.2`，预测与基线 `dropout=.1`。不能直接迁移 `paper_intention.json` 的20步检查点。

速度与加速度上限是民用可配置参数，默认10 m/s、1 m/s²，应按实际民用数据调整。预测采用速度/加速度投影；若末个观察速度已超出同时配置的上限，会报错以避免无法同时满足两种约束。yaw 局部坐标由最后观察速度确定，零水平速度时用零角度。

生成器产生2–6 m/s的独立平滑运动点，用于工程验证。真实实验请使用独立的民用轨迹数据。

## 两阶段训练、评估、预测

```powershell
$py = ".venv/Scripts/python.exe"
$entry = "xuance-master/Trajectory Prediction/run.py"
& $py $entry generate --output "xuance-master/outputs/trajectory/data/civilian.npz" --tracks 90 --steps 96 --dt 0.5
& $py $entry pipeline --data "xuance-master/outputs/trajectory/data/civilian.npz" --config "xuance-master/Trajectory Prediction/configs/cpu_completion.json" --output "xuance-master/outputs/trajectory/run01" --intention-epochs 3 --device cpu
& $py $entry evaluate --data "xuance-master/outputs/trajectory/data/civilian.npz" --checkpoint "xuance-master/outputs/trajectory/run01/forecast/best.pt" --device cpu --output "xuance-master/outputs/trajectory/run01/forecast/test_metrics.json"
& $py $entry predict --data "xuance-master/outputs/trajectory/data/civilian.npz" --checkpoint "xuance-master/outputs/trajectory/run01/forecast/best.pt" --track-id track_000000 --device cpu --output "xuance-master/outputs/trajectory/run01/prediction.csv"
```

`pipeline` 默认分类最多70轮：Adam、StepLR；预测使用配置的AdamW、warm-up+cosine、早停，默认冻结分类器参数和dropout。新配置的 `both`早停要求ADE和FDE均达到最小改善量才重置耐心计数，判定基准及裁剪进度随检查点保存。`best.pt`根据验证集分类交叉熵或预测 `.4 ADE + .6 FDE`选取。训练不读取测试集；流程完成后再显式测试。也可用 `train --stage joint`联合训练。

没有标签的数据可 `train --stage forecast`，或选择 `--model lstm|gru|tcn|transformer`。无分类预训练的图模型后验只属于内部表示，不能声称为已识别的可解释意图。基线和关闭意图的消融在预测 API 中返回 `intention_probabilities=None`。

新训练拒绝覆盖非空目录，生成数据和预测 CSV 拒绝覆盖已有文件。恢复单阶段到总计第20轮：

```powershell
& $py $entry train --data "xuance-master/outputs/trajectory/data/civilian.npz" --output "xuance-master/outputs/trajectory/run01/forecast" --resume "xuance-master/outputs/trajectory/run01/forecast/last.pt" --epochs 20 --device cpu
# 或继续未完成的完整两阶段流程，沿用原配置
& $py $entry pipeline --data "xuance-master/outputs/trajectory/data/civilian.npz" --config "xuance-master/Trajectory Prediction/configs/cpu_completion.json" --output "xuance-master/outputs/trajectory/run01" --intention-epochs 3 --device cpu --resume
```

检查点保存模型、优化器、调度器、可学习不确定性权重、AMP scaler、归一化、划分、数据指纹、epoch、早停状态和 Torch/DataLoader 随机状态。恢复只允许修改总轮数、设备、workers和CPU线程数。余弦调度的 `schedule_epochs` 在初次训练时固定；计划分段训练时预先设置最终调度长度。`last.pt` 用于续训，`best.pt` 用于推理。相同 CPU 环境的恢复已验证逐位一致；跨设备/版本不保证逐位一致。

## 基线与民用消融

```powershell
& $py $entry compare --data "xuance-master/outputs/trajectory/data/civilian.npz" --config "xuance-master/Trajectory Prediction/configs/cpu_completion.json" --output "xuance-master/outputs/trajectory/comparison01" --epochs 2 --intention-epochs 3 --seeds 42,43,44 --models graph,lstm,gru,tcn,transformer,graph_no_intention,graph_no_attention_supervision,graph_no_both
```

默认 `--seeds 42,43,44 --graph-training two_stage --intention-epochs 70`。固定输入、轨迹划分、归一化、预测训练预算与优化器；记录实际参数量与额外分类预训练轮数。每个种子只预训练一个分类器，四组图模型复用冻结的相同权重；无意图组将后验/context置为无信息常量。I/T四格为意图与三个民用教师的组合，不能对应为原文专用E因素的消融。完整配置需 `attention_supervision=true`。`--graph-training joint`可复用旧比较模式。

`comparison_runs.csv` 保存每次运行，`comparison_summary.csv` 给出跨初始化种子均值和样本标准差。单个种子标准差为空，不得当成多次试验稳定性结论。

## 滚动预测、图表与独立民用样本库

```powershell
& $py $entry rolling-evaluate --data "xuance-master/outputs/trajectory/data/civilian.npz" --checkpoint "xuance-master/outputs/trajectory/run01/forecast/best.pt" --output "xuance-master/outputs/trajectory/rolling01" --device cpu --refresh-steps 1
& $py $entry report --comparison "xuance-master/outputs/trajectory/comparison01" --data "xuance-master/outputs/trajectory/data/civilian.npz" --output "xuance-master/outputs/trajectory/figures01" --device cpu
& $py $entry study --config "xuance-master/Trajectory Prediction/configs/cpu_completion.json" --output "xuance-master/outputs/trajectory/study01" --data-seeds 101,102 --model-seeds 42,43 --tracks 24 --steps 64 --epochs 1 --intention-epochs 1 --models graph,gru --device cpu --figures
# 同一参数继续实验矩阵
# 在上一条study命令末尾加 --resume
```

滚动评估从检查点保存的测试轨迹逐条读取**实测**状态，每 `refresh_steps`更新后验、历史图与预测；warm-up需O个测量，预测步数P不变。只有未来完整的窗口计分，预测后才读取未来真值作评分。它与stride=5的固定窗口报告支持数不同，不能直接当作同一个测试协议。输出逐窗口/时刻误差、独立场景或轨迹bootstrap区间、预测与真值长表、CPU/设备推理耗时，计时含返回CPU数组。

`RollingPredictor(checkpoint, device, refresh_steps).push(timestamp, state)`可接入民用实测数据流，输入只有时间和六维当前状态；历史未满或未到刷新时返回None，刷新时返回三维位置、时间戳、可用分类后验。更换轨迹需调用 `reset()`，不自动混用轨迹。

`report`保存SVG/PDF及300dpi PNG，涵盖分类CE/准确率、预测训练曲线、混淆矩阵、ADE/FDE、e(t)含种子SD、灰度图和3D复合运动案例。只从保存的测试数据取值；案例取各复合类的首个测试窗口，训练曲线取第一个指定模型/种子，均不按误差筛选。`report_values.json`保存数值、原始来源hash和聚合规则，`REPORT.md`可直接预览。未提供标签/分类输出时省略混淆矩阵。

`study`的数据种子决定独立民用轨迹库，模型种子控制初始化；它们是两个不同重复层级。每库重新做整轨迹划分，库内各模型共享划分。结果报告库内种子均值、库间样本SD及1000次层次bootstrap区间；单次评估的轨迹bootstrap仍独立保留。这属于独立民用合成实验，不是原论文场景的Monte Carlo复现。小样本区间只作流程验证。绘图依赖将NumPy限制在 `<2`，与本机SciPy/Xuance环境保持兼容。

## 军用拦截交战版（论文 III/IV 章对齐）

军用版在民用骨架上替换了四层：交战数据层、对抗边特征层、E/R/T 教师层、候选分支裁剪层，并把协议对齐 IV-A / IV-B。所有民用能力保持可用且回归不破坏。

**数据生成**（式1–2运动学、r ≤ (2/3)·r₀ 触发反LOS逃逸、9g/15g 饱和、60° FOV 可见性、Table IV 采样）：

```powershell
# IV-B 协议：dt=0.1 s 直接采样；防御机初始在目标 FOV 内（论文 IV-B 设定）
& $py $entry simulate-engagement --output "xuance-master/outputs/trajectory/data/engagement.npz" --scenes 60 --steps 400 --dt 0.1 --sample-stride 1 --seed 42 --initial-fov-coverage 1.0
# IV-A 协议：dt=0.5 s 存储，内部仍按 0.1 s 积分后 5:1 抽取（同一仿真服务两套协议）
# IV-A 库要求 FOV 内外都存在，故 initial-fov-coverage=0.5
& $py $entry simulate-engagement --output "xuance-master/outputs/trajectory/data/engagement_iva.npz" --scenes 60 --steps 400 --dt 0.5 --sample-stride 5 --seed 42 --initial-fov-coverage 0.5
# 论文口径规模：自动放大 --scenes 直到滑动窗口数达到 5×10^4
& $py $entry simulate-engagement --output ".../engagement.npz" --steps 400 --dt 0.1 --sample-stride 1 --initial-fov-coverage 1.0 --target-windows 50000 --observation-steps 80 --prediction-steps 60 --window-stride 5
```

生成器额外写入 `overload_labels`（11档过载）、`engagement_states`（r/ṙ/e_L/ė_L/可见性）与 `evasion_mask`（逃逸触发），民用加载器忽略这些字段。机动基元**按语义类别抽样**（九类含"无机动"），过载量化取**带符号**法向过载，metadata 会报告 `semantic_class_counts` 与 `overload_level_counts` 供 IV-A 核对。

**时间尺度**：`--dt` 是存储步长，内部积分恒以 `dt/sample_stride` 进行（决策 D4），因此同一 `--seed` 下两套协议是同一仿真的 5:1 抽取。

**三套协议配置**：`configs/engagement_intention.json`（IV-A 9类语义意图，dt=.5/O=20/**P=0**（纯分类，无预测域）、`balance_intention_classes=true` 类别平衡重采样）、`configs/engagement_overload.json`（IV-A 11档过载强度，`intention_task=overload11`、`intention_classes=11`）、`configs/engagement_forecast.json`（IV-B，dt=.1/O=80/P=60、K=8、τ=.1、AdamW+warm-up cosine、200轮、早停20轮且改善<1 m即停、`fov_attention_mask=true`、`smoothness_weight=0`、`teacher_regularizers=paper`、`holdout_test=false` 的 80/20 划分）。`configs/cpu_engagement.json` 为 CPU 冒烟缩样。

论文口径要点（均已实现，详见 `PAPER_ALIGNMENT_GAP_AUDIT.md` §10）：意图编码器为式(12)序列分支 + 式(13)防御机静态分支；`Q_int/Q_tra` 交战因子（`s_L=[r,ṙ,e_L,ė_L,可见性]`）经 `engagement_states` 注入静态分支与图节点特征；ADE/FDE 保持单位权重，log-variance 不确定性加权只作用于三组注意力教师（式55）。

```powershell
& $py $entry train --data ".../engagement.npz" --config "xuance-master/Trajectory Prediction/configs/engagement_forecast.json" --output ".../run" --device cpu
& $py $entry evaluate --data ".../engagement.npz" --checkpoint ".../run/best.pt" --split validation --device cpu
```

**论文口径评估与图表核对**（IV-A 每标签 1000 次独立蒙特卡洛；Table VI 四场景；四项锚点对照）：

```powershell
# 每标签 N 次独立重采样（不同种子），输出准确率/宏F1/平衡准确率/多数类基线/混淆矩阵
& $py "xuance-master/Trajectory Prediction/scripts/monte_carlo_intention.py" --data ".../engagement_iva.npz" --checkpoint ".../run/best.pt" --trials 1000 --output ".../mc.json"
# Table VI / Fig.14 四复合场景终端误差（VT=300, VM=360, r0=5.3 km, To=8 s, Tp=6 s）
& $py "xuance-master/Trajectory Prediction/scripts/reproduce_figure14.py" --output ".../figure14" --checkpoint ".../run/best.pt"
# 88% / 94% / 102 m / 150-202-81-173 m 四项锚点对照
& $py "xuance-master/Trajectory Prediction/scripts/check_paper_targets.py" --intention ".../iva_mc.json" --overload ".../ovl_mc.json" --forecast ".../validation_metrics.json" --figure14 ".../figure14/figure14_summary.json"
```

**军用四格消融**（I = 意图分支，E = 交战边 + 候选裁剪 + E/R/T 教师）：

```powershell
& $py $entry study --config "xuance-master/Trajectory Prediction/configs/cpu_engagement.json" --output ".../ablation" --data-mode engagement --scenes 6 --steps 60 --data-seeds 101 --model-seeds 42 --epochs 1 --models graph,graph_no_intention,graph_no_engagement,graph_no_intention_engagement --device cpu
```

实现要点：候选分支裁剪采用"锚定候选法"（见 `MILITARY_PORT_PLAN.md` 决策 D1），在 2O 历史节点之外挂 O 大小的候选块（A 个锚点方向 + 掩码填充），按 `m(j)=Cesc·Cenv·Cene` 硬裁剪，并以 `exp(-Λ_crop·P_ij)` 软惩罚随 `crop_progress` 递增；被裁候选不参与注意力、教师与解码记忆；集合被裁空时保留代价最小的分支（恒速退化）。E/R/T 教师（式42–53）只在训练期使用，推理不输入。评估会额外报告逃逸触发前后的 ADE/FDE 分组。

## 服务器运行

在仓库根目录生成源码包，默认不包含数据、检查点、Windows环境和运行日志：

```powershell
.\.venv\Scripts\python.exe "xuance-master/Trajectory Prediction/scripts/prepare_server.py" --output "xuance-master/outputs/server/civilian_trajectory_v04.tar.gz"
```

将源码包和民用数据传至服务器，解压后新建 Linux 环境。先按服务器驱动安装 CUDA PyTorch，再安装依赖；使用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)。完整网络适合单GPU运行，未实现 DDP。

```bash
mkdir -p civilian_run
cd civilian_run
tar -xzf /absolute/path/civilian_trajectory_v04.tar.gz
python3.12 -m venv .venv
source .venv/bin/activate
# 此处安装与驱动匹配的 CUDA torch（要求 >=2.7,<3）
python -m pip install -r "xuance-master/Trajectory Prediction/requirements.txt"
# GPU严格检查 + 数据协议检查 + 完整结构两样本反向，再开始两阶段训练
bash xuance-master/bash/train_trajectory_prediction.sh /absolute/path/civilian.npz /absolute/path/run01 cuda:0
# dt=0.1数据，至少140步/轨迹时，采用论文预测时间协议
bash xuance-master/bash/train_trajectory_prediction.sh /absolute/path/civilian_dt01.npz /absolute/path/run02 cuda:0 "xuance-master/Trajectory Prediction/configs/paper_forecast.json"
# 同一配置继续未完成的流程
bash xuance-master/bash/train_trajectory_prediction.sh /absolute/path/civilian.npz /absolute/path/run01 cuda:0 "xuance-master/Trajectory Prediction/configs/server.json" --resume
# 独立民用库 × 初始化种子的完整八模型实验矩阵与绘图
bash xuance-master/bash/run_trajectory_study.sh /absolute/path/study01 cuda:0 "xuance-master/Trajectory Prediction/configs/paper_forecast.json"
# 同一参数继续矩阵
bash xuance-master/bash/run_trajectory_study.sh /absolute/path/study01 cuda:0 "xuance-master/Trajectory Prediction/configs/paper_forecast.json" --resume
# 军用拦截交战：intention|overload|forecast|ablation 四套协议（自动生成交战数据）
bash xuance-master/bash/run_engagement.sh /absolute/path/engagement01 cuda:0 forecast
bash xuance-master/bash/run_engagement.sh /absolute/path/engagement01 cuda:0 forecast --resume
bash xuance-master/bash/run_engagement.sh /absolute/path/ablation01 cuda:0 ablation
```

服务器脚本不依赖调用者工作目录，使用显式 CUDA 时不可用即报错。`auto` 才允许自动选择CPU。显存不足时调整新实验的 `batch_size`（如32/64），保持结构参数。支持 `mixed_precision=true` 的 CUDA AMP，但本机CPU环境尚未验证GPU/AMP实际运行。

小规模服务器脚本演练可设置 `TRAJECTORY_INTENTION_EPOCHS=1`；正式运行默认70。

实验矩阵脚本默认3个独立数据种子、每库180条/240步、3个初始化种子、完整八模型。可用 `TRAJECTORY_DATA_SEEDS`、`TRAJECTORY_MODEL_SEEDS`、`TRAJECTORY_STUDY_TRACKS`、`TRAJECTORY_STUDY_STEPS`调整规模；必须保证步数至少O+P。模型默认200轮、分类70轮；CPU演练应显式传入 `cpu_completion.json` 并缩小规模。

可用合成数据检查 dt=0.1 协议：`generate --dt 0.1 --tracks 90 --steps 240`；仅用于流程验证。

## 原生 Python 接口与输出

```python
# 从 xuance-master 目录运行；源代码入口会自行定位扩展
from argparse import Namespace
from xuance import get_runner
runner = get_runner(algo="trajectory_prediction", env="civilian_trajectory", env_id="single_track",
    parser_args=Namespace(data_path="/absolute/path/civilian.npz", output_dir="/absolute/path/run01",
                          trajectory_config="/absolute/path/xuance-master/Trajectory Prediction/configs/server.json",
                          device="cuda:0"))
runner.run(mode="benchmark")  # 训练→最优检查点测试→原生清理
```

`trajectory_prediction.inference.Predictor(checkpoint, device).predict(history)` 接收至少O行 `[x,y,z,vx,vy,vz]`，只截取最后O行，返回 `[P,3]` 绝对位置以及可用的意图后验。

每阶段输出 `config.json/environment.json/manifest.json/history.csv/last.pt/best.pt/training_summary.json`；评估输出 ADE/FDE（欧氏距离，米）、恒速基线、RMSE、逐预测步误差、准确率、宏F1、支持类别平均召回率、原始及行归一化混淆矩阵。附带逐窗口、逐轨迹、逐时刻、逐类别CSV和整独立分组聚类bootstrap 95%区间（1000次）。交互数据另导出逐场景、场景类型、观察近邻子集CSV，按整场景重采样；单轨迹数据按整轨迹重采样。速度/加速度诊断与阈值违例比例随报告保存。分类阶段的轨迹为恒速占位输出，只解释其分类指标。
