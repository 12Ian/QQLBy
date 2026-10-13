# 本机验证记录

验证时间：2026-10-11。本记录描述民用三维单轨迹及已测邻居交互预测的工程证据，各版本证据分别列出。

## 环境与框架

Windows 11、Python 3.12.14、PyTorch 2.7.1+cpu、NumPy 1.26.4；项目 `.venv` 可实际训练。Xuance 1.4.4 官方源码已恢复845个文件，下载归档SHA256为 `cd8e706619a5a8ed7cc33234d9879a4363196b9ecce5e19721e4be74170ead29`。本地接入改动见 `../LOCAL_XUANCE_CHANGES.md`。

CLI默认通过真实 `get_runner` 进入RunnerBase，注册Policy和继承Xuance Learner的训练器。Runner结束和异常路径的资源清理有测试覆盖。可编辑安装及 `trajectory-prediction check` 已成功，`pip check`无依赖冲突。最小依赖用于这项监督学习扩展，不代表所有在线RL算法均已验证。

## 自动检查

v0.2执行：`python -m pytest "xuance-master/Trajectory Prediction"`：**24 passed in 15.22s**，日志 `../outputs/trajectory/pytest_native_final.log`。当前v0.3结果见本文件末尾。

覆盖欧氏ADE/FDE数值、窗口标签只取历史、整轨迹无交叉、train-only标准化、边/节点门控梯度、Top-K掩码及self回退、独立head宽度、yaw旋转等变性、速度/加速度边界、三教师detach、四基线真实梯度、原生两阶段/benchmark/比较入口、早停和CUDA严格检查。AdamW+余弦调度+dropout+可学习不确定性权重的中断恢复，与连续训练的模型权重、criterion和验证结果逐位一致。

检查涵盖真实小数据训练，不用模拟优化器替代更新。

## 一键脚本的实跑

运行更新后的 `scripts/cpu_smoke.ps1`，完整流程生成数据→原生预检反向→3轮分类→5轮预测→测试评估→预测CSV，成功结束。

输出：`../outputs/trajectory/native_smoke_final/`。独立数据为90条、每条96步、dt=.5；SHA256：`90ab040f9ca2b238f3ec65b6f0cbc10dcbb1d41287dd29e9959600329c84413b`。按轨迹划分63/13/14条，对应819/169/182个窗口。观察20步、预测12步，未来6秒。CPU仅缩小hidden/embedding/层数，保留新方法所有组件。

| 测试指标 | 两阶段预测 | 恒速基线 |
|---|---:|---:|
| ADE，米 | 0.482187 | 0.760831 |
| FDE，米 | 1.316016 | 1.991062 |

分类准确率57.14%，宏F1=.511788，支持类别平均召回率=.643557。预测测试ADE的整轨迹bootstrap 95%区间为[.364513,.623480]米，FDE为[.987752,1.708379]米；14条测试轨迹、1000次重采样。最大预测速度5.918228 m/s，最大加速度.288187 m/s²；10 m/s和1 m/s²配置下，测试预测阈值违例比例均为0。分类只有3轮，不能将这些分类指标解释为充分训练的性能。

分类训练计时约1.14秒，预测训练计时约11.49秒，不含启动导入、评估及文件导出。详细指标、混淆矩阵、分时刻误差和CSV：`../outputs/trajectory/native_smoke_final/pipeline/forecast/test_metrics.json`。

预测导出 `prediction.csv` 含12个未来时间点和绝对三维位置。`best.pt/last.pt`、pipeline汇总、配置、归一化、SHA256和划分均保存在各阶段。

## 完整结构与基线

`configs/paper_forecast.json` 的完整结构已使用两样本CPU完成前向和一次真实优化反向：d256、6个独立64维GAT头、2图层+2时间层、4时间头、自回归GRU、O80/P60/dt=.1、三个民用教师与不确定性权重，参数量3,002,596，输出[2,60,3]、损失有限。记录 `../outputs/trajectory/paper_architecture_cpu_check.json`。这属于完整结构可训练性验证，不是200轮训练或GPU吞吐验证。

原生比较命令已完成graph、LSTM、GRU、TCN、Transformer和两个民用消融的独立训练/保存/测试，各2轮、1个初始化种子、固定同一划分。结果见 `../outputs/trajectory/native_comparison_01/comparison_summary.csv`；例如TCN的ADE约.432米，graph联合模式约.515米。两轮结果不用于算法排名，也不与3+5轮pipeline直接比较；单个种子的标准差为空。正式实验使用多种子与完整预算。

## 服务器流程与限制

服务器脚本执行严格设备检查、数据检查、完整结构两样本反向，再运行两阶段训练并导出测试报告。已通过Git Bash语法检查，并在Windows CPU上以1轮分类/1轮预测实际执行完整服务器脚本，结果保存在 `../outputs/trajectory/server_cpu_rehearsal/`；这不等同于Linux服务器运行。源代码可用 `scripts/prepare_server.py` 打包，附所有文件SHA256；默认不包含数据、环境和训练结果。上传真实民用数据后按README命令运行。

GPU/CUDA AMP和Linux服务器尚未实机运行，本机不能确认服务器训练耗时、显存和200轮收敛。已完成的CPU训练与完整网络反向证明这套代码可更新模型并产生有效预测，不证明复现了论文的原始场景、数据分布、专用因素或结果表格。

历史 `local_smoke`、`local_learning_check` 的早期平行解码结果保留作历史证据，与当前模型及划分不混用。综合机器可读记录位于 `../outputs/trajectory/validation_final.json`；输出文件留在本机，源码包不携带这些日志和检查点。

## v0.3：补全后的实际验证

新增内容详见 [COMPLETION_AUDIT.md](COMPLETION_AUDIT.md)。本节更新于2026-10-11，以上v0.2证据仍保留，不能混用其配置或指标。

最终回归为 **29 passed in 53.85s**，日志 `../outputs/trajectory/pytest_completion_final.log`。新增覆盖历史可达性阻断、7维边特征的实际梯度、裁剪逐更新、双指标早停状态恢复、连续与中断训练逐位一致、四格共享冻结分类器、滚动测量与批量预测逐位一致/时间戳拒绝、独立数据种子×初始化种子及恢复、图表数值到来源的对应。14条warning来自Matplotlib对新Pyparsing的弃用API调用，测试无失败。

环境已更新可编辑包至0.3.0；NumPy仍为1.26.4、Matplotlib3.10.6、contourpy1.3.3，`pip check`无依赖冲突。精确版本写入CPU依赖锁定表；绘图要求NumPy<2。

### 原生两阶段与逐步滚动

使用 `cpu_completion.json` 的3轮分类+4轮预测，在既有90条/96步独立民用数据上重新训练，仍为63/13/14整轨迹划分、819/169/182固定stride窗口。新增边特征、历史可达性、渐进裁剪、独立分类dropout和双指标早停配置均启用。测试文件为 `../outputs/trajectory/completion_pipeline/forecast/test_metrics.json`。

| 协议 | 测试窗口 | 预测ADE/FDE（米） | 恒速ADE/FDE（米） |
|---|---:|---|---|
| 每5步固定窗口 | 182 | 0.473944 / 1.245518 | 0.760831 / 1.991062 |
| 每1步滚动更新 | 910 | 0.470514 / 1.235860 | 0.758425 / 1.986460 |

固定窗口ADE的整轨迹95%区间[.364688,.599117]米，FDE为[.936994,1.588746]米。最大预测速度5.908240 m/s、加速度.777560 m/s²；配置上限下违例比例均0。分类测试准确率36.26%，宏F1=.236139，支持类别平均召回率=.333704；仅3轮，尚未充分训练，不能作论文识别率对比。

滚动刷新间隔`.5 s`，观察20步、预测12步，重估后验与图，未来信息不输入网络；长表有10,920条未来预测记录。CPU推理平均12.405 ms、中位12.105 ms、P95=15.719 ms，包括返回CPU数组；这只代表本次轻量CPU运行。滚动结果在 `../outputs/trajectory/completion_rolling/rolling_metrics.json`，与固定stride协议的窗口支持数不同。

### 完整结构、八模型对比和图表

新 `paper_forecast.json` 完整结构通过Xuance原生Learner一次CPU反向/优化：3,002,680参数，输入O80、输出 `[2,60,3]`，损失5.370464且有限。记录 `../outputs/trajectory/paper_architecture_completion_check.json`。这不等于200轮收敛或GPU验证。

八模型（完整GAT、四基线、I/T三个消融）×初始化42/43/44，均实际完成2轮预测、验证选模、保存和测试；图模型每种子共享3轮分类器。完整GAT测试ADE为`.535311±.034428`米，FDE为`1.333261±.091542`米；TCN为`.475854±.009626 / 1.262176±.038085`米。短训练没有重现论文的优势排序，全部结果均如实保存，不能据此宣称方法优越。完整表在 `../outputs/trajectory/completion_comparison/comparison_summary.csv`。

`../outputs/trajectory/completion_figures/`包含7类图、每类SVG/PDF/300dpi PNG共21个文件，以及 `report_values.json`与 `REPORT.md`。逐时刻误差均值/种子SD、ADE/FDE、分类和预测原始曲线、混淆矩阵、四复合类3D轨迹均从保存证据生成。已目视检查灰度误差图、ADE/FDE图、3D四面板、分类曲线；坐标单位/图例可读，图未裁切，3D三轴比例一致，包含所有方法，无误差择优筛选。

### 独立民用库与服务器脚本

`study`实际生成两个独立数据种子101/102的库，各24条/64步，交叉模型种子42/43，在GAT和GRU上完成分类1轮+预测1轮，共8次预测训练。两库数据hash不同，库间SD与层次bootstrap表已导出，见 `../outputs/trajectory/completion_study/study_summary.csv`。小规模只验证实验矩阵与分层统计流程；不是原文每类1000次场景试验。恢复行为有自动测试覆盖。

两个Bash脚本通过语法检查。新增 `bash/run_trajectory_study.sh`已在Git Bash/Windows CPU实跑预检反向→独立民用库→完整八模型两阶段训练→图表导出，缩小到12条/64步、1个数据种子和1个模型种子、分类1轮/预测4轮；结果 `../outputs/trajectory/server_study_cpu_rehearsal/`。单个重复的区间退化为点，不用于统计解释。这仍不能确认Linux/CUDA/AMP服务器表现。

更新的PowerShell一键脚本采用v0.3配置并含逐步滚动评估。当前源码打包、校验与解包后原生反向验证记录见 `../outputs/server/package_validation_v03.json`；源码包为 `../outputs/server/civilian_trajectory_v03.tar.gz`。不携带数据、检查点、环境或实验日志。综合机器记录为 `../outputs/trajectory/completion_validation.json`。

## v0.4：双实体民用交互的实际验证

最终回归 **40 passed, 14 warnings in 52.11s**，日志 `../outputs/trajectory/pytest_interaction_release.log`。新增11项检查覆盖确定性低速仿真、双方测量对齐、整场景划分、train-only双方归一化、图时序/空间掩码、实体与边特征梯度、邻居缺测NaN无效性、yaw旋转一致性、关闭C后的邻居输入不变性、原生两阶段/paired_scene/Agent推理、滚动与批量一致、I/C四格图表、连续/恢复模型和criterion逐位一致、独立场景库、CSV缺测格式、带邻居数据缺少scene_id时拒绝以及同场景类别一致性。14条warning仍为绘图库弃用提示。较早的39项运行和1项study元数据恢复检查也保留；最后只调整了3D图表标签/图例边界并实际重导、目视检查。可编辑包0.4.0和依赖检查通过。

### 数据、两阶段和逐步滚动

`../outputs/trajectory/interaction_v04/data.npz`为36个独立场景、每场景两参与者共72轨迹、每轨迹80步、dt=.5，邻居测量随机缺测5%。六类各6场景，速度不超过1.5 m/s、生成加速度不超过1 m/s²；最小间距诊断全部保存，不作任意参数下无碰撞保证。数据SHA256：`7d5532fc472b50062993c6243778afa7c985a47d767df859ca9b819dcb17f33e`。

场景划分25/5/6，对应轨迹50/10/12、固定stride窗口500/100/120。两参与者始终属于同一区；观察20步、预测12步（6秒）。`cpu_interaction.json`实际完成3轮独立分类和6轮冻结预测，最优验证检查点为第6轮。主结果在 `interaction_v04/pipeline/forecast/test_metrics_v04.json`，首轮pipeline原始结果文件也保留。

| 测试协议 | 窗口 | 模型ADE/FDE（米） | 恒速ADE/FDE（米） |
|---|---:|---|---|
| 每5步固定窗口 | 120 | .817817 / 1.672710 | .728827 / 1.772508 |
| 每1步实测滚动 | 588 | .806005 / 1.648538 | .703716 / 1.713433 |

固定窗口整场景bootstrap 95%区间：ADE [.674475,.947859]米、FDE [1.297540,1.974924]米；1000次重采样、6个独立测试场景、12条参与者轨迹。最大预测速度1.374158 m/s、加速度约1.000002 m/s²，2 m/s及1 m/s²上限配合数值容差的违例比例均0。按观察段6米范围内是否存在可用邻居，报告32个远邻/不可用窗口和88个近邻窗口；近邻子集短训ADE/FDE约.970159/2.034769米，恒速为.728046/1.857539米，尚未体现稳定交互预测优势。

滚动长表7,056行，双方测量同步输入，真值在预测后才用于评估。CPU推理均值24.084 ms、中位19.786 ms、P95=27.099 ms；此轮并行进行了其他检查，只记录观测耗时，不是严格独占资源的吞吐基准。结果在 `interaction_v04/rolling/rolling_metrics.json`。`interaction_v04/prediction.csv`另通过CLI导出12个未来点。

**类别支持限制：**主测试集只覆盖合流3场景、优先级变化1场景、共用通道2场景；120个固定窗口和588个滚动窗口的观察多数标签均为直行。分类accuracy=1，按固定九类计算宏F1=1/9，不能据此声称九类识别成功。六类场景已在下面的独立覆盖实验中全部测试，平衡九类识别实验仍未实跑；该仿真用于验证交互数据和代码流程。

### 完整结构、比较和图表

`paper_interaction.json`完整结构已通过Xuance原生Learner的一次CPU前向/反向优化：焦点80+邻居80历史节点，d256、6×64独立图头、2图层、2时间层、4时间头、P60自回归GRU；3,003,192参数，输出[2,60,3]，有限损失5.374929。记录 `../outputs/trajectory/interaction_full_check.json`。dt=.1协议的全库长训未进行。

八方法×初始化42/43，共16次真实原生预测训练，每次2轮；四组图模型每种子共享同一3轮冻结分类器，预测预算一致。完整图模型平均ADE/FDE为`.732365±.005486 / 1.745589±.015786`米；关闭意图组为`.731478±.000480 / 1.722453±.007859`米；关闭邻居组为`.743403±.016235 / 1.762431±.020310`米。单种子均按验证指标选模。短训并未重现原文因素优势或完整排序，四序列基线只读焦点历史，详表 `interaction_v04/comparison/comparison_summary.csv`。

`interaction_v04/figures_release/`导出7类×SVG/PDF/300dpi PNG共21文件，保存曲线数组、输入历史、邻居mask、方法预测和全部来源hash。已目视检查ADE/FDE、灰度e(t)和3D民用场景图，三轴米制等比例、图例完整；本测试分区有3种场景，因此3D图显示3个面板，按每类型首次测试窗口选取，未按预测误差选择。首次导出与标签/图例边界修正过程的旧图表也保留。

### 六类场景的独立覆盖验证

为覆盖主分区未包含的场景，另生成60场景/120轨迹/每轨迹96步的数据库（data_seed=202，dt=.5，缺测5%）；SHA256为`255aea2440b75148bc620a0af60be1aeb40161d9f024e6d7f8d307b3366a3e63`。在训练前仅用类别元数据确定split_seed=1，不读取预测误差，协议保存为`interaction_v04/coverage/protocol.json`；原主实验未改动。划分42/9/9场景，测试18轨迹/234窗口，覆盖全部六类。GAT和GRU各实际训练2轮，GAT另用2轮独立分类，原生验证选模后测试。

| 测试场景 | 独立场景数 | 窗口 | GAT ADE/FDE（米） |
|---|---:|---:|---|
| crossing | 2 | 52 | .767951 / 1.707068 |
| merge | 3 | 78 | .848306 / 1.921624 |
| priority_change | 1 | 26 | .902515 / 1.961550 |
| ramp_crossing | 1 | 26 | 1.013523 / 2.201736 |
| separated_control | 1 | 26 | .718013 / 1.499357 |
| shared_aisle | 1 | 26 | .802961 / 1.809595 |

总体GAT ADE/FDE=.835315/1.850138米，恒速=.855210/1.914198米；单种子、2轮，不作优越性结论。分类支持直行225窗口、上行9窗口，accuracy=.961538，但上行召回为0、支持类平均召回=.5、九类宏F1=.108932，仍未验证充分的方向识别。预测和恒速分类型完整指标在`coverage/comparison/graph/seed_42/test_metrics_scenarios.csv`。

`coverage/figures_release/`另导出7类/21文件，包括六场景3D面板；只按每类型首次测试窗口选例，双方观察历史和全部原始数组记录在report_values.json。已目视检查并修正3D轴标签边界和图例间距，无按误差选例。加上主实验，共两套42个最终图表导出。

### 独立场景库、服务器演练与交付

`interaction_v04/study/`实际生成两个独立场景库（数据种子101/102），每库12场景/24轨迹/48步；交叉初始化42/43，GAT与GRU各完成1轮分类/1轮预测，8次预测训练。恢复命令再次成功，未重新生成数据。层次bootstrap和库间SD保存为 `study_summary.csv`；两种子只是流程验证预算。

新 `bash/run_civilian_interaction.sh`已通过语法检查并在Git Bash/Windows CPU完整执行：原生反向预检→生成6场景/12轨迹/48步→1轮分类→8方法各6轮预测→图表导出。结果 `../outputs/trajectory/server_interaction_cpu_rehearsal/`，日志 `../outputs/trajectory/server_interaction_cpu_rehearsal.log`。这里只使用一个数据/初始化种子，区间退化为点；短48步也只用于演练，不能衡量场景中晚发生的交互。

同一脚本的`--resume`也已实际成功执行，保留原数据和已完成的模型实验，日志为`../outputs/trajectory/server_interaction_cpu_resume.log`。

v0.4源码包为 `../outputs/server/civilian_trajectory_v04.tar.gz`，配套文件manifest和 `package_validation_v04.json`；校验所有源码哈希，并在新解包路径用隔离Python导入包内模块、执行完整双实体原生反向。包不携带环境、数据、检查点或日志。综合机器记录为 `../outputs/trajectory/interaction_validation_v04.json`。

本机已验证CPU更新、评估、恢复和Bash调度；Linux/CUDA/AMP、200轮收敛及原论文专用场景/教师/原始数值仍未验证或实现。对齐范围和全部差异见 [INTERACTION_GUIDE.md](INTERACTION_GUIDE.md) 与 [METHOD_MAPPING.md](METHOD_MAPPING.md)。
