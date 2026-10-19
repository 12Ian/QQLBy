# 实验009持续保持捕获与积极逃逸baseline结果

更新时间：2026-10-14T04:43:54.733716+08:00。训练状态：completed；结束后复评状态：completed。

已执行／已观测环境转移：3,000,000，预算3,000,000。Critic实际更新187,244次，Actor实际更新93,622次。

正在训练或复评未完成时，本文件是进度报告。只有训练真实退出、逐级复评完成后，才作为完整结果交付。loss有限或下降不代表策略收敛。

## 1 本轮修改及与008的关系

本轮从随机权重、空回放池和新优化器开始，不使用008或007模型初始化。仍复用XuanCe MATD3、共享MLP Actor、集中式双Critic；不是重写算法。

删除009的5米直接拦截，只保留同一架追捕机在该级捕获半径内连续保持2秒；按照用户最新确认，不要求全队无碰撞。捕获率和安全成功率分别报告；此判据不能单独证明四机形成完整包围。008的直接拦截配置仅为历史复现保留。

逃逸机感知追捕机由60增至100米；所有课程加速度1改4 m/s²；课程最高速度3、5、7、8、12 m/s。原追捕机全局目标观测、79维内容、最近5栋建筑、身份附加4维和关闭GAT的设置保留。速度比12/11的声明上界修正，真实比值保留。

决策1改0.5秒、物理0.1秒，每步5个物理子步；最长800步仍400秒；gamma=sqrt(0.99)；制动反应时间显式保留1.5秒。300万转移只有008约一半的名义物理交互时间，且本轮同时改变多个因素，不能归因于单一奖励或当作等时长对照。

每次正式评估增加所有已通过旧级的retention复测。没有加入混合课程、分级回放或旧级升级门槛，不把复测本身称为已解决遗忘。

[完整实施计划及参数决定](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/docs/baseline_exp009_plan_2026-10-14.md)

## 2 环境 动力学与课程

固定1000×1000米、飞行高度50–350米；五张嵌套城市20/40/60/80/100栋。普通建筑50%：侧长20或40米、高10–60米；高层40%：侧长40或60米、高60–150米；地标10%：侧长60或80米、高150–350米。允许绕行和飞越。

追捕机期望出生(100,350,180)、(100,450,200)、(100,550,220)、(100,650,200)米；建筑安全修正继续启用。逃逸机出生盒x650–750/y350–650/z150–250米，自由空间出生、真实初距至少450米。

| 级别 | 建筑 | 逃逸机最低/最高速度 m/s | 加速度 m/s² | 捕获半径 m |
|---|---:|---:|---:|---:|
| 1 | 20 | 1 / 3 | 4 | 50 |
| 2 | 40 | 2 / 5 | 4 | 35 |
| 3 | 60 | 3 / 7 | 4 | 25 |
| 4 | 80 | 3 / 8 | 4 | 20 |
| 5 | 100 | 3 / 12 | 4 | 15 |

追捕机速度上限11 m/s、加速度上限3 m/s²、jerk上限6 m/s³；逃逸机jerk8 m/s³。双方加速度滞后0.5秒，目标速度响应1秒。平动积分沿用原实现：位置由旧速度及旧加速度推进，速度按旧加速度更新，加速度通过jerk限制逐步追随指令；没有姿态、风扰或完整四旋翼实机模型。

逃逸机每0.1秒重算人工势场。100米内感知追捕机时改用该级最高期望速度，并沿追捕机、建筑、边界排斥和近边界回中合力方向机动；未感知时采用最低巡航速度。保持自由空间修正和边界／建筑滑动。上限12不代表所有回合或所有时刻实际达到12；第五级目标快于追捕机，需要检查截击策略和实际逃逸行为。

每级至少100000转移，每50000转移评估100回合，连续两次安全成功≥80%、碰撞≤10%升级；失败不会强制升级。评估8并行、关闭探索、碰撞提前终止关闭，记录全回合接触。旧级复测逐级记录，不进入回放池。

## 3 奖励函数

方向与相对闭合速度按用户公式定义，三维速度单位m/s、夹角单位弧度：

$$u_{ie}=\frac{p_e-p_i}{\|p_e-p_i\|_2},\quad v_i^c=(v_i-v_e)^T u_{ie}$$

$$r_{purs}=\frac{1}{N}\sum_i v_i^c+\min_i v_i^c$$

最近队友j*依据三维欧氏距离选择，u_c指向最近队友：

$$r_{form}=\frac{1}{N}\sum_i\arccos(u_{c,i}^Tu_{ie})$$

各机获得相同团队追捕及队形贡献：0.01*r_purs*elapsed、0.01*r_form*elapsed。原公式没有改成覆盖率，也没有额外除以N。公式由决策结束状态计算再乘实际步长，是矩形时间近似。最近队友平局按编号处理、重合方向取0、余弦夹在[-1,1]避免NaN。

| 项目 | 实际贡献 |
|---|---|
| r_purs | 0.01×(平均闭合速度＋最小闭合速度)×elapsed |
| r_form | 0.01×平均最近队友夹角×elapsed |
| r_progress / r_distance / r_teammate | 0 |
| r_obstacle / r_boundary | 各−0.5×max(0,1−有效净空/50)²×elapsed |
| r_collision | 首次建筑−8、边界−10；每类持续接触−2/秒 |
| r_terminal | 成功事件全队每架＋30，未因真实碰撞失败 |
| r_failure | 真实建筑／边界碰撞导致训练终止时每架−30一次 |

有效净空=几何净空−[1.5×接近速度＋接近速度²/(2×3)]，保留负净空惩罚。机间碰撞和机间分离罚分关闭；超时不额外发failure。训练沿用真实建筑／边界碰撞终止；无探索全回合评估仍可在接触后捕获，这类结果计捕获而不计安全成功。

该夹角奖励不能单独保证目标周围所有方向被封堵，正奖励按时间累计也可能偏好延迟结束。通过捕获时间、超时率和分项曲线检查；目前权重只是本轮明确的初始选择。建筑低于z_min及捕获/接触同子步事件时序的既有检查仍列为待办，本轮没有声称修复所有几何风险。

## 4 优化 日志与验证

共享Actor输入83、MLP512/512、三轴tanh加速度指令；集中式双Critic联合观测／动作及身份输入332，MLP512/512。Actor/Critic学习率3e-5/1.5e-4、gamma约0.9949874371、tau0.001、batch512、回放200000、暖启动4096转移、每向量步学习、Actor延迟2、梯度裁剪0.5。目标动作噪声0.2、裁剪0.5，Actor使用第一Critic。探索0.1降到0.03，衰减前300000转移；种子1、16并行，学习率末端倍率1。

新增r_purs/r_form各机及团队日志、每秒分项、各机闭合速度/夹角/最近队友、原始团队奖励、真实完成回合秒数；保留奖励、成功、安全、各类接触、初始/最终距离、逃逸机实际速度/感知比例、loss/Q与真实更新数。root事件文件保持平面结构。

验证记录：{"tests": "63 passed in 12.79s", "gpu_smoke": {"status": "passed", "environment_steps": 2048, "architecture_changed": {"actor": true, "critic_A": true, "critic_B": true}, "critic_updates": 1008, "actor_updates": 504, "root_scalar_tags": 461, "all_scalars_finite": true, "gat_enabled": false, "experiment_name": "exp009_gpu_smoke_v1", "note": "Integration test, not evidence of policy performance or convergence"}}。短测仅验证实现、梯度与日志链路，不代表长训练成功。

## 5 实际训练效果

| 级别 | 完整回合 | 完整回合转移 | 模拟秒数 | 捕获数 | 安全捕获数 | 最新评估捕获/安全/碰撞 |
|---|---:|---:|---:|---:|---:|---|
| 1 | 51,720 | 2,999,022 | 1,499,502.70 | 30 | 29 | 1% / 0% / 100%（3,000,000步） |
| 2 | 0 | 0 | 0.00 | 0 | 0 | 未进入当前级正式训练评估 |
| 3 | 0 | 0 | 0.00 | 0 | 0 | 未进入当前级正式训练评估 |
| 4 | 0 | 0 | 0.00 | 0 | 0 | 未进入当前级正式训练评估 |
| 5 | 0 | 0 | 0.00 | 0 | 0 | 未进入当前级正式训练评估 |

完整回合转移合计2,999,022；总转移与其差额978包含尚未完成回合等进度口径差异，不能把完整回合统计作为总预算。模拟秒数只合计已完成回合。

## 6 最终冻结模型逐级复评

冻结最终同一模型，种子64001，五级每级100回合、8并行、无探索、最长800步、全回合碰撞统计。未训练到的级别依然可以复评，但不算课程毕业。

| 级别 | 捕获率 | 安全成功率 | 碰撞率 | 建筑接触率 | 边界接触率 | 平均决策步 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1% | 0% | 100% | 28% | 100% | 796.26 |
| 2 | 0% | 0% | 100% | 100% | 100% | 800.00 |
| 3 | 0% | 0% | 100% | 100% | 100% | 800.00 |
| 4 | 0% | 0% | 100% | 100% | 100% | 800.00 |
| 5 | 0% | 0% | 100% | 100% | 100% | 800.00 |

训练结束到达第1级；是否通过最后级须以连续正式评估门槛判断。最终单一模型表现与best_stage视频来源分别记录。

## 7 视频 曲线 文件与待办

尚无本轮已完成的视频；未到达或无真实安全成功时不编造样例。

![真实数据曲线](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/docs/baseline_exp009_curves_2026-10-14.png)


文档待办1：0.1秒物理＋0.5秒决策已实施；待办2：用户确认固定300万转移，等物理时间600万未采用；待办3：旧级周期复测已实施，混合采样／分级回放与多级共同选模仍待单独对照。

后续还需多训练种子、新地图、真实四机围捕判据、奖励消融与主动逃逸轨迹诊断。008与009任务、奖励、动力学、初始化和时间尺度都不同，不能直接以两次成功率差归因于某一项改变。持续监控uav按用户要求保持暂停。

[baseline_exp009.yaml](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/experiments/baseline_exp009.yaml)
[metadata.json](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/experiments/exp009/metadata.json)
[process.json](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/experiments/exp009/process.json)
[config.json](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/results/matd3/baseline_exp009/config.json)
[training_episodes.csv](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/results/matd3/baseline_exp009/training_episodes.csv)
[training_updates.csv](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/results/matd3/baseline_exp009/training_updates.csv)
[curriculum_eval.csv](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/results/matd3/baseline_exp009/curriculum_eval.csv)
[manifest.json](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/reviews/exp009_final_v1/manifest.json)

TensorBoard日志根：`outputs/logs/matd3/baseline_exp009/seed_*`。
