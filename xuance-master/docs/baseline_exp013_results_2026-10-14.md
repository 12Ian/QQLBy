# 013纯MATD3改进实验：设置、进度与最终结果

更新时间：2026-10-14T04:22:51.186060+00:00；状态：running；已观测转移：841,296 / 3,000,000。

进行中数据不能当最终结果；升级由原安全门槛决定，未通过门槛不强行升级。

## 1 方案选择与算法口径

先完成010、011、012三项75万转移对照及67101种子独立100回合验证。若某方案验证安全成功率≥5%，优先保留真实的渐进改善，按安全率最高、碰撞最低、捕获率最高选择方案做完整300万转移训练。5%只用于筛选长实验，课程升级仍为连续两次安全≥80%、碰撞≤10%。若没有重复出安全改善，则采用010仅将Actor及目标Actor输出改为Softsign的候选。

实际选择：{'pilot': '012', 'checkpoint_label': 'final', 'metrics': {'episodes': 100, 'success_rate': 0.18, 'safe_success_rate': 0.17, 'collision_rate': 0.83, 'building_collision_rate': 0.82, 'boundary_collision_rate': 0.83, 'uav_collision_rate': 0.0, 'mean_episode_steps': 681.83}, 'method': 'validated_pilot_configuration', 'initialization': 'random, not selected checkpoint'}。从随机权重、空回放池、新优化器重新开始；不使用教师、PD控制器、示范初始化或旧训练模型。

仍复用XuanCe共享MLP Actor、集中式双Critic的MATD3；三维动作范围[-1,1]、83/332维网络输入、512/512隐藏层、观测内容与全局目标信息保持一致。若采用Softsign，只改变有界输出a=z/(1+|z|)，导数为1/(1+|z|)²；不提供自动朝向目标或人工安全控制。

009/010的实际logit约10–15时，float32 tanh导数为0；Softsign可保留约0.008–0.004的导数。数值验证和GPU测试只证明实现及梯度有效，不证明训练成绩提高。

## 2 环境、运动学、课程和奖励

城市与出生设置沿用009：固定20/40/60/80/100栋；追捕机西侧集结，逃逸机东侧自由空间出生，最小初距450米；普通/高层/地标比例50/40/10。物理0.1秒、决策0.5秒、800决策步=400秒。追捕机速度11、加速度3、jerk6；逃逸机各级最高速度3/5/7/8/12、加速度4、jerk8、感知100米；GAT关闭。

只保留同一架追捕机在该级半径50/35/25/20/15米内连续保持2秒，捕获不要求全队此前无碰撞；安全成功另行统计。训练真实建筑/边界碰撞终止，评估关闭提前终止并统计完整回合。

保留用户闭合速度公式“平均＋最小”和最近队友夹角平均公式；progress/distance/teammate为0，机间碰撞不计；建筑/边界稠密制动余量惩罚、真实接触及全队失败惩罚、捕获奖励保留。

实际追捕/队形权重：1.0/0.1；总体缩放0.1；探索0.1→0.03，衰减300000转移。实际配置JSON作为所有参数的权威记录。

每级至少100000转移，每50000转移评估100回合，连续两次安全≥80%、碰撞≤10%升级；每次追加所有已通过旧级的retention复测。没有混合课程回放。

## 3 各级真实训练统计

| 级别 | 完整回合 | 完整回合转移 | 捕获 | 安全捕获 | 最近当前级正式评估捕获/安全/碰撞 |
|---|---:|---:|---:|---:|---|
| 1 | 9,753 | 840,088 | 241 | 241 | 0%/0%/100%（800,000步） |
| 2 | 0 | 0 | 0 | 0 | 未进入正式训练评估 |
| 3 | 0 | 0 | 0 | 0 | 未进入正式训练评估 |
| 4 | 0 | 0 | 0 | 0 | 未进入正式训练评估 |
| 5 | 0 | 0 | 0 | 0 | 未进入正式训练评估 |

完整回合转移合计840,088；总转移841,296；正在进行的回合没有计入完整回合转移。

实际Critic/Actor优化次数：51744/25872；以training_updates.csv为准。

## 4 最终单一模型逐级复评

最终模型冻结、核验源文件和模型SHA256后，用新盲测种子组67201–67300，五级各100回合、8环境批量策略推理、无探索、最长800步、全回合碰撞统计。每回合单独设定种子，避免不同策略回合长度导致异步环境采样到不同案例。组内种子没有用于013调参；无论训练到哪一级，都如实复评。

| 级别 | 捕获率 | 安全成功率 | 碰撞率 | 建筑接触率 | 边界接触率 |
|---|---:|---:|---:|---:|---:|

尚未完成最终复评。

## 5 真实成功视频与限制

当前没有已验证的本轮成功视频。无安全案例或未到达该级时不编造样例。

视频使用实际保存的best_stage模型，CPU原生MATD3策略，无探索，原环境逐动作重放一致，核验模型和视频哈希。筛选样例不是成功率估计。

结果仅能证明本轮配置、固定城市和给定评估种子上的表现。若尚未稳定通过课程或最终复评仍差，目标继续进行。后续需要更多训练种子、新地图、Q与真实回报校准以及实际四机围捕判据；连续2秒判据本身不保证四机包围。

[原始失败分析与对照](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/docs/baseline_exp009_deep_analysis_2026-10-14.md)

[真实配置](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/results/matd3/baseline_exp013/config.json)

[过程状态](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/experiments/exp013/process.json)

[训练转移与优化记录](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/results/matd3/baseline_exp013/training_updates.csv)

[正式评估与旧级复测](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/results/matd3/baseline_exp013/curriculum_eval.csv)

![真实正式评估曲线](/home/ryy/UAV_Mine_baseline_coverage/xuance-master/docs/baseline_exp013_curves_2026-10-14.png)
