# 实验源码归档与当前基线

这些归档保存已经实际执行的代码，避免服务器 outputs/ 被 Git 忽略而丢失实现。每个完整运行时使用 source.patch＋source_manifest.json 保存，基准 Git 提交为 4ac54745a14dc1010a438ae8bd2033dbdf56b8dd。模型、轨迹数据和视频继续保留在服务器，不纳入 Git；配置与报告记录原始路径和模型指纹。

| 归档 | 用途 |
|---|---|
| exp020_matd3_reference_800step | 普通 MATD3 第5级42%的原始800步实验，附最终检查点来源及百局记录 |
| exp024_025_validated_600step | 安全过滤性能优化、经过验证的600步运行时 |
| lstm_prediction_v2_5000 | 5000回合预测器的优化训练、对照评估与交付源码，source/ 为独立模块源码 |
| lstm_matd3_integration_v1 | 冻结预测器、88维观测、旧策略权重映射与集成验证 |
| lstm_matd3_capture_long_v1 | 当前三种子LSTM→三种子匀速长程队列、固定场景验证和独立测试 |

恢复一个完整运行时，在当前仓库 xuance-master 目录执行：

    python tools/restore_experiment_source.py --archive experiments/source_archives/exp020_matd3_reference_800step --destination /绝对路径/一个不存在的新目录

工具在新目录建立独立checkout，切换到记录的基准提交，检查并应用源码补丁，再逐个验证源码SHA-256。恢复的代码位于新目录/xuance-master。它不会覆盖当前工作目录或正在训练的运行时。其余完整运行时只需替换archive路径。服务器上的原始运行时位置见各 archive.json，外部配置位于各 configs/。

lstm_prediction_v2_5000/source/uav_lstm_prediction 是独立预测模块版本；完整600步采集环境依赖对应600步运行时，而非任意当前checkout。历史模型必须配合其记录的配置与评估口径。

42%属于800步/400秒、100个固定场景的第5级结果。原100局中只有28次成功发生在600步之前；这是历史轨迹前缀统计，不等同重新评估。当前LSTM初始化使用的就是同一份exp020最终权重。不同种子、600/800步预算不能直接混合排名。

当前尚未实现讨论中的实体注意力Actor-Critic；归档不会把方案讨论标记为已实现。
