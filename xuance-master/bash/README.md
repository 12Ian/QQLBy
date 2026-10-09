# Bash 脚本说明

本目录集中存放服务器项目的 Shell 启动脚本。以下命令从 `/home/ryy/UAV_Mine/xuance-master` 执行；脚本也会自行定位项目根目录。训练结果统一写入项目根目录的 `outputs/`；脚本不再使用根目录下的旧结果路径。

| 脚本 | 任务 | 默认规模或要求 |
| --- | --- | --- |
| `run_crlg_dynamic_training.sh` | CRL-G 记录字段检查、完整训练和独立评估 | 必填实验名称；检查后训练 300 万步、16 并行、100 回合评估 |
| `run_feat_new_env_mt1.sh` | 旧版 Apollonius 多目标环境短程联调 | 4 追 1、空地图、2000 步、1 并行 |
| `run_feat_new_env_st.sh` | 旧版 Apollonius 单目标环境训练 | 4 追 1、空地图、500 万步、16 并行 |
| `run_medium_3m_background.sh` | 旧版 Apollonius 中密度建筑训练 | 4 追 1、K=10 建筑特征、300 万步、16 并行 |

## 启动命令

```bash
cd /home/ryy/UAV_Mine/xuance-master
bash bash/run_crlg_dynamic_training.sh my_unique_run
bash bash/run_feat_new_env_mt1.sh
bash bash/run_feat_new_env_st.sh
bash bash/run_medium_3m_background.sh
```

三个 Apollonius 脚本默认在后台启动，可设置 `RUN_IN_BACKGROUND=0` 在前台运行，并可用 `STEPS`、`PARALLELS`、`EXP_NAME` 等环境变量覆盖其默认参数。例如：

```bash
RUN_IN_BACKGROUND=0 STEPS=1000 PARALLELS=1 bash bash/run_feat_new_env_st.sh
```

CRL-G 脚本使用 `/home/ryy/miniconda3/envs/rl_env/bin/python`，并固定执行检查、训练、评估三个阶段。其他脚本激活 `CONDA_ENV` 指定的 Conda 环境，默认为 `rl_env`。

这些脚本保留了各自创建时的实验配置。当前三维动力学、观测和模型结构已改动，使用历史检查点或直接重复旧实验前，应核对模型与配置是否匹配。
