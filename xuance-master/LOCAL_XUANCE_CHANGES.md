# Xuance 1.4.4 本地接入改动

官方 PyPI 源码恢复记录：`UPSTREAM_XUANCE.json`。原有重名文件的备份位于本机 `outputs/bootstrap/original_xuance`，未删除用户自定义环境。恢复的官方源码及许可证保留；不宣称全套在线强化学习算法已经本机验证。

相对于官方包，框架入口有以下改动：

- `xuance/__init__.py`：增加 `get_configs = load_yaml` 兼容名称。
- `xuance/torch/__init__.py`：四类注册表延迟导入，监督学习任务不强制加载在线 Agent 的日志依赖。
- `xuance/engine/__init__.py`：识别 `trajectory_prediction` 并注册扩展。
- `xuance/common/common_tools.py`：轨迹 Runner 使用严格设备检查，显式 CUDA 不可用时报错。
- `xuance/torch/learners/model_based/dreamer_v3_learner.py`：移除未使用的 `from pyglet.resource import model`，避免监督训练导入 Learner 注册表时依赖 GUI 包。
- 新增 `xuance/extensions/trajectory_prediction` 和对应 YAML 配置。复用真实 RunnerBase 调度和清理机制、Policy/Learner 注册表及 Learner 基类。离线 Agent 直接接收数据，不构造 Gym 在线环境。

算法代码在 `Trajectory Prediction/src/trajectory_prediction`。最小依赖用于本次民用监督训练任务；运行其他 Xuance 在线算法时，应另行安装该算法需要的官方可选依赖。

官方来源：[Xuance 仓库](https://github.com/agi-brain/xuance)、[1.4.4 PyPI 发布包](https://pypi.org/project/xuance/1.4.4/)。
