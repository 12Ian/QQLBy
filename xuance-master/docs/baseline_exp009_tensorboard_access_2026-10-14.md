# 实验009 TensorBoard查看方式

[打开009 TensorBoard](http://127.0.0.1:6007/)。009单独使用6007端口；原6006继续保留旧实验入口。

已验证HTTP 200，run为`seed_1_2026_1014_030750`，读取009的完整根事件文件；验证时共有1049个标量标签，包含`Train-Rewards/uav_0/r_purs`、`r_form`、`r_progress`。标签数会随训练日志增长，不能当作训练效果指标。

服务器日志根：`/home/ryy/UAV_Mine_baseline_coverage/xuance-master/outputs/logs/matd3/baseline_exp009/seed_1_2026_1014_030750`。

主要观察项：

- `Train-Rewards/uav_0–3/*`：各机完成回合的实际缩放奖励分项。
- `Train-Rewards-Per-Second/uav_0–3/*`：奖励除以真实回合秒数，便于比较0.5秒决策。
- `Train-Pursuit/*`：原始闭合速度、最小闭合速度、夹角及最近队友。
- `Baseline-Eval/current/Level-*/*`：当前级100回合无探索评估。
- `Baseline-Eval/retention/Level-*/*`：进入更高级后，对所有已通过旧级的周期复测；尚未升级时没有旧级行。
- `Train-Outcomes-Rolling100/*`：当前级最近100个训练回合结果，不等于正式评估成功率。
- `Train-Learning/*`及共享网络loss/Q：优化诊断；loss下降不能代替安全成功率。

根据用户最新确认，持续保持捕获不要求全队无碰撞，捕获率与安全成功率分别显示。`r_progress`保留零值日志列，实际权重为0。

本机SSH转发已建立。若以后转发连接结束，可在终端执行：

```powershell
ssh -N -L 6007:127.0.0.1:6007 -o ExitOnForwardFailure=yes UAV-RL-Server
```

密钥口令仅在终端提示时输入。持续监控uav仍保持用户指定的暂停状态；TensorBoard入口及实验自身的结束后复评程序不改变该自动监控设置。
