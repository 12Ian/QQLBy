# 服务器 Runbook：第四轮「论文字面对齐包」（A3/B5 复核 + E1/B4 实验）

> 日期：2026-10-13 · 服务器：本机 Ubuntu（RTX 5080，lh@Long-Jiang-MS-7E45），工作根 `~/trajpre`。新服务器无第三轮产物：链内按同 seed（42）确定性重生成数据，逐字节等同第三轮；对照数字取第三轮报告（T3 239.7 / T4 284.9-374.3）
> 包：`server_package_v4.tar.gz` · sha256 `66820cb71f3f72277020a992a0e6e36ee73d7417160e7a8dcb8e14a4d4ed8a52` · 942 文件
> Python 环境：`source ~/trajpre/venv/bin/activate`（setup_server.sh 创建，见 §2）

## 0. 这轮在做什么（一段话）

审计 P2 清单里登记的 A3/B5 经逐行复核**已在第五轮修成论文字面**（linear 解码头默认、式41 求和口径、λ_sep=0.1 恰为论文明给值，均有守护测试），无需重跑。真正剩下的两个"论文没说、我们做了工程选择"的旋钮：

| # | 旋钮 | 工程默认（现状） | 论文字面读法 | 依据 |
|---|---|---|---|---|
| B4 | 教师分布温度 τ_T | 0.1（锐化 10 倍） | 1.0（式 36-37 无温度，softmax 原样） | 论文式(36) 无 τ；"attention τ=0.1" 是注意力温度（已对齐），教师温度未提 |
| E1 | 解码头初始化 | 零初始化（恒速起点） | PyTorch 默认（论文未指定初始化） | 论文只说 "linear regression output head" |

两者都**只作用于 IV-B**（IV-A 两配置 military_teachers=False，T1/T2 不受影响——第三轮数字即对照）。本轮目的：把"论文字面"这个格子彻底清空，跑完报告可写"论文明确规定的每一项均已按字面验证"。

## 1. 上传与解包

```bash
# 本地上传 server_package_v4.tar.gz 到 $HOME/trajpre/
cd $HOME/trajpre
sha256sum server_package_v4.tar.gz   # 应为 66820cb7...ed8a52
mkdir -p pkg_v4 && tar -xzf server_package_v4.tar.gz -C pkg_v4
cd pkg_v4/xuance-master
```

## 2. 服务器冒烟（约 3 分钟）

```bash
source <你的venv>/bin/activate
cd $HOME/trajpre/pkg_v4/xuance-master
python run.py check --config "Trajectory Prediction/configs/engagement_forecast_literal.json" --device cuda:0 --backward | tail -5
# 预期: "backward_verified": true
```

## 3. 过夜链（上传两个脚本后一条命令挂上）

上传本地 `tmp/overnight_literal.sh` 与 `tmp/progress_literal.sh` 到服务器（如 `$HOME/trajpre/`），然后：

```bash
cd $HOME/trajpre/pkg_v4/xuance-master
nohup bash $HOME/trajpre/overnight_literal.sh > $HOME/trajpre/literal_nohup.log 2>&1 &
echo $!
```

链共 3 步（预计 ~2.5-3 h，5090）：

1. `lit_forecast` —— 字面配置训练 forecast（数据复用第三轮的 `out_f1f2/data/engagement_forecast.npz`，同一 seed 42 确定性数据，单变量=训练侧两旋钮）
2. `lit_targets_best` —— T3/T4/Fig.14 重评（写入 `paper_targets_literal/`，**不覆盖第三轮报告**）
3. `lit_targets_last` —— last 口径完整性重评（`paper_targets_literal_last/`）

## 4. 查进度

```bash
bash $HOME/trajpre/progress_literal.sh
```

或看链日志：`tail -f $HOME/trajpre/out_f1f2/chain.log`（找 `STEP lit_* OK/FAIL`）。

## 5. 判读规则

对照基线（第三轮，工程默认口径）：T3 ADE **239.7 m**、T4 终端 **284.9/312.6/284.7/374.3 m**、forecast best_score 252.7（86 ep 早停）。

| 结果 | 判读 |
|---|---|
| T3/T4 显著改善（如 T3 < 200 m） | 工程锐化/零初始化确实伤了模型 → 字面口径升级为主线，重算四锚点表 |
| 持平（±5-10% 噪声带内） | 字面包排除完毕 → 报告写"论文字面项全部按字面验证"，绝对值鸿沟归因维持"论文未披露细节" |
| 显著变差 | 同样是有价值结论：我们的工程选择是必要的，写进报告作为稳健性证据 |
| last 口径 | 预期延续第三轮过拟合崩的模式（第三轮 last 498.7 m），仅作口径完整性，不作判读主体 |

判读铁律：**两套口径（工程版 vs 字面版）并排进报告，不挑好的说**。无论结果如何，这一轮跑完即收口：论文字面 100% 覆盖。

## 6. 红线自查

- [x] 未改锚点定义、未动测试集、未挑种子
- [x] 字面配置与工程配置逐键 diff 仅 2 项（守护测试 `test_literal_alignment.py` 断言）
- [x] 默认路径字节不变（`decoder_zero_init` 默认 True = 现状；127 项测试全过）
