# F1+F2 过夜链运行手册（服务器版）

## 这轮改了什么（一句话）

十一档窗口标签从「滑窗分箱→多数投票」两段式改为论文 §IV-A 的「**观察窗均值一次分箱**」（F1）；
基元段长与逃逸段长从硬编码改为参数（F2，默认值不变=与旧行为逐字节等价，已证）。
九类任务保持多数投票不动（论文原文 dominant semantics，不动是对的）。

## 为什么预期 T2（十一档）大幅提升

旧口径有两层问题：
1. **2s/10s 滑窗均值分箱方差大**，骑界窗口像掷硬币；
2. **更致命的**：每个逐点 bin 用的是「过去 10s 滑窗均值」，观察窗开头的 bin 依赖观察窗**之前**的历史过载——
   模型根本看不到那段时间 → 标签含不可观测分量 → 精度存在原理性上限。
新口径：窗口均值只用观察窗**内部**的过载序列，而过载可由观察窗内相邻速度差分近似恢复 → 标签几乎完全可观测。
这就是论文能到 94% 而我们卡 46% 的最可能解释。预期 T2 46 → 70+（保守）乃至 85+（若观测噪声小）。
T1（九类）不受 F1 影响（逐点语义标签本身即时可观测，走多数投票分支不变）。

## 上传与冒烟（10 分钟）

```bash
# 1) 本机已构建: server_package_v3.tar.gz (sha256 4c85619847cadae61b7cccb0e007590f88d35f7e7e313cc357494e01721cbb64)
# 在服务器上（数据盘）:
cd /root/autodl-tmp
tar xzf server_package_v3.tar.gz   # 解出 xuance-master/
cd xuance-master

# 2) 冒烟（约 5 分钟，验证 F1 新口径 + F2 默认等价；forecast 步在本机
#    Windows 有已知段错误、服务器正常，冒烟一并验证该路径）
TRAJECTORY_EPOCHS=1 TRAJECTORY_TARGET_WINDOWS=300 TRAJECTORY_ENGAGEMENT_STEPS=200 \
  bash bash/run_engagement.sh /root/autodl-tmp/smoke_f1f2 cuda:0 overload
TRAJECTORY_EPOCHS=1 TRAJECTORY_TARGET_WINDOWS=300 TRAJECTORY_ENGAGEMENT_STEPS=200 \
  bash bash/run_engagement.sh /root/autodl-tmp/smoke_f1f2_fc cuda:0 forecast
# 期望: 两链全 OK（300 窗 1 epoch 下指标不必高，只要不崩不报错）
# 然后删掉冒烟目录避免占盘:
rm -rf /root/autodl-tmp/smoke_f1f2 /root/autodl-tmp/smoke_f1f2_fc
```

## 过夜全自动链（一次粘贴，睡觉）

设计：F1 主实验（censored 十一档 + 九类）+ F2 段长敏感性（十一档×3 档段长）+ IV-B forecast 复训 + 四锚点汇总。
全部走 run_step 容错封装（单步 FAIL 不断链），日志 tee 到独立文件，最后自动打印汇总表。

```bash
cd /root/autodl-tmp/xuance-master
ROOT=/root/autodl-tmp/out_f1f2
mkdir -p "$ROOT"
run_step() {  # run_step <name> <command...>
  local name="$1"; shift
  echo "===== [$(date +%H:%M:%S)] STEP $name START =====" | tee -a "$ROOT/chain.log"
  if "$@" >> "$ROOT/logs_$name.log" 2>&1; then
    echo "STEP $name OK" | tee -a "$ROOT/chain.log"
  else
    echo "STEP $name FAIL (see $ROOT/logs_$name.log)" | tee -a "$ROOT/chain.log"
  fi
}
export TRAJECTORY_TARGET_WINDOWS=50000
export TRAJECTORY_ENGAGEMENT_STEPS=400

# ── F1 主实验: 九类 + 十一档 censored（论文口径窗口标签）────────────────
run_step f1_intention bash bash/run_engagement.sh "$ROOT" cuda:0 intention
export TRAJECTORY_CENSORED_ENDS=1
run_step f1_overload bash bash/run_engagement.sh "$ROOT" cuda:0 overload

# ── F2 段长敏感性: 十一档 × {短段 1-3s, 长段 6-12s}（同为 censored 口径）──
export TRAJECTORY_PRIMITIVE_DURATION="1.0,3.0"
run_step f2_overload_short bash bash/run_engagement.sh "$ROOT/seg_short" cuda:0 overload
export TRAJECTORY_PRIMITIVE_DURATION="6.0,12.0"
run_step f2_overload_long bash bash/run_engagement.sh "$ROOT/seg_long" cuda:0 overload
unset TRAJECTORY_PRIMITIVE_DURATION TRAJECTORY_CENSORED_ENDS

# ── IV-B forecast（F1 对 IV-B 无直接影响，但 overload_series 也存了，
#    顺带用同版代码复训；与 intention/overload 同根目录以便四锚点汇总）──
run_step f1f2_forecast bash bash/run_engagement.sh "$ROOT" cuda:0 forecast

# ── 四锚点汇总（T1/T2 用 last.pt 口径，T3/T4 用 best.pt 口径）────────
export TRAJECTORY_CHECKPOINT=last
run_step targets_last bash bash/run_paper_targets.sh "$ROOT" cuda:0
export TRAJECTORY_CHECKPOINT=best
run_step targets_best bash bash/run_paper_targets.sh "$ROOT" cuda:0
unset TRAJECTORY_CHECKPOINT

echo "===== CHAIN DONE $(date -Is) =====" | tee -a "$ROOT/chain.log"
```

## 早上看的三个文件

```bash
# 1) 链路状态（六步 OK/FAIL 一览）
cat /root/autodl-tmp/out_f1f2/chain.log

# 2) 四锚点汇总（T1/T2 last 口径 + T3/T4 best 口径，两份报告）
cat /root/autodl-tmp/out_f1f2/paper_targets_last/paper_targets_report.json
cat /root/autodl-tmp/out_f1f2/paper_targets/paper_targets_report.json

# 3) 段长敏感性对比（F2: 短/默认/长 三档 T2）
for d in "" seg_short seg_long; do
  echo "== $d =="; cat "/root/autodl-tmp/out_f1f2/${d}/run/overload/test_metrics.json" 2>/dev/null | head -20
done
```

## 预期与判读

| 锚点 | 论文值 | 当前最优 | 本轮预期 | 说明 |
|------|--------|----------|----------|------|
| T1 九类 | 88% | 62.5 (R1-l) | 60-65 | F1 不影响九类（多数投票保留）|
| T2 十一档 | 94% | 46.3 (CEN-b) | **70+** | F1 主战场：窗口均值一次分箱 |
| T3 ADE/FDE | 102 m | 249/202 | 200-260 | F1 不影响回归任务 |
| T4 Fig.14 | 150-202-81-173 | 311-380 | 300-400 | 同上 |

判读规则：
- T2 ≥ 70%：F1 假设验证成功，十一档口径以本轮为准，进 F5（种子均值）收尾
- T2 < 55%：F1 提升有限，查 f2 段长档位——若短段明显更好，说明跨段含混是主要矛盾，F2 定稿
- T2 变差（<46）：异常，保留旧数据集对比 logs，回报分析
- F2 短/长档与默认差异 <2 点：段长不是主要矛盾，IV-A 段长自由度以默认 2-6s 收档

## 红线自查（本轮改动是否违背论文）

| 改动 | 论文依据 | 合规性 |
|------|----------|--------|
| 十一档窗口标签=观察窗均值一次分箱 | §IV-A 原文公式定义 | ✅ 正是论文要求 |
| 九类保留多数投票 | §IV-A "dominant maneuver semantics" | ✅ 论文原文 |
| 段长/逃逸段参数化（默认不变）| §IV-B "sampled within predefined bounds" 未给值 | ✅ 论文留白的自由度，已在报告披露 |
| npz 新增 overload_series/censored 字段 | 工程实现，不涉协议 | ✅ 纯数据管道 |
| 旧数据集回退旧行为 | 向后兼容 | ✅ 不影响已跑实验复评 |
