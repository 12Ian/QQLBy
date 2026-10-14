"""Generate evidence-based diagnosis and a live table of independent pilots."""
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DIAG=ROOT/'outputs/reviews/exp009_deep_diagnostic_v1'
PILOTS=ROOT/'outputs/experiments/exp010_012_pilots_v1'
STEM=ROOT/'docs/baseline_exp009_deep_analysis_2026-10-14'


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def rows(path):
    return list(csv.DictReader(path.open(encoding='utf-8'))) if path.exists() else []


def link(label,path):
    return f'[{label}]({path.resolve().as_posix()})'


def figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=read(DIAG/'final.json')
    if not result:
        return None
    row=result['rows'][0]
    trace=row['trace']
    p=np.array([x['positions'] for x in trace])
    e=np.array([x['target'] for x in trace])
    a=np.array([x['actions'] for x in trace])
    t=np.array([x['step'] for x in trace])*.5
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),constrained_layout=True)
    for i in range(4):
        axes[0].plot(p[:,i,0],p[:,i,1],label=f'P{i}')
        axes[0].scatter(*p[0,i,:2],s=20)
    axes[0].plot(e[:,0],e[:,1],color='black',label='evader')
    axes[0].set(xlim=(0,1000),ylim=(0,1000),aspect='equal',xlabel='X / m',ylabel='Y / m',
                title='Actual trajectory before first collision')
    axes[0].legend(fontsize=8)
    selected=read(DIAG/'summary.json')
    labels=['final','best']
    x=np.arange(2)
    axes[1].bar(x-.18,[selected[k]['mean_initial_Q_min'] for k in labels],width=.36,label='min(Q1,Q2)')
    axes[1].bar(x+.18,[selected[k]['mean_discounted_return'] for k in labels],width=.36,label='actual discounted return')
    axes[1].set(xticks=x,xticklabels=labels,title='Matched 32 episodes: Q vs real return',ylabel='Return')
    axes[1].axhline(0,color='grey',linewidth=.8)
    axes[1].legend(fontsize=8)
    heat=axes[2].imshow(a.reshape(len(a),12).T,aspect='auto',origin='lower',vmin=-1,vmax=1,
                        cmap='coolwarm',extent=[0,t[-1]+.5,0,12])
    axes[2].set(yticks=np.arange(12)+.5,yticklabels=[f'P{i} {axis}' for i in range(4) for axis in 'xyz'],
                xlabel='Simulation seconds',title='Actual deterministic acceleration commands')
    fig.colorbar(heat,ax=axes[2],label='Normalized command')
    for ax in axes[:2]:
        ax.grid(alpha=.15)
    out=STEM.with_suffix('.png')
    fig.savefig(out,dpi=170)
    plt.close(fig)
    return out


def feasibility_figure(result):
    if not result:
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    labels=['rl','pd','pd_clearance']
    x=np.arange(3)
    axes[0].bar(x-.18,[result['modes'][k]['safe_captures'] for k in labels],width=.36,label='Safe captures')
    axes[0].bar(x+.18,[result['modes'][k]['collisions'] for k in labels],width=.36,label='Collisions')
    axes[0].set(xticks=x,xticklabels=['009 RL','PD controller','PD + clearance'],ylim=(0,35),ylabel='Episodes / 32',
                title='Paired initial states; controllers are NOT RL results')
    axes[0].legend(fontsize=9)
    buildings=read(ROOT/'xuance/environment/multi_agent_env/buildings/buildings_uniform_20.json')
    for b in buildings:
        axes[1].add_patch(Rectangle((b[0],b[2]),b[1]-b[0],b[3]-b[2],color='grey',alpha=.25))
    trace=result['modes']['pd']['records'][0]['trajectory']
    p=np.array([row['positions'] for row in trace])
    e=np.array([row['evader'] for row in trace])
    for i in range(4):
        axes[1].plot(p[:,i,0],p[:,i,1],label=f'P{i}')
    axes[1].plot(e[:,0],e[:,1],color='black',linewidth=2,label='evader')
    axes[1].set(xlim=(0,1000),ylim=(0,1000),aspect='equal',xlabel='X / m',ylabel='Y / m',
                title='Real 3D rollout projected to XY; first diagnostic seed')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.15)
    out=ROOT/'docs/pursuit_feasibility_2026-10-14.png'
    fig.savefig(out,dpi=160)
    plt.close(fig)
    return out


def main():
    summary=read(DIAG/'summary.json')
    base=ROOT/'outputs/results/matd3/baseline_exp009'
    episodes=rows(base/'training_episodes.csv')
    updates=rows(base/'training_updates.csv')
    stats={}
    for label,values in [('all',episodes),('first1000',episodes[:1000]),('last1000',episodes[-1000:])]:
        stats[label]={key:float(np.mean([float(row[key]) for row in values])) for key in
                      ['episode_steps','success','safe_success','mean_return','r_purs','r_form','r_failure',
                       'r_boundary','r_obstacle','initial_target_distance_mean','final_target_distance_mean']}
    state=read(PILOTS/'process.json')
    pilots=[]
    for n in ('010','011','012'):
        result=ROOT/f'outputs/results/matd3/baseline_exp{n}_pilot'
        status=read(result/'training_status.json')
        evals=[row for row in rows(result/'curriculum_eval.csv') if row['phase']=='current']
        run=next((row for row in state.get('runs',[]) if row['number']==n),{})
        pilots.append(dict(number=n,status=run.get('status','pending'),training_status=status,
                           observed_transfers=run.get('training_summary',{}).get('environment_steps',status.get('step',0)),
                           latest_current_eval=evals[-1] if evals else None,evaluations=run.get('evaluations',[])))
    feasibility=read(ROOT/'outputs/reviews/pursuit_feasibility_v1/results.json')
    bootstrap=read(ROOT/'outputs/reviews/exp009_target_actor_diagnostic_v1/results.json')
    gradients=read(ROOT/'docs/baseline_exp009_saturation_gradient_2026-10-14.json')
    pilot_q=read(ROOT/'outputs/reviews/exp010_012_q_diagnostic_v1/summary.json')
    output_probe=read(ROOT/'outputs/reviews/actor_output_saturation_v1/results.json')
    next_run=read(ROOT/'outputs/experiments/exp013/process.json')
    bellman=read(ROOT/'outputs/reviews/baseline_bellman_profile_v1/summary.json')
    record=dict(updated_at=datetime.now(timezone.utc).isoformat(),baseline_training_stats=stats,
                feasibility_diagnostic=feasibility,target_actor_diagnostic=bootstrap,saturation_gradient=gradients,
                pilot_q_diagnostics=pilot_q,
                actor_output_probe=output_probe,next_run=next_run,
                bellman_diagnostics=bellman,
                diagnostic_summary=summary,pilot_process=state,pilots=pilots)
    STEM.with_suffix('.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    text=['# 009失败诊断与追捕改进试验','',f'生成时间：{record["updated_at"]}。改进目标仍在进行，试验结果未齐时不能认定改进成功。','',
          '## 1 已确认的实验结果','',
          '009正常退出，实际300万环境转移；51,720个完整回合，30次捕获、29次无碰撞捕获；未通过第1级。最终同一模型在64001种子上逐级100回合复评，第1级捕获1%、安全0%、碰撞100%，第2–5级捕获0%、安全0%、碰撞100%。第1级视频20次筛选无安全案例，保留失败日志，未制造成功视频。','',
          f'实际优化次数：Critic {updates[-1]["updates"]}，Actor {updates[-1]["actor_updates"]}；loss降低没有对应安全成功率提高。', '',
          link('009完整最终结果与原始数据',ROOT/'docs/baseline_exp009_results_2026-10-14.md'),'',
          '## 2 深度诊断：32回合真实执行','',
          '使用诊断种子66001，第1级，同一初始布局分布，关闭探索，沿用训练的真实建筑／边界碰撞终止；每次重新调用原环境。该种子用于诊断，不能再作为新的盲测种子。初始随机、最终和best_stage_0分别32回合。额外8回合关闭碰撞终止以查看完整轨迹；不能把这种回报与训练终止条件下的Q直接比较。','',
          '| 模型 | 捕获/32 | 碰撞/32 | 平均步数 | 饱和分量比例 | 初始min(Q1,Q2) | 实际折扣回报 | Q偏差 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('initial','final','best'):
        s=summary[name]
        text.append(f'| {name} | {s["captures"]} | {s["collisions"]} | {s["mean_steps"]:.2f} | {s["saturation_fraction"]:.1%} | {s["mean_initial_Q_min"]:.2f} | {s["mean_discounted_return"]:.2f} | {s["mean_bias_terminal_only"]:.2f} |')
    text += ['', '饱和定义为动作分量绝对值>0.95。初始模型的Critic未经训练，因此其误差不作为训练是否退化的基准。最终32回合均以真实碰撞终止，没有截断尾部；这里的Q偏差是终止前完整折扣回报的直接差值。最终双Critic较小值平均+21.90，真实回报−16.20，仍高估约38.10；两个Critic在这些状态上接近，因此取较小值没有消除共同偏差。', '',
             '最终模型32回合首次接触均为uav_2撞y_max。首条轨迹从(100,550,220)出发，在约64.5秒撞北边界；撞前y≈998.63米、向北速度≈6.67 m/s，命令仍接近(1,1,−1)。该飞机没有及时向南制动。出生点不是贴在北边界，不能将这类失败解释为初始净空不足。', '',
             '首条初始状态，Critic对uav_2的y动作导数为正（约+0.074），而实际策略最终向北撞墙。这说明至少在该诊断样本中，Actor会受到错误价值梯度的引导；不能据此断言所有状态的梯度都错误。', '',
             'TD3通过双Critic取小值及延迟策略更新减轻价值高估，但无法保证在本任务中消除高估。来源：[TD3原论文](https://arxiv.org/abs/1802.09477)。', '',
             '## 3 奖励信号与经验分布','',
             '| 区间 | 平均回合步数 | 平均r_purs | 平均r_form | 平均r_failure | 平均最终目标距离 |',
             '|---|---:|---:|---:|---:|---:|']
    for name in ('first1000','last1000','all'):
        s=stats[name]
        text.append(f'| {name} | {s["episode_steps"]:.2f} | {s["r_purs"]:.3f} | {s["r_form"]:.3f} | {s["r_failure"]:.3f} | {s["final_target_distance_mean"]:.2f} m |')
    text += ['', '全训练平均回合只有约29秒；初距约617米，尚未充分接近便因某一飞机碰撞终止。末1000回合已更接近目标，但仍主要是碰撞经验。一次全队失败每架−30；闭合速度原始值约8 m/s时，009稠密追捕贡献约+0.08/秒，需要约375秒才抵一笔失败惩罚（此比较未折扣，仅用于说明量级）。', '',
             '因此存在两个需要检验的假设：早期追捕信号相对终止事件较弱；噪声在前30万转移内从0.1降到0.03，在长期饱和策略附近缺少不同动作的经验。不能仅凭这些量级确认因果，需要分别进行对照。', '',
             '夹角奖励最高约+0.0314/秒，悬停也可能获得正值；该奖励不保证真正覆盖逃逸方向。闭合速度奖励更强也可能增加冲撞或过冲，所以必须同时比较安全捕获、接触、动作饱和和Q误差。保留用户公式，当前不引入覆盖奖励、不启用GAT、不改变观测范围。', '',
             '## 4 已复现和修复的动作边界问题','',
             '继承的DDPG探索函数在tanh后加高斯噪声，没有再裁剪。例如网络动作0.95加噪声0.5后返回1.45；环境将其裁到1，回放池却保存1.45，违反动作空间声明。目标Actor的平滑动作反而已经裁剪，因此经验动作与目标动作使用了不同边界处理。此问题确认存在，是否是全部训练失败的主要原因仍需试验。', '',
             '在MATD3_Agents.action返回之前增加配置开关matd3_clip_actions，对每个环境／每架飞机按Box上下界裁剪；随后环境和回放池都接收同一个命令。历史配置默认关闭，保留旧实验行为；010–012显式开启。没有改变动作维度、物理积分或奖励。', '',
             '回归测试先出现2个预期失败，修复后67项通过。GPU1024转移、探索标准差3.0强制越界条件下，四架飞机回放动作最大绝对值均1.0，Actor与双Critic均实际更新、参数有限。短测只是验证动作／回放链路，不是学习成绩。', '',
             link('动作边界GPU审计',ROOT/'outputs/reviews/exp010_bound_smoke_v1/audit.json'),'',
             '## 5 单因素改进试验与当前进度','',
             '三个试验顺序执行，每个75万转移、16并行、种子1，从随机权重和空回放池开始。010初始36个策略状态张量已与009逐项完全一致，另外两次结束时亦将核验。仍使用0.1秒物理、0.5秒决策、强逃逸机课程、原成功／安全升级条件、原城市场景与出生布局。原009完整记录保留。', '',
             '| 试验 | 相对010的改变 | 状态 | 已记录步数 | 当前级最近正式捕获/安全/碰撞 |',
             '|---|---|---|---:|---|']
    variants={'010':'动作裁剪修复基线','011':'探索0.3→0.1，在300万转移内衰减','012':'追捕权重0.1→1.0，队形权重不变'}
    for p in pilots:
        ev=p['latest_current_eval']
        metrics=(f'{float(ev["success_rate"]):.0%}/{float(ev["safe_success_rate"]):.0%}/{float(ev["collision_rate"]):.0%}' if ev else '尚无同步评估')
        text.append(f'| {p["number"]} | {variants[p["number"]]} | {p["status"]} | {p["observed_transfers"]:,} | {metrics} |')
    completed_evaluations=[(p['number'],ev) for p in pilots for ev in p['evaluations']]
    if completed_evaluations:
        text += ['', '### 已完成试验的独立验证','',
                 '种子67101，第1级每模型100回合，8并行，无探索、全回合碰撞统计。此验证用于方案选择，未将结果作为训练回合或课程毕业成绩。','',
                 '| 试验 | 模型 | 捕获率 | 安全成功率 | 碰撞率 |',
                 '|---|---|---:|---:|---:|']
        for number,ev in completed_evaluations:
            m=ev['metrics']
            text.append(f'| {number} | {ev["label"]} | {m["success_rate"]:.0%} | {m["safe_success_rate"]:.0%} | {m["collision_rate"]:.0%} |')
    text += ['', '75万转移的试验预算并不证明已经充分训练；009在75万步的安全成功率也是0。这轮先筛查因素，再决定长训练。所有试验末模型与第1级最佳模型都将使用同一验证种子67101各评估100回合（无探索、全回合碰撞统计），用安全成功率优先、碰撞率次之、捕获率再次比较。该种子用于方案选择；67201保留给选定方案盲测，不提前用于调参。', '',
             '用户已明确后续坚持纯MATD3随机初始化训练。进一步优化围绕有界探索、Q与真实回报校准、目标动作与回放分布、Actor输出数值稳定性和早期失败截断开展；PD控制器仅用于独立可行性诊断。后续对照继续使用已确认的任务，不用缩短出生距离或减少建筑替代当前任务的学习。', '',
             '## 6 证据文件与后续要求','',
             '改进成功的证据应是同等任务下更多无探索安全捕获、较少真实碰撞、真实追捕轨迹及保留旧级能力；loss、实现测试和筛选视频均不能替代成功率。最终选择还需较长预算、独立种子逐级复评，以及适当的训练种子重复。', '',
             link('诊断原始数据',DIAG/'summary.json'),'',link('试验监督状态',PILOTS/'process.json'),'',
             link('可复现诊断程序',ROOT/'tools/analyze_exp009_policy.py'),'',
             link('顺序试验程序',ROOT/'tools/run_pursuit_improvement_pilots.py'),'',
             '本机同步文件可能落后于服务器，应核对更新时间与/proc真实进程；连接失败不意味着训练停止。原uav定时监控仍保持暂停，当前使用已授权的一次性实验流程。']
    if feasibility:
        text += ['', '## 7 补充诊断：任务可行性、目标网络和饱和梯度','',
                 '32个独立诊断种子68101–68132，每种方法逐回合重新设定同一环境种子并核验同一追捕机／逃逸机初始位置；第1级、原0.1秒物理／0.5秒决策、原建筑、真实碰撞提前终止、同一架追捕机连续保持2秒。没有修改位置、速度、运动方程或成功判据。', '',
                 '| 方法 | 安全捕获/32 | 碰撞/32 | 平均决策步 | 平均折扣回报 |',
                 '|---|---:|---:|---:|---:|']
        for name,mode in feasibility['modes'].items():
            text.append(f'| {name} | {mode["safe_captures"]} | {mode["collisions"]} | {mode["mean_steps"]:.2f} | {mode["mean_discounted_return"]:.2f} |')
        text += ['', 'PD跟踪控制器令期望速度为v_e+0.35(p_e−p_i)，范数限制11 m/s，再令加速度为(期望速度−当前速度)/1秒，范数限制3 m/s²；原滞后、jerk和物理子步仍由环境执行。另一对照仅增加根据最近5栋建筑和边界净空计算的速度限幅。第1级两种控制器均32次安全捕获，平均约55.6秒。', '',
                 '这证明诊断样本中的原任务可完成，也证明当前奖励允许安全追捕获得明显高于失败轨迹的回报。控制器不是RL，不计入网络训练成绩，不证明第五级可行性或32回合以外的普遍成功率。图中的建筑为俯视投影，真正净空与碰撞均由原三维模拟器判断。', '',
                 link('可行性数据及逐步轨迹',ROOT/'outputs/reviews/pursuit_feasibility_v1/results.json')]
        graph2=feasibility_figure(feasibility)
        if graph2:
            text += ['',f'![可行性诊断]({graph2.resolve().as_posix()})']
    if bootstrap:
        text += ['', '### 7.1 目标Actor滞后和平滑噪声核查','',
                 '同一32组状态分别执行当前Actor、冻结目标Actor、目标Actor加TD3平滑噪声（标准差0.2、噪声裁剪0.5、最终动作裁剪到[-1,1]）。三种策略均0次捕获、32次真实碰撞；平均回报分别约−15.92、−15.54、−15.60，初始双Critic较小Q分别约+21.91、+21.90、+21.82。这些目标策略真实回报也没有支持正Q估计。', '',
                 link('目标策略真实回报诊断',ROOT/'outputs/reviews/exp009_target_actor_diagnostic_v1/results.json')]
    if gradients:
        text += ['', '### 7.2 tanh局部梯度及探索方向核查','',
                 '两条已记录的最终模型轨迹共258个决策状态，约19.9%的动作分量达到浮点数边界（|a|≥1−1e−7）。对tanh输出a，局部导数为1−a²；uav_2的x轴平均导数为0，y轴约0.0000107。这会削弱Critic价值梯度向这些输出logit的传播；共享网络仍可能通过其他状态和输出更新，不能据此说整网全部没有梯度。', '',
                 '当策略单轴输出为+1时，加独立高斯噪声后反向（动作<0）的概率为0.5*erfc[1/(σ√2)]。σ=0.03时约6.35×10⁻²⁴⁴；σ=0.1时约7.62×10⁻²⁴；σ=0.3时约0.043%。增强噪声的试验需要实际验证是否足以改善状态／动作分布。若对照仍失败，下一步应检查避免数值饱和的Actor输出表示，而非仅继续延长失败策略。', '',
                 link('饱和局部梯度统计',ROOT/'docs/baseline_exp009_saturation_gradient_2026-10-14.json')]
    if pilot_q:
        text += ['', '## 8 已完成对照的Q校准核查','',
                 '每模型32回合、诊断种子66001、无探索，真实碰撞终止；Q与折扣回报使用该试验自身奖励。不同权重或训练预算的Q数值不能作为横向性能排名。','',
                 '| 试验 | 模型 | 捕获/32 | 初始较小Q | 实际回报 | Q偏差 | 饱和分量比例 |',
                 '|---|---|---:|---:|---:|---:|---:|']
        for number,models in pilot_q.items():
            for label,s in models.items():
                text.append(f'| {number} | {label} | {s["captures"]} | {s["mean_initial_Q_min"]:.2f} | {s["mean_discounted_return"]:.2f} | {s["mean_bias_terminal_only"]:.2f} | {s["saturation_fraction"]:.1%} |')
        text += ['', '010最终模型的动作仍明显饱和，实际回报也仍低于正Q估计。动作裁剪修复未单独解决价值估计与策略执行之间的偏差。', '',
                 link('对照Q诊断原始数据',ROOT/'outputs/reviews/exp010_012_q_diagnostic_v1/summary.json')]
    if output_probe:
        text += ['', '## 9 Actor输出表示候选与后续长实验','',
                 '直接读取冻结009/010网络最后一层的真实logit。009初态uav_2的x logit约13.46，tanh输出为float32数值1、局部导数0；同一logit的Softsign输出约0.9308、导数约0.00478。Softsign为a=z/(1+|z|)，保持[-1,1]动作范围，梯度以1/(1+|z|)²衰减。它针对数值饱和，不保证所有状态梯度都更大，也不能自动纠正错误Critic梯度。','',
                 '候选保持83/332输入、512/512网络、相同观测／任务／动力学，仅改变Actor与目标Actor的末层输出；通过进程内注册提供独立入口，历史源码默认激活保持不变。GPU1024转移验证两Critic和Actor实际更新、参数有限、有界回放、logit10/15/20处非零梯度。CPU原生策略与手动权重前向及原环境20步重放完全一致。两种激活的真实评估入口亦通过短测，并逐项核验对应初始状态一致；这些都不是训练效果证据。', '',
                 f'013流程当前状态：{next_run.get("status","未启动")}；实际选择记录：{next_run.get("selection","尚待独立验证")}。012在55万正式评估出现安全9%，因此将长实验筛选条件修订为独立验证安全≥5%，优先延续已有实际改善；课程升级继续要求安全80%/碰撞10%连续两次。67101验证后选择012配置：闭合速度权重1.0、输出仍为tanh，Softsign候选未启用。从随机权重开始300万转移，随后冻结最终模型，在67201–67300新种子组上逐级各100回合，与009冻结最终模型做完全相同初始案例的比较。', '',
                 '所有追捕动作仍由网络产生，没有接入PD追踪器、安全接管或示范初始化；逃逸机沿用原APF。该流程是一次性顺序实验，uav定时监控继续保持暂停。', '',
                 link('实际logit与导数诊断',ROOT/'outputs/reviews/actor_output_saturation_v1/results.json'),'',
                 link('013进度与最终报告',ROOT/'docs/baseline_exp013_results_2026-10-14.md')]
    if bellman:
        text += ['', '## 10 一步误差沿轨迹累积的实测核查','',
                 '诊断种子68101–68104，各模型4回合，第1级，关闭探索并按训练碰撞终止。这里是实际政策轨迹上的诊断，不是训练回放批次的loss。另对每个非终止状态用8次目标动作噪声采样，近似检查原TD3学习目标的一步残差。', '',
                 '令q_t为当前Actor真实动作下的双Critic较小值，真实奖励为r_t，真正终止后的q_T按Bellman终止条件为0。定义δ_t=q_t−r_t−γq_(t+1)，则沿完整终止回合精确有：', '',
                 r'$$q_0-\sum_{t=0}^{T-1}\gamma^t r_t=\sum_{t=0}^{T-1}\gamma^t\delta_t.$$','',
                 '009四个诊断回合的平均初态偏差约+37.54。终止一步对偏差的平均折扣贡献约−0.70，真正偏差主要来自此前的步骤；终止前预测Q约−32至−35，与实际终止奖励约−32.8接近。近89%–91%的当前政策一步残差为正，显示沿轨迹持续偏乐观。', '',
                 'γ约0.994987、回合129步时，折扣权重合计约95；每步约0.39的同向平均误差就能累积到约37的初态偏差。因此较小或下降的一步均方loss仍可能对应明显失真的长期回报。该恒等式本身是代数关系，实测误差的分布和终止前预测值才是诊断证据。', '',
                 '| 模型 | 诊断回合 | 平均初态Q偏差 | 终止一步对偏差平均占比 | 目标网络一步残差RMS |',
                 '|---|---:|---:|---:|---:|']
        for number,s in bellman.items():
            text.append(f'| {number} | {s["episodes"]} | {s["mean_bias"]:.2f} | {s["mean_terminal_bias_share"]:.1%} | {s["mean_learner_target_residual_rms"]:.2f} |')
        text += ['', '不同训练预算、不同政策状态分布和噪声采样下的残差不用于横向训练性能排名。这里未发现共享样本打包顺序错误：原实现按(batch,agent)堆叠，回放为全部Agent使用相同环境／步索引，终止布尔值按原值保存。后续仍需监测真实回报校准；若长实验效果不足，再独立研究多步训练目标或Actor启动时机，保持原物理任务和奖励公式。', '',
                 link('完整轨迹误差与终止前Q',ROOT/'outputs/reviews/baseline_bellman_profile_v1/009.json')]
        first=read(ROOT/'outputs/reviews/baseline_bellman_profile_v1/009.json')['records'][0]
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        r=np.asarray(first['mean_reward_by_step'])
        cfg=read(ROOT/'outputs/results/matd3/baseline_exp009/config.json')
        tail=np.zeros(len(r))
        value=0.
        for index in range(len(r)-1,-1,-1):
            value=r[index]+cfg['gamma']*value
            tail[index]=value
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        t=np.arange(len(r))*cfg['decision_dt']
        axes[0].plot(t,first['mean_q_by_step'],label='Predicted min Q')
        axes[0].plot(t,tail,label='Actual discounted future return')
        axes[0].set(xlabel='Simulation seconds',ylabel='Return',title='Native 009 policy, diagnostic seed 68101')
        axes[0].legend()
        axes[1].plot(t,np.cumsum(first['mean_weighted_residual_by_step']))
        axes[1].set(xlabel='Simulation seconds',ylabel='Accumulated weighted residual',title='Value bias accumulated before the first crash')
        for axis in axes:
            axis.grid(alpha=.2)
        path=ROOT/'docs/baseline_bellman_error_2026-10-14.png'
        fig.savefig(path,dpi=160)
        plt.close(fig)
        text += ['',f'![真实轨迹上的价值偏差]({path.resolve().as_posix()})']
    video=ROOT/'outputs/reviews/exp012_pilot_stage1_video_v2/level_1/success.mp4'
    if video.exists():
        text += ['', '## 11 012第1级实际安全捕获视频','',
                 '采用012_pilot真实best_stage_0模型，独立视频筛选种子69501起，种子69507的案例成功；原生网络、无探索，原环境逐动作重放一致，全程无建筑／边界接触。uav_0在73.383秒连续保持捕获；其余追捕机尚未形成完整包围，该判据仍是单机持续捕获。视频字幕已明确删除5米直接拦截，并区分课程评估安全21%与独立验证安全17%。该筛选样例不替代正式100回合成功率，也不作为013的训练或视频。', '',link('012实际成功视频',video)]
    feasibility_report=ROOT/'docs/curriculum_feasibility_2026-10-14.md'
    if feasibility_report.exists():
        text += ['', '## 12 五级课程的补充可行性核查','',
                 '在新的69601–69616诊断种子上，每级分别执行直接PD与制动PD并核验同一初始案例。制动PD第1–4级均16/16安全捕获，第5级12/16安全捕获、4/16超时且无碰撞；这些控制器结果与MATD3成绩分开。第2–5级直接PD均在约4秒撞楼，首个案例为uav_1接触building:28，提示升级时需要立即绕开出生前方障碍。', '',
                 link('五级原始案例、APF真实速度与限制',feasibility_report)]
    drift_path=ROOT/'outputs/reviews/exp013_policy_drift_v1'
    drift=read(drift_path/'summary.json')
    if drift:
        manifest=read(drift_path/'manifest.json')
        text += ['', '## 13 013改善回落时的同案例策略核查','',
                 f'冻结时训练状态为{manifest["training_status_at_freeze"]["step"]:,}转移，最后正式当前级评估为70万步安全0%。最佳与最近保存模型在69701–69716这16组相同初始状态上执行无探索、原训练碰撞终止；逐项核验完整初始观测及建筑／位置一致。模型哈希与冻结出处见manifest.json。该快照不代表之后75万步或最终模型。', '',
                 '| 冻结模型 | 安全捕获/16 | 碰撞/16 | 初始较小Q均值 | 实际折扣回报均值 | 终止回合Q偏差 | 动作分量超过0.95的比例 |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for name in ('best','recent'):
            r=drift[name]
            text.append(f'| {name} | {r["safe_captures"]} | {r["collisions"]} | {r["mean_initial_Q_min"]:.2f} | {r["mean_discounted_return"]:.2f} | {r["mean_terminal_Q_bias"]:.2f} | {r["saturation_fraction"]:.1%} |')
        text += ['', '最佳模型在两个案例中安全捕获，最近模型在同两案例中碰撞，实际折扣回报分别从55.66降至18.37、从74.70降至21.20。其余失败案例中，最近模型多数获得了较高的接近回报，造成平均回报变化较小而安全率明显下降。这显示这些案例的实际行为发生退化；不能仅归因于评估抽样。', '',
                 '当前两模型的初态Q在这组终止回合中反而低估真实回报，不能继续用009旧模型的正向Q偏差解释所有后续问题。两个模型使用各自Critic，不能据此宣称同一Critic必然偏好失败动作。动作分量超过0.95的比例下降，亦不足以证明本次退化源于更加饱和；该比例本身也不是float32梯度为零的判据。', '',
                 '同一013奖励下，第1级制动PD的另外16组安全案例平均折扣回报约103.59，说明更好的安全追踪路径可获得显著更高回报；这只是控制器可行性证据，不提供网络已经学会的证据。下一项优化应先等完整训练与独立复评，再针对安全动作学习和价值校准设计单项对照。', '',
                 link('配对策略原始轨迹与回报',drift_path/'summary.json'),'',link('冻结模型哈希与训练时点',drift_path/'manifest.json')]
    counterexample_path=ROOT/'outputs/reviews/formation_objective_counterexamples_v1/results.json'
    counterexample=read(counterexample_path)
    if counterexample:
        text += ['', '## 14 最近队友夹角奖励不等价于包围目标：实现级反例','',
                 '直接调用当前源码的_pursuit_reward_terms，使用明确构造的瞬时几何状态；这是公式核查，不是训练轨迹、成功率或对模拟器的改写。所有例子都设逃逸机速度0、追捕机以4 m/s朝目标运动，因此每机闭合速度均4 m/s、r_purs_raw均8；只比较几何项。', '',
                 '| 几何状态 | 平均最近队友夹角 | 原始队形奖励（弧度） | 当前0.5秒决策的队形奖励 |',
                 '|---|---:|---:|---:|']
        labels={'square_around_target':'目标周围的平面正方形','tetrahedron_around_target':'包含目标的三维正四面体',
                'one_sided_near':'全部位于目标一侧、距目标22–45米','one_sided_far':'全部位于目标一侧、距目标277–300米'}
        for r in counterexample['records']:
            text.append(f'| {labels[r["name"]]} | {r["formation_mean_degrees"]:.3f}° | {r["formation_raw_radians"]:.6f} | {r["decision_formation_reward"]:.6f} |')
        text += ['', '一侧直线例的相邻间隔为3、9、11米，最近队友选择没有距离并列：最靠后者指向前方队友，夹角0°；其余三架指向后方最近队友，夹角180°，平均135°。它们完全没有从不同方向包围目标，却获得比正方形45°和正四面体35.264°更大的队形奖励。远近两种一侧状态奖励相同，说明该项也没有围捕距离门控。', '',
                 '这进一步限定了用户原公式的含义：它能改变最近队友与目标的相对夹角，不能将增大的数值直接解释为空间包围更完整。当前仍保留已确认的公式。其当前0.5秒奖励差仅约0.008，与相同闭合速度例中的追捕奖励0.4相比很小，尚无证据称它是013安全率波动的主要原因。', '',
                 '如果下一阶段研究真正的多机围捕，应先明确是否要求至少多架同时接近、目标位于多机几何包络中或各逃逸方向被覆盖，再使成功判据与队形奖励对应；不能仅把现有单机保持视频称为完整围捕。', '',
                 link('精确坐标、速度、最近队友及实现哈希',counterexample_path)]
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
        records=counterexample['records']
        axes[0].bar(['Square','Tetrahedron','One side near','One side far'],[r['formation_mean_degrees'] for r in records])
        axes[0].set(ylabel='Mean nearest-teammate angle (degrees)',title='Static formula examples; identical closing speeds')
        axes[0].tick_params(axis='x',labelrotation=15)
        for name,color in [('square_around_target','tab:blue'),('one_sided_near','tab:orange')]:
            r=next(r for r in records if r['name']==name)
            p=np.asarray(r['positions'])-np.asarray(r['target'])
            square=name=='square_around_target'
            axes[1].scatter(p[:,0],p[:,1],label='Square: surrounds target' if square else 'Line: one side of target',color=color)
            offsets=[(-10,22),(2,-18),(0,18),(0,-18)]
            for index,point in enumerate(p):
                axes[1].annotate(('S' if square else 'L')+str(index+1),point[:2],
                    xytext=(3,3) if square else offsets[index],textcoords='offset points',fontsize=8,color=color,
                    arrowprops=None if square else dict(arrowstyle='-',color=color,lw=.6))
        axes[1].scatter(0,0,marker='*',s=140,color='black',label='Target')
        axes[1].set(xlabel='X relative to target (m)',ylabel='Y relative to target (m)',aspect='equal',title='One-sided geometry obtains a larger angle reward')
        axes[1].legend(fontsize=7)
        for axis in axes:axis.grid(alpha=.2)
        path=ROOT/'docs/formation_objective_counterexamples_2026-10-14.png'
        fig.savefig(path,dpi=160)
        plt.close(fig)
        text += ['',f'![夹角奖励的几何反例]({path.resolve().as_posix()})']
    graph=figure()
    if graph:
        text += ['',f'![真实轨迹、Q偏差和动作]({graph.resolve().as_posix()})']
    STEM.with_suffix('.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(str(STEM.with_suffix('.md')),flush=True)


if __name__=='__main__':
    main()
