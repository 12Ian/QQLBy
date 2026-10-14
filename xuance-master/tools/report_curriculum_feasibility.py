"""Report controller diagnostics separately from learned policy performance."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reviews/curriculum_feasibility_v2'
STEM=ROOT/'docs/curriculum_feasibility_2026-10-14'


def main():
    result=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    process=json.loads((OUT/'process.json').read_text(encoding='utf-8'))
    assert process['status']=='completed' and process['exit_code']==0
    assert all(s['paired_initial_states_exact'] for s in result['levels'])
    assert result['diagnostic_controllers_not_RL'] and not result['simulator_modified']
    count=result['episodes_per_controller_per_level']
    text=['# 五级课程的物理可行性与早期障碍诊断','',
          '本诊断采用013的冻结任务配置，保留原环境、物理0.1秒／决策0.5秒、速度／加速度／jerk／滞后、建筑、出生、连续2秒捕获与真实碰撞。没有接入训练、提供示范数据或改写网络动作。以下数据属于人工控制器诊断，须与MATD3学习成绩分开。','',
          f'每级各{count}个诊断种子69601–69616；两种控制器逐案例核验完全相同的建筑几何、初始位置、初始速度和观测。发生真实建筑或边界碰撞便终止，符合训练制度；这些小样本不是正式100回合无探索评估，也不能证明新地图上的普遍可行性。','',
          '## 1 结果','',
          '| 级别 | 建筑数 | 逃逸机最高速度 m/s | 保持半径 m | 直接PD安全捕获 | 制动PD安全捕获 | 制动PD碰撞 | 制动PD超时 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for stage in result['levels']:
        a=stage['controllers']['pd']
        b=stage['controllers']['pd_clearance']
        course=stage['course']
        text.append(f'| {stage["level"]} | {course["building_count"]} | {course["target_max_speed"]:g} | {course["capture_radius"]:g} | {a["safe_captures"]}/{count} | {b["safe_captures"]}/{count} | {b["collisions"]}/{count} | {sum(r["steps"]==800 and not r["success"] for r in b["records"])}/{count} |')
    text += ['', '第1–4级制动PD均16次安全捕获；第5级12次安全捕获、4次400秒超时，零碰撞。这支持在当前固定城市与这组初始案例中存在安全捕获路径；不能把这些控制器成功归于MATD3，也不能把单机连续保持称为四机完整包围。','',
             '## 2 控制器具体做法','',
             '直接PD令期望速度为v_e+0.35(p_e−p_i)，其范数限制为11 m/s；加速度命令为(期望速度−当前速度)/1秒，范数限制3 m/s²。执行时仍经过原模拟器的0.5秒加速度滞后、6 m/s³ jerk限制和0.1秒物理积分。','',
             '制动PD在直接PD基础上，用原最近5栋建筑的真实三维净空与6个空域边界限制朝障碍方向的速度。按d−20=v·2+v²/(2a)，a=3 m/s²计算v_max=a·[sqrt(4+2·max(d−20,0)/a)−2]；该控制器取2秒反应余量与20米距离余量。这是诊断用参数，不是对训练奖励或动力学的修改。','',
             '它没有完整地图路径搜索、四机围捕规划或最优性保证。控制器可使用目标位置／速度以及最近建筑几何，属于当前网络观测包含的信息类别；实现的计算形式与网络不同。','',
             '## 3 课程跳变的直接困难','',
             '从第2级起，直接PD的16个案例全部在第8个决策步（约4秒）撞楼；首个案例明确为uav_1接触building:28。第2–5级这栋建筑及其邻近出生点的配置保留，表现不是逃逸机提前逃脱或训练loss异常。出生时处于合法自由空间，不代表沿目标直线加速的路径畅通。','',
             '第1级主要能够直接接近目标；后续级别必须在刚出生时便根据净空、速度和制动余量决定绕行／飞越／减速。若013在升级后安全率急跌，应优先核查这处早期接触及真实动作，而不是直接判断高层建筑太多或逃逸机速度太快。当前不改正在训练的冻结课程。','',
             '## 4 逃逸机是否真的积极加速','',
             '| 级别 | 配置最高速度 m/s | 制动PD案例平均峰值速度 m/s | 最小／最大峰值 m/s | 平均进入100米感知区的决策比例 |',
             '|---|---:|---:|---:|---:|']
    for s in result['levels']:
        c=s['controllers']['pd_clearance']
        peaks=[r['target_peak_speed'] for r in c['records']]
        text.append(f'| {s["level"]} | {s["course"]["target_max_speed"]:g} | {c["mean_target_peak_speed"]:.3f} | {min(peaks):.3f}／{max(peaks):.3f} | {c["mean_target_sensed_fraction"]:.1%} |')
    text += ['', '原APF在至少一架追捕机进入100米后将期望速度切换到该级最高速度；此前按该级最低速度巡航。加速度4 m/s²、jerk8和滞后使真实速度逐渐变化，不会瞬间变成12 m/s。表中峰值是决策末的真实速度采样，感知比例是决策末最短距离小于100米的比例。','',
             '第5级多数案例达到约12 m/s，但追捕机仅11 m/s，因此不能只依赖长期直线尾追；目标受建筑、边界和回中APF影响转向时仍存在捕获机会。12/16的控制器安全成功支持这些案例可完成，4个超时也表明难度确实升高。该诊断尚不证明逃逸机的APF是最优逃逸策略。','',
             '## 5 对后续改进的约束','',
             '先完成013的原300万转移预算与五级同案例独立复评，再根据真实碰撞来源、旧级retention和Q校准决定下一项对照。当前增加闭合速度权重已产生原生RL的部分安全改善，尚需观察稳定性。','',
             '若第2级主要在出生前方撞楼，优先检查网络是否利用净空／制动观测，以及共享Actor饱和和回放中新旧课程样本的作用。若第5级主要超时，优先分析目标转向时机和多机协同切入；将四机空间围捕作为另一个明确判据讨论，避免用当前单机2秒捕获替代。','',
             f'[原始逐案例数据与轨迹]({(OUT/"results.json").resolve().as_posix()})','',
             f'[可复现诊断程序]({(ROOT/"tools/diagnose_curriculum_feasibility.py").resolve().as_posix()})']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    levels=np.arange(1,6)
    for mode,label,offset in [('pd','Direct PD',-.16),('pd_clearance','PD with braking clearance',.16)]:
        rates=[100*s['controllers'][mode]['safe_captures']/count for s in result['levels']]
        axes[0].bar(levels+offset,rates,width=.32,label=label)
    axes[0].set(xticks=levels,xlabel='Course level',ylabel='Safe captures (%)',ylim=(0,110),title='Diagnostic controllers; 16 paired cases per level')
    axes[0].legend(fontsize=8)
    axes[1].plot(levels,[s['course']['target_max_speed'] for s in result['levels']],label='Configured evader maximum')
    axes[1].plot(levels,[s['controllers']['pd_clearance']['mean_target_peak_speed'] for s in result['levels']],marker='o',label='Mean observed evader peak')
    axes[1].axhline(11,color='gray',linestyle='--',label='Pursuer speed limit')
    axes[1].set(xticks=levels,xlabel='Course level',ylabel='Speed (m/s)',title='Unmodified APF evader actively accelerates')
    axes[1].legend(fontsize=8)
    for axis in axes:axis.grid(axis='y',alpha=.2)
    fig.savefig(STEM.with_suffix('.png'),dpi=160)
    plt.close(fig)
    text+=['',f'![五级控制器诊断]({STEM.with_suffix(".png").resolve().as_posix()})']
    STEM.with_suffix('.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    STEM.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(STEM.with_suffix('.md'),flush=True)


if __name__=='__main__':
    main()
