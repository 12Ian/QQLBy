"""Build an honest progress/final report from the selected pure-RL experiment."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'outputs/experiments/exp013'
RESULT=ROOT/'outputs/results/matd3/baseline_exp013'
STEM=ROOT/'docs/baseline_exp013_results_2026-10-14'


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def rows(path):
    return list(csv.DictReader(path.open(encoding='utf-8'))) if path.exists() else []


def link(label,path):
    return f'[{label}]({path.resolve().as_posix()})'


def main():
    process=read(EXP/'process.json')
    metadata=read(EXP/'metadata.json')
    status=read(RESULT/'training_status.json')
    summary=read(RESULT/'baseline_summary.json')
    cfg=read(RESULT/'config.json')
    updates=rows(RESULT/'training_updates.csv')
    episodes=rows(RESULT/'training_episodes.csv')
    evaluations=rows(RESULT/'curriculum_eval.csv')
    final=[]
    reference=[]
    for level in range(5):
        item=read(EXP/f'final_review/final_level{level+1}.json')
        if item:
            final.append(item)
        old=read(EXP/f'final_review/baseline009_level{level+1}.json')
        if old:
            reference.append(old)
    steps=summary.get('environment_steps',max([int(status.get('step',0))]+[int(row['step']) for row in updates]))
    evidence=dict(updated_at=datetime.now(timezone.utc).isoformat(),process=process,metadata=metadata,
        observed_steps=steps,training_summary=summary,training_status=status,final_review=final,reference_review=reference,
        updates=updates[-1] if updates else {},configuration=cfg)
    STEM.with_suffix('.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False),encoding='utf-8')
    text=['# 013纯MATD3改进实验：设置、进度与最终结果','',
        f'更新时间：{evidence["updated_at"]}；状态：{process.get("status","未启动")}；已观测转移：{steps:,} / 3,000,000。','',
        '进行中数据不能当最终结果；升级由原安全门槛决定，未通过门槛不强行升级。', '',
        '## 1 方案选择与算法口径','',
        '先完成010、011、012三项75万转移对照及67101种子独立100回合验证。若某方案验证安全成功率≥5%，优先保留真实的渐进改善，按安全率最高、碰撞最低、捕获率最高选择方案做完整300万转移训练。5%只用于筛选长实验，课程升级仍为连续两次安全≥80%、碰撞≤10%。若没有重复出安全改善，则采用010仅将Actor及目标Actor输出改为Softsign的候选。', '',
        f'实际选择：{metadata.get("selection","尚待三个对照完成后确定")}。从随机权重、空回放池、新优化器重新开始；不使用教师、PD控制器、示范初始化或旧训练模型。', '',
        '仍复用XuanCe共享MLP Actor、集中式双Critic的MATD3；三维动作范围[-1,1]、83/332维网络输入、512/512隐藏层、观测内容与全局目标信息保持一致。若采用Softsign，只改变有界输出a=z/(1+|z|)，导数为1/(1+|z|)²；不提供自动朝向目标或人工安全控制。', '',
        '009/010的实际logit约10–15时，float32 tanh导数为0；Softsign可保留约0.008–0.004的导数。数值验证和GPU测试只证明实现及梯度有效，不证明训练成绩提高。', '',
        '## 2 环境、运动学、课程和奖励','',
        '城市与出生设置沿用009：固定20/40/60/80/100栋；追捕机西侧集结，逃逸机东侧自由空间出生，最小初距450米；普通/高层/地标比例50/40/10。物理0.1秒、决策0.5秒、800决策步=400秒。追捕机速度11、加速度3、jerk6；逃逸机各级最高速度3/5/7/8/12、加速度4、jerk8、感知100米；GAT关闭。', '',
        '只保留同一架追捕机在该级半径50/35/25/20/15米内连续保持2秒，捕获不要求全队此前无碰撞；安全成功另行统计。训练真实建筑/边界碰撞终止，评估关闭提前终止并统计完整回合。', '',
        '保留用户闭合速度公式“平均＋最小”和最近队友夹角平均公式；progress/distance/teammate为0，机间碰撞不计；建筑/边界稠密制动余量惩罚、真实接触及全队失败惩罚、捕获奖励保留。', '',
        f'实际追捕/队形权重：{cfg.get("baseline_pursuit_weight","待配置")}/{cfg.get("baseline_formation_weight","待配置")}；总体缩放{cfg.get("baseline_reward_scale","待配置")}；探索{cfg.get("start_noise","待配置")}→{cfg.get("end_noise","待配置")}，衰减{cfg.get("noise_decay_steps","待配置")}转移。实际配置JSON作为所有参数的权威记录。', '',
        '每级至少100000转移，每50000转移评估100回合，连续两次安全≥80%、碰撞≤10%升级；每次追加所有已通过旧级的retention复测。没有混合课程回放。', '',
        '## 3 各级真实训练统计','',
        '| 级别 | 完整回合 | 完整回合转移 | 捕获 | 安全捕获 | 最近当前级正式评估捕获/安全/碰撞 |',
        '|---|---:|---:|---:|---:|---|']
    complete=0
    for level in range(5):
        selected=[r for r in episodes if int(r['level'])==level]
        transfers=sum(int(r['episode_steps']) for r in selected)
        complete+=transfers
        current=[r for r in evaluations if r['phase']=='current' and int(r['level'])==level]
        rates='未进入正式训练评估'
        if current:
            ev=current[-1]
            rates='/'.join(f'{float(ev[k]):.0%}' for k in ('success_rate','safe_success_rate','collision_rate'))+f'（{int(ev["step"]):,}步）'
        text.append(f'| {level+1} | {len(selected):,} | {transfers:,} | {sum(int(r["success"]) for r in selected)} | {sum(int(r["safe_success"]) for r in selected)} | {rates} |')
    text += ['',f'完整回合转移合计{complete:,}；总转移{steps:,}；正在进行的回合没有计入完整回合转移。', '',
        f'实际Critic/Actor优化次数：{updates[-1]["updates"] if updates else 0}/{updates[-1]["actor_updates"] if updates else 0}；以training_updates.csv为准。','',
        '## 4 最终单一模型逐级复评','',
        '最终模型冻结、核验源文件和模型SHA256后，用新盲测种子组67201–67300，五级各100回合、8环境批量策略推理、无探索、最长800步、全回合碰撞统计。每回合单独设定种子，避免不同策略回合长度导致异步环境采样到不同案例。组内种子没有用于013调参；无论训练到哪一级，都如实复评。', '',
        '| 级别 | 捕获率 | 安全成功率 | 碰撞率 | 建筑接触率 | 边界接触率 |',
        '|---|---:|---:|---:|---:|---:|']
    for item in final:
        m=item['metrics']
        text.append(f'| {item["level"]+1} | {m["success_rate"]:.0%} | {m["safe_success_rate"]:.0%} | {m["collision_rate"]:.0%} | {m["building_collision_rate"]:.0%} | {m["boundary_collision_rate"]:.0%} |')
    if not final:
        text += ['', '尚未完成最终复评。']
    if reference:
        text += ['', '### 与009冻结最终模型逐案例比较','',
                 '009也在同一新盲测种子组、相同任务参数和初始状态下评估；每级100个案例的完整初始状态逐项核验，审计见pairing_audit.json。这里的009成绩属于新复评，不能以其原64001组或训练内成绩代替。','',
                 '| 级别 | 009捕获/安全/碰撞 | 013捕获/安全/碰撞 |',
                 '|---|---|---|']
        for old in reference:
            new=next((x for x in final if x['level']==old['level']),None)
            if new:
                fields=('success_rate','safe_success_rate','collision_rate')
                left='/'.join(f'{old["metrics"][k]:.0%}' for k in fields)
                right='/'.join(f'{new["metrics"][k]:.0%}' for k in fields)
                text.append(f'| {old["level"]+1} | {left} | {right} |')
    text += ['', '## 5 真实成功视频与限制','']
    videos=list((ROOT/'outputs/reviews').glob('exp013_stage*_video_v1/level_*/success.mp4'))
    if videos:
        for p in videos:
            text += [link('原生网络无碰撞成功视频',p)]
    else:
        text += ['当前没有已验证的本轮成功视频。无安全案例或未到达该级时不编造样例。']
    text += ['', '视频使用实际保存的best_stage模型，CPU原生MATD3策略，无探索，原环境逐动作重放一致，核验模型和视频哈希。筛选样例不是成功率估计。', '',
        '结果仅能证明本轮配置、固定城市和给定评估种子上的表现。若尚未稳定通过课程或最终复评仍差，目标继续进行。后续需要更多训练种子、新地图、Q与真实回报校准以及实际四机围捕判据；连续2秒判据本身不保证四机包围。', '',
        link('原始失败分析与对照',ROOT/'docs/baseline_exp009_deep_analysis_2026-10-14.md')]
    for label,path in [('真实配置',RESULT/'config.json'),('过程状态',EXP/'process.json'),
                       ('训练转移与优化记录',RESULT/'training_updates.csv'),('正式评估与旧级复测',RESULT/'curriculum_eval.csv')]:
        if path.exists():
            text += ['',link(label,path)]
    if evaluations:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,4.7),constrained_layout=True)
        current=[r for r in evaluations if r['phase']=='current']
        for field,label in [('success_rate','Capture'),('safe_success_rate','Safe capture'),('collision_rate','Collision')]:
            ax.plot([int(r['step']) for r in current],[100*float(r[field]) for r in current],label=label)
        for r in current:
            ax.annotate('L'+str(int(r['level'])+1),(int(r['step']),100*float(r['safe_success_rate'])),fontsize=7)
        ax.set(xlabel='Environment transitions',ylabel='Percent',ylim=(-3,103),title='Current-level formal evaluation: 100 episodes')
        ax.grid(alpha=.2)
        ax.legend()
        path=ROOT/'docs/baseline_exp013_curves_2026-10-14.png'
        fig.savefig(path,dpi=160)
        plt.close(fig)
        text += ['',f'![真实正式评估曲线]({path.resolve().as_posix()})']
    STEM.with_suffix('.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=process.get('status'),steps=steps,output=str(STEM.with_suffix('.md')))),flush=True)


if __name__=='__main__':
    main()
