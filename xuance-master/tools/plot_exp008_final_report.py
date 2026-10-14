"""Plot completed exp008 records without changing training artifacts."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
RESULT = ROOT / 'outputs/results/matd3/baseline_exp008'
REVIEW = ROOT / 'outputs/reviews/exp008_final_v1'
font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False,
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

def rows(name):
    with (RESULT / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

evals = [r for r in rows('curriculum_eval.csv') if r['phase'] == 'current']
episodes = rows('training_episodes.csv')
updates = rows('training_updates.csv')
edges = [0, 1.45, 2.05, 2.20, 2.75, 3.0]
colors = ['#1764AB', '#E68A24', '#2F8C62', '#9666B3', '#C3454B']

def decorate(ax):
    for level, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        ax.axvspan(lo, hi, color=colors[level], alpha=.055)
        if level:
            ax.axvline(lo, color='#7F8791', lw=.8, ls=':')
    ax.set_xlim(0, 3)
    ax.grid(alpha=.16)
    ax.set_xlabel('累计训练环境转移  百万')

fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
for level in range(5):
    rr = [r for r in evals if int(r['level']) == level]
    x = [int(r['step']) / 1e6 for r in rr]
    axes[0].plot(x, [float(r['safe_success_rate'])*100 for r in rr],
                 'o-', color=colors[level], markersize=3, label=f'第{level+1}级安全成功')
    axes[0].plot(x, [float(r['success_rate'])*100 for r in rr],
                 ls='--', color=colors[level], alpha=.55, lw=1)
all_x = [int(r['step']) / 1e6 for r in evals]
for key, label, color in [('collision_rate', '总碰撞', '#C3454B'),
                          ('building_collision_rate', '建筑接触', '#9A6D3A'),
                          ('boundary_collision_rate', '边界接触', '#1764AB')]:
    axes[1].plot(all_x, [float(r[key])*100 for r in evals], lw=1.6, label=label, color=color)
axes[0].axhline(80, color='#555555', lw=1, ls=':', label='安全成功门槛80%')
axes[1].axhline(10, color='#555555', lw=1, ls=':', label='碰撞门槛10%')
for ax in axes:
    decorate(ax)
    ax.set_ylim(-3, 105)
    ax.set_ylabel('100回合比例  %')
    ax.legend(fontsize=8, ncol=3, loc='lower right')
axes[0].set_title('008训练期间当前课程正式评估  实线为安全成功  虚线为捕获')
axes[1].set_title('完整评估回合中的接触  建筑和边界允许重叠')
fig.tight_layout()
fig.savefig(DOCS / 'baseline_exp008_final_curriculum_2026-10-13.png', dpi=185)
plt.close(fig)

terms = ['r_progress', 'r_obstacle', 'r_boundary', 'r_collision', 'r_terminal', 'r_failure']
termnames = ['接近势函数', '建筑预警', '边界预警', '真实接触', '捕获事件', '全队失败']
bins = []
for endpoint in range(50000, 3000001, 50000):
    rr = [r for r in episodes if endpoint-50000 < int(r['step']) <= endpoint]
    if rr:
        bins.append({'step': endpoint, 'episodes': len(rr), **{
            key: float(np.mean([float(r[key]) for r in rr])) for key in terms}})
fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
for key, label in zip(terms, termnames):
    axes[0].plot([r['step']/1e6 for r in bins], [r[key] for r in bins], label=label, lw=1.4)
axes[0].set_title('训练奖励分项  每5万转移窗口的完整回合均值  已乘0.1')
axes[0].set_ylabel('每队员回合累计奖励')
axes[0].legend(fontsize=8, ncol=3)
x = np.array([int(r['step'])/1e6 for r in updates])
axes[1].plot(x, [float(r['loss_actor_mean']) for r in updates], color='#1764AB', label='Actor loss')
axes[1].plot(x, [float(r['predictQ_mean']) for r in updates], color='#2F8C62', label='预测 min Q')
axes[1].set_ylabel('Actor loss 和 Q')
other = axes[1].twinx()
other.plot(x, [float(r['loss_critic_mean']) for r in updates], color='#C3454B', alpha=.7, lw=1, label='Critic loss')
other.set_yscale('symlog', linthresh=1)
other.set_ylabel('Critic loss  对称对数尺度')
axes[1].legend(loc='upper left', fontsize=8)
other.legend(loc='upper right', fontsize=8)
axes[1].set_title('优化日志窗口均值  有限 loss 不表示策略已经收敛')
for ax in axes:
    decorate(ax)
fig.tight_layout()
fig.savefig(DOCS / 'baseline_exp008_final_rewards_and_losses_2026-10-13.png', dpi=185)
plt.close(fig)

final = [json.loads((REVIEW / f'final008_level{i}.json').read_text())['metrics'] for i in range(1, 6)]
initial = [json.loads((REVIEW / f'initial007_level{i}.json').read_text())['metrics'] for i in range(1, 6)]
best = [json.loads((REVIEW / f'beststage_level{i}.json').read_text())['metrics'] for i in range(1, 4)]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
xx = np.arange(5)
axes[0].bar(xx-.17, [r['safe_success_rate']*100 for r in initial], .34, label='007初始化权重  新布局', color='#AEB8C3')
axes[0].bar(xx+.17, [r['safe_success_rate']*100 for r in final], .34, label='008最终权重', color='#1764AB')
axes[0].set_xticks(xx, [f'第{i}级' for i in range(1,6)])
axes[0].set_title('最终模型逐级复评  种子60001')
xx = np.arange(3)
axes[1].bar(xx-.17, [r['safe_success_rate']*100 for r in best], .34, label='各级历史最好权重', color='#2F8C62')
axes[1].bar(xx+.17, [r['safe_success_rate']*100 for r in final[:3]], .34, label='008最终权重', color='#C3454B')
axes[1].set_xticks(xx, [f'第{i}级' for i in range(1,4)])
axes[1].set_title('旧级能力保留对照  同一种子60001')
for ax in axes:
    ax.set_ylim(0, 112)
    ax.set_ylabel('100回合安全成功率  %')
    ax.axhline(80, color='#555555', lw=1, ls=':')
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.09), ncol=2)
    ax.grid(axis='y', alpha=.15)
    for bar in ax.patches:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{bar.get_height():.0f}%', ha='center', fontsize=9)
fig.tight_layout()
fig.savefig(DOCS / 'baseline_exp008_final_retention_2026-10-13.png', dpi=185)
plt.close(fig)

(DOCS / 'baseline_exp008_final_plot_data_2026-10-13.json').write_text(
    json.dumps({'training_reward_bins': bins, 'final_metrics': final, 'initial_metrics': initial,
                'beststage_retention_metrics': best, 'course_transition_steps': edges,
                'training_updates_last': updates[-1]}, ensure_ascii=False, indent=2), encoding='utf-8')
print('Three final figures and their source data written.')
