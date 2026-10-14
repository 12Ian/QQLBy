"""Report live 014 evidence and, with --wait-final, review its final checkpoint."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'outputs/experiments/exp014'
RESULT = ROOT / 'outputs/results/matd3/baseline_exp014'
REVIEW = ROOT / 'outputs/reviews/exp014_final_v1'
STEM = ROOT / 'docs/baseline_exp014_results_2026-10-14'


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def rows(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def live_pid(pid, script):
    path = Path('/proc') / str(pid) / 'cmdline'
    return path.exists() and any(Path(a.decode(errors='replace')).name == script
        for a in path.read_bytes().split(b'\x00') if a)


def report():
    process = read(EXP / 'process.json')
    status = read(RESULT / 'training_status.json')
    summary = read(RESULT / 'baseline_summary.json')
    updates = rows(RESULT / 'training_updates.csv')
    evaluations = rows(RESULT / 'curriculum_eval.csv')
    episodes = rows(RESULT / 'training_episodes.csv')
    latest = updates[-1] if updates else {}
    step = summary.get('environment_steps', max(int(status.get('step', 0)), int(latest.get('step', 0))))
    counters = {key: max(int(status.get(key, 0)), int(latest.get(key, 0)))
                for key in ('updates', 'actor_updates')}
    live = live_pid(process.get('child_pid'), 'run_baseline.py') if Path('/proc').exists() else None
    levels = []
    for level in range(5):
        selected = [r for r in episodes if int(r['level']) == level]
        current = [r for r in evaluations if r['phase'] == 'current' and int(r['level']) == level]
        retention = [r for r in evaluations if r['phase'] == 'retention' and int(r['level']) == level]
        levels.append(dict(level=level+1, complete_episodes=len(selected),
            complete_transfers=sum(int(r['episode_steps']) for r in selected),
            captures=sum(int(r['success']) for r in selected),
            safe_captures=sum(int(r['safe_success']) for r in selected),
            latest_current=current[-1] if current else None,
            latest_retention=retention[-1] if retention else None))
    final = [read(REVIEW / f'final_level{i+1}.json') for i in range(5)]
    evidence = dict(updated_at=datetime.now(timezone.utc).isoformat(), observed_steps=step,
        budget=6000000, process=process, training_process_live=live, training_status=status,
        updates_observed=counters, initialization=read(RESULT / 'initialization_audit.json'),
        levels=levels, evaluations=evaluations, final_review=final,
        final_review_process=read(REVIEW / 'process.json'), training_summary=summary)
    write_json(STEM.with_suffix('.json'), evidence)
    text = ['# 实验014训练与评估结果', '', f'更新时间：{evidence["updated_at"]}。', '',
        f'已观测环境转移：{step:,}/6,000,000；记录状态：{process.get("status") }；服务器实际训练进程：{live}。', '',
        '007最终模型全策略迁移初始化，新优化器、新回放池；从第一级开始。夹角奖励0，逃逸机感知200米，闭合速度权重1.0，总缩放0.1；物理0.1秒、决策0.5秒，其他任务参数沿用013。', '',
        '捕获条件为同一追捕机连续保持2秒，不要求全队此前无碰撞。捕获率与安全成功率分别统计；正式评估无探索、100回合、最长800步，并统计整个回合的碰撞。007原任务94%不是本轮成绩。', '',
        '| 级别 | 完整训练回合 | 完整回合转移 | 训练捕获/安全捕获 | 最近当前级评估捕获/安全/碰撞 |',
        '|---|---:|---:|---:|---|']
    for item in levels:
        ev = item['latest_current']
        rates = '/'.join(f'{float(ev[k]):.0%}' for k in ('success_rate', 'safe_success_rate', 'collision_rate')) + f'（{int(ev["step"]):,}步）' if ev else '未进入'
        text.append(f'| {item["level"]} | {item["complete_episodes"]:,} | {item["complete_transfers"]:,} | {item["captures"]}/{item["safe_captures"]} | {rates} |')
    text += ['', f'完整回合转移合计{sum(r["complete_transfers"] for r in levels):,}，进行中的回合未计入。最近记录的Critic/Actor累计更新：{counters["updates"]:,}/{counters["actor_updates"]:,}；训练进行中，计数可能落后于真实执行。', '',
        '## 最终单一模型五级复评', '',
        '训练正常结束后核验训练源哈希、冻结最终模型并记录SHA256；每级100回合、8并行、种子71001、无探索、800步上限、完整回合碰撞统计。这些是复评设置；结果出现前不能称为已完成。', '',
        '| 级别 | 捕获率 | 安全成功率 | 碰撞率 |', '|---|---:|---:|---:|']
    for item in final:
        if item:
            m = item['metrics']
            text.append(f'| {item["level"]+1} | {m["success_rate"]:.0%} | {m["safe_success_rate"]:.0%} | {m["collision_rate"]:.0%} |')
    if not any(final):
        text += ['', '最终复评尚未完成。']
    text += ['', '## 原始记录', '']
    for label, path in [('修改说明', ROOT / 'docs/baseline_exp014_changes_2026-10-14.md'),
                        ('真实配置', RESULT / 'config.json'), ('初始化审计', RESULT / 'initialization_audit.json'),
                        ('课程评估与旧级复测', RESULT / 'curriculum_eval.csv'),
                        ('训练回合', RESULT / 'training_episodes.csv'), ('优化更新', RESULT / 'training_updates.csv')]:
        text.append(f'- [{label}]({path.resolve().as_posix()})')
    STEM.with_suffix('.md').write_text('\n'.join(text) + '\n')
    print(json.dumps(dict(step=step, live=live, level=status.get('level'), report=str(STEM.with_suffix('.md')))), flush=True)


def wait_final():
    REVIEW.mkdir(parents=True, exist_ok=False)
    state = dict(status='waiting_training', reviewer_pid=os.getpid(), seed=71001,
                 episodes_per_level=100, parallels=8, completed_levels=[])
    write_json(REVIEW / 'process.json', state)
    write_json(REVIEW / 'reviewer_source.json', dict(path=str(Path(__file__).resolve()),
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    try:
        while True:
            process = read(EXP / 'process.json')
            if process.get('status') in ('completed', 'failed'):
                break
            if not live_pid(process.get('supervisor_pid'), 'start_exp014.py'):
                raise RuntimeError('Training supervisor disappeared without a recorded exit; no restart')
            report()
            time.sleep(60)
        assert process.get('exit_code') == 0, f'Training failed: {process}'
        summary = read(RESULT / 'baseline_summary.json')
        assert summary['environment_steps'] == 6000000
        metadata = read(EXP / 'metadata.json')
        assert all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == h
                   for n, h in metadata['source_hashes'].items()), 'Training sources changed'
        folders = list((ROOT / 'outputs/models/matd3/baseline_exp014').glob('seed_*'))
        assert len(folders) == 1
        model = REVIEW / 'final.pth'
        shutil.copy2(folders[0] / 'final_train_model.pth', model)
        configuration = ROOT / 'experiments/baseline_exp014.yaml'
        write_json(REVIEW / 'manifest.json', dict(model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256(configuration.read_bytes()).hexdigest(), source_hashes_verified=True,
            seed=71001, episodes_per_level=100, exploration=False, collision_termination=False, max_steps=800))
        state['status'] = 'evaluating'
        write_json(REVIEW / 'process.json', state)
        environment = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        for level in range(5):
            output = REVIEW / f'final_level{level+1}.json'
            command = [sys.executable, '-u', 'examples/evaluate_baseline_checkpoint.py',
                '--config', str(configuration), '--checkpoint', str(model), '--env-seed', '71001',
                '--episodes', '100', '--parallels', '8', '--level', str(level), '--device', 'cuda:0', '--output', str(output)]
            with (REVIEW / f'level{level+1}.log').open('x') as log:
                worker = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                state.update(evaluation_pid=worker.pid, current_level=level+1)
                write_json(REVIEW / 'process.json', state)
                code = worker.wait()
            assert code == 0, f'Level {level+1} evaluation failed: {code}'
            state['completed_levels'].append(level+1)
            write_json(REVIEW / 'process.json', state)
            report()
        state.update(status='completed', finished_at=datetime.now(timezone.utc).isoformat())
        write_json(REVIEW / 'process.json', state)
    except BaseException as error:
        state.update(status='failed', error=repr(error), finished_at=datetime.now(timezone.utc).isoformat())
        write_json(REVIEW / 'process.json', state)
        raise
    finally:
        report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-final', action='store_true')
    arguments = parser.parse_args()
    wait_final() if arguments.wait_final else report()
