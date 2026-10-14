"""Freeze the user-selected 014 baseline without altering experiment artifacts."""
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'outputs/acceptance/exp014_baseline_v1'
CORE_ZIP = DEST.with_name(DEST.name + '_core.zip')
TB_ZIP = DEST.with_name(DEST.name + '_tensorboard.zip')
EXP = ROOT / 'outputs/experiments/exp014'
RESULT = ROOT / 'outputs/results/matd3/baseline_exp014'
REVIEW = ROOT / 'outputs/reviews/exp014_final_v1'
MOTION = ROOT / 'outputs/reviews/exp014_motion_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    assert not any(p.exists() for p in (DEST, CORE_ZIP, TB_ZIP)), 'Preserve existing archive'
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    assert branch == 'feat-coverage-reaward'
    process, summary, metadata = read(EXP/'process.json'), read(RESULT/'baseline_summary.json'), read(EXP/'metadata.json')
    assert process['status'] == 'completed' and process['exit_code'] == 0
    assert summary['environment_steps'] == 6000000 and summary['curriculum_level'] == 2
    for rel, expected in metadata['source_hashes'].items():
        assert sha(ROOT/rel) == expected, ('Source changed', rel)
    original = read(REVIEW/'manifest.json')
    assert original['model_sha256'] == sha(REVIEW/'final.pth')
    assert original['config_sha256'] == sha(ROOT/'experiments/baseline_exp014.yaml')
    assert read(REVIEW/'process.json')['completed_levels'] == [1, 2, 3, 4, 5]
    assert read(MOTION/'process.json')['status'] == 'completed'
    for level in range(1, 6):
        m = read(MOTION/f'level_{level}/manifest.json')
        assert m['model_sha256'] == original['model_sha256']
        assert m['exact_unmodified_simulator_replay']
        for filename, key in [('motion.mp4', 'video_sha256'), ('trajectory.npz', 'trajectory_sha256')]:
            assert sha(MOTION/f'level_{level}'/filename) == m[key]
    seed_dirs = list((ROOT/'outputs/models/matd3/baseline_exp014').glob('seed_*'))
    assert len(seed_dirs) == 1
    assert sha(seed_dirs[0]/'final_train_model.pth') == original['model_sha256']
    initial = Path(metadata['initial_checkpoint'])
    assert sha(initial) == metadata['initial_checkpoint_sha256']
    active = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = [a.decode(errors='replace') for a in (proc/'cmdline').read_bytes().split(b'\0') if a]
            if any(Path(a).name in ('run_baseline.py', 'start_exp014.py', 'render_exp014_motion.py') for a in args):
                active.append(proc.name)
        except OSError:
            pass
    assert not active, ('Relevant process still active', active)
    print('VERIFIED original source, final model, initialization, evaluations, videos, stopped processes', flush=True)
    DEST.mkdir(parents=True)
    copied = {}

    def copy(source, relative):
        target = DEST/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        expected = sha(source)
        assert sha(target) == expected, str(source)
        copied[str(relative)] = dict(source=str(Path(source).relative_to(ROOT)), sha256=expected, bytes=target.stat().st_size)

    # Standalone source tree, including unchanged modules needed by the frozen training sources.
    excluded = {'__pycache__', '.pytest_cache', '.git', 'outputs', 'docs'}
    for name in ('xuance', 'examples', 'experiments', 'tests', 'tools', 'bash'):
        for directory, dirs, files in os.walk(ROOT/name):
            dirs[:] = [d for d in dirs if d not in excluded]
            for filename in sorted(files):
                source = Path(directory)/filename
                if source.is_symlink() or source.suffix in ('.pyc', '.pyo'):
                    continue
                copy(source, Path('code')/source.relative_to(ROOT))
    copy(ROOT/'.gitignore', 'code/.gitignore')
    for name in ('metadata.json', 'process.json', 'source_snapshot.zip', 'train.log',
                 'initialization_tensor_audit.json', 'supplemental_review.json'):
        if (EXP/name).exists():
            copy(EXP/name, Path('experiment')/name)
    for source in RESULT.iterdir():
        if source.is_file():
            copy(source, Path('results')/source.name)
    model_sources = {'final.pth': REVIEW/'final.pth', 'initial_007.pth': initial,
        'loaded_007_start.pth': seed_dirs[0]/'initial_random_model.pth',
        'best_level1.pth': RESULT/'best_model/best_stage_0.pth',
        'best_level2.pth': RESULT/'best_model/best_stage_1.pth',
        'best_level3.pth': RESULT/'best_model/best_stage_2.pth',
        'best_task5.pth': RESULT/'best_model/best_model.pth'}
    for name, source in model_sources.items():
        copy(source, Path('models')/name)
    for source in REVIEW.iterdir():
        if source.is_file() and source.suffix != '.pth':
            copy(source, Path('evaluation')/source.name)
    for source in MOTION.rglob('*'):
        if source.is_file():
            copy(source, Path('videos')/source.relative_to(MOTION))
    for name in ('baseline_exp014_changes_2026-10-14.md', 'baseline_exp014_results_2026-10-14.md',
                 'baseline_exp014_results_2026-10-14.json', 'pursuit_optimization_constraints_2026-10-14.md',
                 'baseline_exp008_complete_report_todo_updated_2026-10-14.md'):
        if (ROOT/'docs'/name).exists():
            copy(ROOT/'docs'/name, Path('reference_docs')/name)
    for source in (ROOT/'outputs/reviews/exp014_archive_notes_v1').glob('*'):
        if source.is_file():
            copy(source, Path('reference_docs')/source.name)
    checks = ROOT/'outputs/reviews/exp014_gpu_check_v1'
    for source in checks.rglob('*'):
        if source.is_file() and source.suffix in ('.json', '.log'):
            copy(source, Path('implementation_checks')/source.relative_to(checks))

    import torch
    start_state = torch.load(seed_dirs[0]/'initial_random_model.pth', map_location='cpu', weights_only=False)
    initial_state = torch.load(initial, map_location='cpu', weights_only=False)
    assert set(start_state['policy']) == set(initial_state['policy'])
    assert all(torch.equal(value, initial_state['policy'][key]) for key, value in start_state['policy'].items())
    assert all(not value['state'] for group in start_state['optimizer'].values() for value in group.values())
    final_state = torch.load(REVIEW/'final.pth', map_location='cpu', weights_only=False)
    optimizer_steps = {kind: sorted(set(int(value['step']) for value in state['state'].values()))
                       for kind, state in final_state['optimizer']['uav_0'].items()}
    checkpoint_audit = dict(keys=list(final_state), policy_tensors=len(final_state['policy']),
        optimizer_steps=optimizer_steps, initial_policy_exactly_matches_007=True,
        initial_optimizer_empty=True, initial_saved_after_007_load=True,
        original_initial_filename_misleading='initial_random_model.pth',
        optimizer_state_saved=True, torch_rng_state_saved=True, replay_buffer_saved=False,
        scheduler_state_saved=False, environment_state_saved=False)
    write(DEST/'checkpoint_audit.json', checkpoint_audit)
    runtime = dict(python=platform.python_version(), platform=platform.platform(),
        packages=sorted([dict(name=d.metadata.get('Name'), version=d.version) for d in importlib.metadata.distributions()], key=lambda x: x['name'] or ''),
        torch_version=str(torch.__version__), torch_cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        warning='Default editable xuance install points to the original checkout: always set PYTHONPATH to the intended frozen code directory.')
    write(DEST/'runtime.json', runtime)
    git_state = dict(branch=branch, head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
        captured_at=datetime.now(timezone.utc).isoformat(), committed_or_pushed=False)
    write(DEST/'git_state.json', git_state)
    (DEST/'git_diff.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT))

    # Full TensorBoard event records are a separate compressed archive; no tags are discarded.
    log_root = ROOT/'outputs/logs/matd3/baseline_exp014'
    logs = sorted(p for p in log_root.rglob('*') if p.is_file())
    assert logs
    log_records = []
    with zipfile.ZipFile(TB_ZIP, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source in logs:
            before = source.stat()
            relative = Path('logs')/source.relative_to(log_root)
            h, count = hashlib.sha256(), 0
            with source.open('rb') as stream, archive.open(str(relative), 'w', force_zip64=True) as target:
                for block in iter(lambda: stream.read(4*1024*1024), b''):
                    target.write(block)
                    h.update(block)
                    count += len(block)
                    if count % (256*1024*1024) == 0:
                        print('TENSORBOARD compressed input MB', count//1024//1024, flush=True)
            after = source.stat()
            assert before.st_size == count == after.st_size and before.st_mtime_ns == after.st_mtime_ns
            log_records.append(dict(source=str(source.relative_to(ROOT)), entry=str(relative), sha256=h.hexdigest(), bytes=count))
    print('TENSORBOARD ZIP', TB_ZIP.stat().st_size, flush=True)
    with zipfile.ZipFile(TB_ZIP) as archive:
        for item in log_records:
            h = hashlib.sha256()
            with archive.open(item['entry']) as stream:
                for block in iter(lambda: stream.read(4*1024*1024), b''):
                    h.update(block)
            assert h.hexdigest() == item['sha256']
    tb_record = dict(file=TB_ZIP.name, sha256=sha(TB_ZIP), bytes=TB_ZIP.stat().st_size, entries=log_records,
                     verified_full_event_records=True)
    write(DEST/'tensorboard_archive.json', tb_record)

    with (RESULT/'training_updates.csv').open() as stream:
        updates = list(csv.DictReader(stream))
    with (RESULT/'training_episodes.csv').open() as stream:
        episodes = list(csv.DictReader(stream))
    assert optimizer_steps['critic'] == [int(updates[-1]['updates'])]
    assert optimizer_steps['actor'] == [int(updates[-1]['actor_updates'])]
    metrics = []
    for level in range(5):
        selected = [r for r in episodes if int(r['level']) == level]
        metrics.append(dict(level=level+1, complete_episodes=len(selected),
            complete_episode_transfers=sum(int(r['episode_steps']) for r in selected),
            captures=sum(int(r['success']) for r in selected), safe_captures=sum(int(r['safe_success']) for r in selected),
            final_model_metrics=read(REVIEW/f'final_level{level+1}.json')['metrics']))
    manifest = dict(archive='exp014_baseline_v1', designation='user_selected_baseline',
        selected_on='2026-10-14', archived_at=datetime.now(timezone.utc).isoformat(),
        environment_steps=6000000, reached_training_level=3,
        critic_updates=int(updates[-1]['updates']), actor_updates=int(updates[-1]['actor_updates']),
        last_update_input_step=int(updates[-1]['step']), last_update_precedes_final_transition_batch=True,
        model_sha256=original['model_sha256'], config_sha256=original['config_sha256'],
        initial_model_sha256=metadata['initial_checkpoint_sha256'], initialization=metadata['initialization'],
        evaluation=original, level_results=metrics, stage_best_training_evaluations=summary['best_by_stage'],
        final_task_best_training_evaluation=summary['best_final_task'],
        tensorboard_archive=tb_record, original_source_hashes_verified=True,
        original_sources=metadata['source_hashes'], checkpoint_source_map={name: str(source.relative_to(ROOT)) for name, source in model_sources.items()},
        checkpoint_contents=checkpoint_audit,
        checkpoint_limit='Policy, target networks, optimizer and Torch RNG are saved; replay, scheduler, environment and runner state are absent, so full exact training-state resume is unavailable.',
        preserved_original_artifacts=True, new_training_started=False, files_copied=copied)
    write(DEST/'manifest.json', manifest)
    copy(ROOT/'outputs/reviews/exp014_archive_notes_v1/BASELINE.md', 'BASELINE.md')
    copied.pop('BASELINE.md', None)
    files = {str(p.relative_to(DEST)): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(DEST.rglob('*')) if p.is_file()}
    write(DEST/'file_hashes.json', files)
    verification = dict(status='verified', verified_at=datetime.now(timezone.utc).isoformat(),
        payload_files=len(files), payload_bytes=sum(v['bytes'] for v in files.values()),
        source_hashes_match_training_metadata=True, final_model_matches_training_output=True,
        all_five_evaluations_and_videos_present=True, full_tensorboard_verified=True,
        no_training_running=True, optimizer_state_saved=True, replay_buffer_saved=False,
        optimizer_steps_match_training_update_csv=True, initialized_model_tensors_match_007=True)
    write(DEST/'verification.json', verification)
    with zipfile.ZipFile(CORE_ZIP, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source in sorted(DEST.rglob('*')):
            if source.is_file():
                archive.write(source, Path(DEST.name)/source.relative_to(DEST))
    with zipfile.ZipFile(CORE_ZIP) as archive:
        for rel, expected in files.items():
            h = hashlib.sha256(archive.read(str(Path(DEST.name)/rel))).hexdigest()
            assert h == expected['sha256'], rel
    for archive in (CORE_ZIP, TB_ZIP):
        archive.with_suffix(archive.suffix+'.sha256').write_text(f'{sha(archive)}  {archive.name}\n')
    print('CORE ZIP', CORE_ZIP.stat().st_size, sha(CORE_ZIP), flush=True)
    print('TENSORBOARD ZIP SHA', tb_record['sha256'], flush=True)
    for source in DEST.rglob('*'):
        source.chmod(source.stat().st_mode & ~0o222)
    DEST.chmod(DEST.stat().st_mode & ~0o222)
    for source in (CORE_ZIP, TB_ZIP, CORE_ZIP.with_suffix('.zip.sha256'), TB_ZIP.with_suffix('.zip.sha256')):
        source.chmod(source.stat().st_mode & ~0o222)
    print('ARCHIVE_COMPLETE', json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
