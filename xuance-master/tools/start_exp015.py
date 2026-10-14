"""Freeze and run the safety baseline once, then evaluate and render it."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'outputs/experiments/exp015'
RESULT = ROOT / 'outputs/results/matd3/baseline_exp015_safety'
REVIEW = ROOT / 'outputs/reviews/exp015_final_v1'
MIN_FREE = 1024 ** 3


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def run_worker(command, directory, environment, log_path, state, phase):
    if shutil.disk_usage(ROOT).free < MIN_FREE:
        raise RuntimeError('Less than 1 GiB available; preserve historical files and stop')
    with log_path.open('x') as log:
        worker = subprocess.Popen(command, cwd=directory, env=environment,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status='running', phase=phase, child_pid=worker.pid,
                     phase_started_at=timestamp(), command=command)
        save(EXP / 'process.json', state)
        while True:
            try:
                code = worker.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                state['free_disk_bytes'] = shutil.disk_usage(ROOT).free
                state['last_health_check'] = timestamp()
                save(EXP / 'process.json', state)
                if state['free_disk_bytes'] < MIN_FREE:
                    os.killpg(worker.pid, signal.SIGTERM)
                    try:
                        worker.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(worker.pid, signal.SIGKILL)
                        worker.wait()
                    raise RuntimeError('Disk protection stopped this worker; no automatic restart')
    if code:
        raise RuntimeError(f'{phase} exited with code {code}; see {log_path}')
    return code


def main():
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    assert branch == 'feat-coverage-reaward'
    assert not EXP.exists() and not RESULT.exists() and not REVIEW.exists()
    assert not (ROOT / 'outputs/reviews/exp015_motion_v1').exists()
    assert shutil.disk_usage(ROOT).free > 2 * MIN_FREE
    verified = json.loads((ROOT / 'outputs/reviews/exp015_safety_implementation_v1/verification.json').read_text())
    assert verified['unit_tests_passed'] == 82 and verified['pipeline']['environment_transfers'] == 64
    assert all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == h
               for n, h in verified['source_hashes'].items()), 'Safety sources changed since validation'
    configuration = ROOT / 'experiments/baseline_exp015_safety.yaml'
    cfg = yaml.safe_load(configuration.read_text())
    assert cfg['running_steps'] == 6000000 and cfg['parallels'] == 16
    assert cfg['safety_filter_enabled'] and cfg['curriculum_start_level'] == 0
    model = Path(cfg['baseline_init_checkpoint'])
    assert hashlib.sha256(model.read_bytes()).hexdigest() == 'cfd17132d4ae95241f07315617bcf0d8f35593bf7dadc774acfcf7705801a215'
    EXP.mkdir(parents=True)
    state = dict(status='preparing', phase='snapshot', supervisor_pid=os.getpid(),
                 experiment='015', started_at=timestamp(), budget=6000000,
                 evaluation_seed=84001, video_seed=85001)
    save(EXP / 'process.json', state)
    try:
        source = EXP / 'code'
        source.mkdir()
        ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo', '.pytest_cache', '*.sh')
        for directory in ('xuance', 'examples', 'experiments', 'tools', 'tests'):
            shutil.copytree(ROOT / directory, source / directory, ignore=ignore)
        # Every executed import/config is frozen; output paths remain the conventional project paths.
        hashes = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(source.rglob('*')) if p.is_file()}
        assert all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == h for n, h in hashes.items()), 'Source changed during freeze'
        with zipfile.ZipFile(EXP / 'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for name in hashes:
                archive.write(source / name, name)
        (source / 'outputs').symlink_to(ROOT / 'outputs', target_is_directory=True)
        frozen_config = source / 'experiments/baseline_exp015_safety.yaml'
        command = [sys.executable, '-u', str(source / 'examples/run_baseline.py'),
            '--name', 'exp015_safety', '--config', str(frozen_config), '--steps', '6000000',
            '--parallels', '16', '--seed', '1', '--device', 'cuda:0', '--start-level', '0',
            '--eval-interval', '50000', '--test-episodes', '100']
        metadata = dict(experiment='015', branch=branch,
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            source_hashes=hashes, frozen_source_root=str(source), command=command, budget=6000000,
            training_from_scratch=False, initialization='014 full policy; fresh optimizer and replay',
            initial_checkpoint=str(model), initial_checkpoint_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            config_sha256=hashes['experiments/baseline_exp015_safety.yaml'],
            free_disk_bytes_at_start=shutil.disk_usage(ROOT).free,
            safety_filter_enabled=True, evaluation_seed=84001, episodes_per_level=100,
            evaluation_modes=['filtered', 'raw_network'], video_seed=85001,
            known_video_comparison_seed=72001, automatic_restart=False)
        save(EXP / 'metadata.json', metadata)
        (EXP / 'git_diff.patch').write_bytes(subprocess.check_output(['git', 'diff'], cwd=ROOT))
        (EXP / 'git_status.txt').write_text(subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True))
        environment = dict(os.environ, PYTHONPATH=str(source), OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TQDM_MININTERVAL='30')
        run_worker(command, source, environment, EXP / 'train.log', state, 'training')
        state['training_completed_at'] = timestamp()
        save(EXP / 'process.json', state)
        summary = json.loads((RESULT / 'baseline_summary.json').read_text())
        assert summary['environment_steps'] == 6000000
        assert all(hashlib.sha256((source / n).read_bytes()).hexdigest() == h for n, h in hashes.items())
        init = json.loads((RESULT / 'initialization_audit.json').read_text())
        assert init['sha256'] == metadata['initial_checkpoint_sha256'] and not init['optimizer_restored']
        folders = list((ROOT / 'outputs/models/matd3/baseline_exp015_safety').glob('seed_*'))
        assert len(folders) == 1
        REVIEW.mkdir(parents=True)
        final = REVIEW / 'final.pth'
        shutil.copy2(folders[0] / 'final_train_model.pth', final)
        shutil.copy2(folders[0] / 'initial_random_model.pth', REVIEW / 'loaded_014_start.pth')
        save(REVIEW / 'manifest.json', dict(model_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
            config_sha256=metadata['config_sha256'], source_hashes_verified=True,
            seed=84001, episodes_per_level=100, max_steps=800,
            collision_termination=True, modes=['filtered', 'raw_network']))
        review_state = dict(status='evaluating', completed_levels=[], raw_completed_levels=[])
        save(REVIEW / 'process.json', review_state)
        for mode in ('filtered', 'raw_network'):
            for level in range(5):
                output = REVIEW / (f'final_level{level+1}.json' if mode == 'filtered' else f'raw_level{level+1}.json')
                evaluation = [sys.executable, '-u', str(source / 'examples/evaluate_safety_checkpoint.py'),
                    '--config', str(frozen_config), '--checkpoint', str(final), '--env-seed', '84001',
                    '--episodes', '100', '--parallels', '8', '--level', str(level),
                    '--device', 'cuda:0', '--output', str(output)]
                if mode == 'raw_network':
                    evaluation.append('--disable-filter')
                run_worker(evaluation, source, environment, REVIEW / f'{mode}_level{level+1}.log',
                           state, f'evaluation_{mode}_level{level+1}')
                key = 'completed_levels' if mode == 'filtered' else 'raw_completed_levels'
                review_state[key].append(level+1)
                save(REVIEW / 'process.json', review_state)
        review_state.update(status='completed', finished_at=timestamp())
        save(REVIEW / 'process.json', review_state)
        run_worker([sys.executable, '-u', str(source / 'tools/render_exp015_motion.py')],
                   source, environment, EXP / 'videos.log', state, 'videos')
        state.update(status='completed', phase='complete', finished_at=timestamp())
        save(EXP / 'process.json', state)
    except BaseException as error:
        state.update(status='failed', error=repr(error), finished_at=timestamp())
        save(EXP / 'process.json', state)
        raise


if __name__ == '__main__':
    main()
