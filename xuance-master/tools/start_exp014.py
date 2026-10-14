"""Start the 6M-transition run with verified 007 initialization exactly once."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/experiments/exp014'


def save(state):
    temporary = OUT / 'process.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(OUT / 'process.json')


def main():
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    assert branch == 'feat-coverage-reaward', branch
    configuration = ROOT / 'experiments/baseline_exp014.yaml'
    cfg = yaml.safe_load(configuration.read_text())
    assert cfg['running_steps'] == 6000000 and cfg['curriculum_start_level'] == 0
    assert cfg['evader_sense'] == 200 and cfg['baseline_formation_weight'] == 0
    assert not (ROOT / 'outputs/results/matd3/baseline_exp014').exists()
    check = ROOT / 'outputs/reviews/exp014_gpu_check_v1/audit.json'
    assert json.loads(check.read_text())['status'] == 'passed'
    checkpoint = Path(cfg['baseline_init_checkpoint'])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == '6c1f41773b47238f898d0f4243b092b8506d2f1ce3bba70e47cac3df0a7d5a32'
    # Keep 013's frozen sources unchanged, and include the new run configuration.
    names = set(json.loads((ROOT / 'outputs/experiments/exp013/metadata.json').read_text())['source_hashes'])
    names.update(['experiments/baseline_exp014.yaml', 'tools/start_exp014.py', 'tools/check_exp014.py'])
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)}
    command = [sys.executable, '-u', 'examples/run_baseline.py', '--name', 'exp014',
        '--config', str(configuration), '--steps', str(cfg['running_steps']),
        '--parallels', str(cfg['parallels']), '--seed', str(cfg['seed']), '--device', 'cuda:0',
        '--start-level', str(cfg['curriculum_start_level']), '--eval-interval', str(cfg['eval_interval']),
        '--test-episodes', str(cfg['test_episode'])]
    OUT.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(OUT / 'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(ROOT / name, name)
    metadata = dict(experiment='014', branch=branch, source_hashes=hashes, command=command,
        budget=cfg['running_steps'], training_from_scratch=False, initial_checkpoint=str(checkpoint),
        initial_checkpoint_sha256=digest, initialization='007 full policy; fresh optimizer and replay',
        curriculum_start_display_level=1, config_sha256=hashes['experiments/baseline_exp014.yaml'],
        free_disk_bytes_at_start=shutil.disk_usage(ROOT).free)
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    environment = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TQDM_MININTERVAL='30')
    state = dict(status='starting', supervisor_pid=os.getpid(), command=command)
    save(state)
    try:
        with (OUT / 'train.log').open('x') as log:
            worker = subprocess.Popen(command, cwd=ROOT, env=environment,
                                      stdout=log, stderr=subprocess.STDOUT)
            state.update(status='running', child_pid=worker.pid,
                         started_at=datetime.now(timezone.utc).isoformat())
            save(state)
            code = worker.wait()
        state.update(status='completed' if code == 0 else 'failed', exit_code=code,
                     finished_at=datetime.now(timezone.utc).isoformat())
        save(state)
    except BaseException as error:
        state.update(status='failed', error=repr(error),
                     finished_at=datetime.now(timezone.utc).isoformat())
        save(state)
        raise


if __name__ == '__main__':
    main()
