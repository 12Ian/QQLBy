"""Launch the approved far-side MATD3 experiment once, with provenance and exit status."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "outputs/experiments/exp008"
COMMAND = [sys.executable, "-u", "examples/run_baseline.py", "--name", "exp008",
           "--config", "experiments/baseline_exp008.yaml", "--steps", "3000000",
           "--parallels", "16", "--seed", "1", "--device", "cuda:0",
           "--start-level", "0", "--eval-interval", "50000", "--test-episodes", "100"]


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    assert branch == "feat-coverage-reaward", branch
    assert not (ROOT / "outputs/results/matd3/baseline_exp008").exists()
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    previous = ROOT / "outputs/experiments/exp007/source_snapshot.zip"
    with zipfile.ZipFile(previous) as archive:
        names = set(archive.namelist())
    names.update(["experiments/baseline_exp008.yaml", "examples/run_tensorboard.py",
                  "xuance/environment/multi_agent_env/uav_pursuit_coverage_3d.py",
                  "xuance/engine/uav_baseline_logging.py", "xuance/torch/agents/base/agents_marl.py",
                  "tests/test_uav_baseline.py", "tools/start_exp008.py", "tools/report_exp008.py"])
    hashes = {}
    with zipfile.ZipFile(DIRECTORY / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            path = ROOT / name
            if path.is_file():
                data = path.read_bytes()
                archive.writestr(name, data)
                hashes[name] = hashlib.sha256(data).hexdigest()
    checkpoint = ROOT / "outputs/acceptance/exp007_final_acceptance_v1/model.pth"
    metadata = dict(experiment="008", branch=branch,
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    command=COMMAND, source_hashes=hashes, initial_checkpoint=str(checkpoint),
                    initial_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    changes=["distance reward disabled; potential progress retained", "teammate penalties and collisions disabled",
                             "flat TensorBoard scalars and detailed episode diagnostics", "west-side pursuers, distant eastern target"],
                    training_from_scratch=False, curriculum_start_display_level=1,
                    tests="47 passed in 12.83s", gpu_smoke_steps=2048,
                    gpu_smoke_critic_updates=1008, gpu_smoke_actor_updates=504,
                    gpu_smoke_tensorboard_root_tags=338, gpu_smoke_all_scalars_finite=True)
    (DIRECTORY / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", TQDM_MININTERVAL="30")
    with (DIRECTORY / "train.log").open("x") as log:
        worker = subprocess.Popen(COMMAND, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        state = dict(experiment="008", supervisor_pid=os.getpid(), child_pid=worker.pid,
                     command=COMMAND, started_at=now(), status="running")
        (DIRECTORY / "process.json").write_text(json.dumps(state, indent=2) + "\n")
        code = worker.wait()
    state.update(status="completed" if code == 0 else "failed", exit_code=code, finished_at=now())
    temporary = DIRECTORY / "process.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(DIRECTORY / "process.json")
    subprocess.run([sys.executable, "tools/report_exp008.py"], cwd=ROOT, env=environment)


if __name__ == "__main__":
    main()
