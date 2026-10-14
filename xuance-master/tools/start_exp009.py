"""Start the authorized exp009 once with a source snapshot and real exit status."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "outputs/experiments/exp009"
COMMAND = [sys.executable, "-u", "examples/run_baseline.py", "--name", "exp009",
           "--config", "experiments/baseline_exp009.yaml", "--steps", "3000000",
           "--parallels", "16", "--seed", "1", "--device", "cuda:0",
           "--start-level", "0", "--eval-interval", "50000", "--test-episodes", "100"]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    assert branch == "feat-coverage-reaward", branch
    assert not (ROOT / "outputs/results/matd3/baseline_exp009").exists()
    audit = json.loads((ROOT / "outputs/reviews/exp009_gpu_smoke_v1/audit.json").read_text())
    assert audit["status"] == "passed" and audit["environment_steps"] == 2048
    tests = Path("/tmp/uav_exp009_tests_final.log").read_text()
    assert "63 passed" in tests and "failed" not in tests
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(ROOT / "outputs/experiments/exp008/source_snapshot.zip") as previous:
        names = set(previous.namelist())
    names.update(["experiments/baseline_exp009.yaml", "tests/test_uav_baseline009.py", "tools/check_exp009_gpu.py",
                  "tools/start_exp009.py", "tools/finish_exp009.py", "tools/report_exp009.py"])
    hashes = {}
    with zipfile.ZipFile(DIRECTORY / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            path = ROOT / name
            if path.is_file():
                data = path.read_bytes()
                archive.writestr(name, data)
                hashes[name] = hashlib.sha256(data).hexdigest()
    metadata = dict(experiment="009", branch=branch,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        command=COMMAND, source_hashes=hashes, initial_checkpoint=None, training_from_scratch=True,
        changes=["hold-only capture; collisions do not invalidate the capture criterion",
                 "evader sense100m, acceleration4, maxima3/5/7/8/12",
                 "physics0.1/decision0.5, episode800, gamma=sqrt(0.99), braking reaction1.5",
                 "progress replaced by shared closing-speed and nearest-teammate-angle rewards",
                 "periodic previous-level retention evaluation"],
        validation=dict(tests="63 passed in 12.79s", gpu_smoke=audit),
        final_review=dict(seed=64001, episodes_per_level=100, parallels=8, max_steps=800),
        live_monitor_automation="user requested paused")
    save(DIRECTORY / "metadata.json", metadata)
    shutil_log = DIRECTORY / "test.log"
    shutil_log.write_text(tests, encoding="utf-8")
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       TQDM_MININTERVAL="30")
    with (DIRECTORY / "train.log").open("x") as log:
        worker = subprocess.Popen(COMMAND, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        state = dict(experiment="009", supervisor_pid=os.getpid(), child_pid=worker.pid,
                     command=COMMAND, started_at=now(), status="running")
        save(DIRECTORY / "process.json", state)
        code = worker.wait()
    state.update(status="completed" if code == 0 else "failed", exit_code=code, finished_at=now())
    save(DIRECTORY / "process.json", state)
    subprocess.run([sys.executable, "tools/report_exp009.py"], cwd=ROOT, env=environment)
    if code == 0:
        final_code = subprocess.run([sys.executable, "tools/finish_exp009.py"], cwd=ROOT, env=environment).returncode
        if final_code:
            current = DIRECTORY / "finalization.json"
            if not current.exists():
                save(current, dict(status="failed", exit_code=final_code,
                                   error="Finalization failed before state creation; see supervisor.log"))
            subprocess.run([sys.executable, "tools/report_exp009.py"], cwd=ROOT, env=environment)


if __name__ == "__main__":
    main()
