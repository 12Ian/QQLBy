"""Compare frozen earlier stage policies with the final policy on the same seed."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "outputs/reviews/exp008_final_v1"
STATE = REVIEW / "retention_process.json"


def main():
    folder = REVIEW / "retention_checkpoints"
    folder.mkdir(exist_ok=False)
    state = dict(status="running", pid=os.getpid(), jobs=[])
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for level in range(3):
        source = ROOT / f"outputs/results/matd3/baseline_exp008/best_model/best_stage_{level}.pth"
        checkpoint = folder / f"best_stage_{level}.pth"
        shutil.copy2(source, checkpoint)
        output = REVIEW / f"beststage_level{level + 1}.json"
        job = dict(level=level + 1, source=str(source), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   started_at=datetime.now(timezone.utc).isoformat(), status="running")
        state["jobs"].append(job)
        STATE.write_text(json.dumps(state, indent=2) + "\n")
        command = [sys.executable, "-u", "examples/evaluate_baseline_checkpoint.py", "--config", str(REVIEW / "frozen_config.yaml"),
                   "--checkpoint", str(checkpoint), "--env-seed", "60001", "--episodes", "100", "--parallels", "8",
                   "--level", str(level), "--device", "cpu", "--output", str(output)]
        with (REVIEW / f"beststage_level{level + 1}.log").open("w") as log:
            code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        job.update(exit_code=code, status="completed" if code == 0 else "failed")
        if code:
            state.update(status="failed", exit_code=code)
            STATE.write_text(json.dumps(state, indent=2) + "\n")
            raise RuntimeError(f"Retention comparison failed at level {level + 1}")
        assert json.loads(output.read_text())["checkpoint_sha256"] == job["checkpoint_sha256"]
    state.update(status="completed", exit_code=0)
    STATE.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
