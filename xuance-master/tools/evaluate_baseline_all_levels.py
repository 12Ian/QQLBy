"""Wait for exp007, freeze its two current models, and evaluate every course."""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "outputs/reviews/exp007_all_levels_v1"
PYTHON = "/home/ryy/miniconda3/envs/rl_env/bin/python"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def save_state(state):
    temporary = DIRECTORY / "process.tmp"
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(DIRECTORY / "process.json")


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    state = dict(status="waiting_training", supervisor_pid=os.getpid(), started_at=now(), jobs=[])
    save_state(state)
    try:
        process_path = ROOT / "outputs/experiments/exp007/process.json"
        while True:
            parent = read_json(process_path)
            cmdline = Path(f"/proc/{parent['child_pid']}/cmdline")
            live = cmdline.exists() and b"run_baseline.py" in cmdline.read_bytes()
            if not live and parent["status"] in ("completed", "failed"):
                assert parent["status"] == "completed" and parent["exit_code"] == 0, parent
                break
            state.update(parent_pid=parent["child_pid"], parent_live=live, checked_at=now())
            save_state(state)
            time.sleep(15)
        result = ROOT / "outputs/results/matd3/baseline_exp007"
        summary = read_json(result / "baseline_summary.json")
        assert summary["environment_steps"] == 3000000 and summary["curriculum_level"] == 4
        final_paths = list((ROOT / "outputs/models/matd3/baseline_exp007").glob("seed_*/final_train_model.pth"))
        assert len(final_paths) == 1
        sources = {"final": final_paths[0], "best": result / "best_model/best_stage_4.pth"}
        metadata = read_json(ROOT / "outputs/experiments/exp007/metadata.json")
        config_path = ROOT / "experiments/baseline_exp007.yaml"
        assert hashlib.sha256(config_path.read_bytes()).hexdigest() == metadata["source_hashes"]["experiments/baseline_exp007.yaml"]
        shutil.copyfile(config_path, DIRECTORY / "config.yaml")
        with (result / "curriculum_eval.csv").open(encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row["phase"] == "current" and row["level"] == "4"]
        best = max(rows, key=lambda row: (float(row["safe_success_rate"]), -float(row["collision_rate"]), float(row["mean_return"])))
        models = {}
        import torch
        for role, source in sources.items():
            destination = DIRECTORY / f"{role}.pth"
            data = source.read_bytes()
            policy = torch.load(source, map_location="cpu", weights_only=False)["policy"]
            assert len(policy) == 36 and all(torch.isfinite(value).all().item() for value in policy.values())
            destination.write_bytes(data)
            models[role] = dict(source_checkpoint=str(source), frozen_checkpoint=str(destination),
                                sha256=hashlib.sha256(data).hexdigest(), finite_policy_tensors=36,
                                training_step=3000000 if role == "final" else int(best["step"]))
        provenance = dict(experiment="007", frozen_at=now(), models=models, best_checkpoint_eval=best,
                          parent_process=parent, eval_seed=10001, episodes_per_course=100,
                          episode_length=400, collision_termination=False,
                          scope="two frozen policies evaluated on every original curriculum course")
        (DIRECTORY / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        state.update(status="evaluating", parent_live=False)
        save_state(state)
        for role in ("final", "best"):
            for level in range(5):
                output = DIRECTORY / f"{role}_level{level + 1}.json"
                command = [PYTHON, "-u", "examples/evaluate_baseline_checkpoint.py", "--config", str(DIRECTORY / "config.yaml"),
                           "--checkpoint", str(DIRECTORY / f"{role}.pth"), "--env-seed", "10001", "--episodes", "100",
                           "--parallels", "8", "--level", str(level), "--device", "cuda:0", "--output", str(output)]
                with (DIRECTORY / f"{role}_level{level + 1}.log").open("w") as log:
                    worker = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    job = dict(role=role, level=level, pid=worker.pid, command=command, status="running")
                    state["jobs"].append(job)
                    save_state(state)
                    code = worker.wait()
                    job.update(status="completed" if code == 0 else "failed", exit_code=code)
                    save_state(state)
                    assert code == 0, job
        state.update(status="completed", finished_at=now())
        save_state(state)
        status = dict(time=now(), process=parent, live=False, training=read_json(result / "training_status.json"))
        (ROOT / "docs/baseline_exp007_status_2026-10-13.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        commands = [
            [PYTHON, "tools/write_baseline_history_report.py", "--extra-videos", "outputs/videos/baseline_level_5_exp006_20261013_v1",
             "--replace-videos", "outputs/videos/baseline_level_4_graduation_exp006_20261013_v1",
             "--replace-videos", "outputs/videos/baseline_level_5_exp007_candidate_20261013_v1"],
            [PYTHON, "tools/write_baseline_final_report.py"],
        ]
        for command in commands:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
        state.update(report_status="completed", report_finished_at=now())
        save_state(state)
        print("ALL_LEVELS_REVIEW_AND_DOCUMENT_COMPLETE", flush=True)
    except Exception as error:
        state.update(status="failed" if state["status"] != "completed" else "completed",
                     report_status="failed", error=repr(error), checked_at=now())
        save_state(state)
        raise


if __name__ == "__main__":
    main()
