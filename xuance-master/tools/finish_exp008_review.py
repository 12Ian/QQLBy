"""Freeze the completed exp008 run, evaluate it, and render unrepeated stage videos."""
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
PROCESS = ROOT / "outputs/experiments/exp008/final_review_process.json"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(value):
    PROCESS.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    original = json.loads((ROOT / "outputs/experiments/exp008/process.json").read_text())
    assert original["status"] == "completed" and original["exit_code"] == 0
    metadata = json.loads((ROOT / "outputs/experiments/exp008/metadata.json").read_text())
    mismatch = [p for p, h in metadata["source_hashes"].items() if sha(ROOT / p) != h]
    assert not mismatch, mismatch
    REVIEW.mkdir(exist_ok=False)
    config = REVIEW / "frozen_config.yaml"
    shutil.copy2(ROOT / "experiments/baseline_exp008.yaml", config)
    models = list((ROOT / "outputs/models/matd3/baseline_exp008").glob("seed_*/final_train_model.pth"))
    assert len(models) == 1
    final = REVIEW / "final008.pth"
    initial = REVIEW / "initial007.pth"
    shutil.copy2(models[0], final)
    shutil.copy2(metadata["initial_checkpoint"], initial)
    assert sha(initial) == metadata["initial_checkpoint_sha256"]
    manifest = dict(frozen_at=stamp(), source_hash_mismatches=mismatch,
                    source_hash_count=len(metadata["source_hashes"]), metadata=metadata,
                    final_source=str(models[0]), final_sha256=sha(final),
                    initial_source=metadata["initial_checkpoint"], initial_sha256=sha(initial),
                    config_sha256=sha(config), env_seed=60001, episodes_per_level=100,
                    evaluation_parallels=8, exploration_noise=False, decision_dt=1.0,
                    physics_dt=0.1, maximum_decisions=400, all_episode_collision_statistics=True)
    (REVIEW / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")
    state = dict(status="running", supervisor_pid=os.getpid(), started_at=stamp(), jobs=[])
    video_path = ROOT / "outputs/reviews/exp008_stages2to5_video_v1"
    video_log = (REVIEW / "stage_videos.log").open("w")
    video_command = [sys.executable, "-u", "tools/render_baseline_success_videos.py",
                     "--root", str(ROOT), "--stages", "008:1,008:2,008:3,008:4",
                     "--env-seed", "63001", "--max-attempts", "20", "--output", str(video_path)]
    video = subprocess.Popen(video_command, cwd=ROOT, env=env, stdout=video_log,
                             stderr=subprocess.STDOUT)
    state["video"] = dict(pid=video.pid, command=video_command, output=str(video_path), status="running")
    save(state)
    try:
        for label, checkpoint in [("final008", final), ("initial007", initial)]:
            for level in range(5):
                output = REVIEW / f"{label}_level{level + 1}.json"
                command = [sys.executable, "-u", "examples/evaluate_baseline_checkpoint.py",
                           "--config", str(config), "--checkpoint", str(checkpoint),
                           "--env-seed", "60001", "--episodes", "100", "--parallels", "8",
                           "--level", str(level), "--device", "cuda:0", "--output", str(output)]
                job = dict(model=label, level=level + 1, command=command, started_at=stamp(), status="running")
                state["jobs"].append(job)
                with (REVIEW / f"{label}_level{level + 1}.log").open("w") as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    job["pid"] = child.pid
                    save(state)
                    code = child.wait()
                job.update(exit_code=code, finished_at=stamp(), status="completed" if code == 0 else "failed")
                save(state)
                if code != 0:
                    raise RuntimeError(f"Evaluation {label} level {level + 1} failed with {code}")
                report = json.loads(output.read_text())
                assert report["checkpoint_sha256"] == sha(checkpoint)
                assert report["metrics"]["episodes"] == 100
                print("EVALUATION_COMPLETE", label, level + 1, report["metrics"], flush=True)
        video_code = video.wait()
        state["video"].update(exit_code=video_code, status="completed" if video_code == 0 else "failed")
        if video_code:
            raise RuntimeError(f"Video rendering failed with {video_code}; inspect stage_videos.log")
        state.update(status="completed", exit_code=0, finished_at=stamp())
        save(state)
        print("FINAL_REVIEW_COMPLETE", flush=True)
    except Exception as error:
        state.update(status="failed", exit_code=1, finished_at=stamp(), error=str(error))
        state["video"]["status"] = "running" if video.poll() is None else "finished"
        if video.poll() is not None:
            state["video"]["exit_code"] = video.returncode
        save(state)
        raise
    finally:
        video_log.close()


if __name__ == "__main__":
    main()
