"""Freeze a finished exp009, evaluate once, render real successes and report."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "outputs/experiments/exp009"
RESULT = ROOT / "outputs/results/matd3/baseline_exp009"
REVIEW = ROOT / "outputs/reviews/exp009_final_v1"


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    process = json.loads((EXPERIMENT / "process.json").read_text())
    assert process["status"] == "completed" and process["exit_code"] == 0
    summary = json.loads((RESULT / "baseline_summary.json").read_text())
    assert summary["environment_steps"] == 3000000
    metadata = json.loads((EXPERIMENT / "metadata.json").read_text())
    mismatches = [name for name, expected in metadata["source_hashes"].items() if sha(ROOT / name) != expected]
    if mismatches:
        raise RuntimeError("Source changed since launch: " + repr(mismatches))
    REVIEW.mkdir(parents=True, exist_ok=False)
    state = dict(status="running", started_at=now(), pid=os.getpid(), evaluations=[], videos=[])
    save(EXPERIMENT / "finalization.json", state)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        models = list((ROOT / "outputs/models/matd3/baseline_exp009").glob("seed_*/final_train_model.pth"))
        assert len(models) == 1, models
        model = REVIEW / "final009.pth"
        shutil.copy2(models[0], model)
        config = REVIEW / "config.yaml"
        shutil.copy2(ROOT / "experiments/baseline_exp009.yaml", config)
        shutil.copy2(RESULT / "config.json", REVIEW / "effective_config.json")
        manifest = dict(experiment="009", checkpoint_source=str(models[0]), model_sha256=sha(model),
                        config_sha256=sha(config), source_hash_count=len(metadata["source_hashes"]),
                        source_hash_mismatches=[], training_from_scratch=True, evaluation_seed=64001,
                        episodes_per_level=100, parallels=8, exploration_noise=False,
                        collision_termination=False, max_episode_steps=800)
        save(REVIEW / "manifest.json", manifest)
        for level in range(5):
            output = REVIEW / f"final009_level{level+1}.json"
            command = [sys.executable, "-u", "examples/evaluate_baseline_checkpoint.py", "--config", str(config),
                       "--checkpoint", str(model), "--env-seed", "64001", "--episodes", "100", "--parallels", "8",
                       "--level", str(level), "--device", "cuda:0", "--output", str(output)]
            with (REVIEW / f"level{level+1}.log").open("x") as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            state["evaluations"].append(dict(level=level+1, output=str(output), status="completed"))
            save(EXPERIMENT / "finalization.json", state)
            subprocess.run([sys.executable, "tools/report_exp009.py"], cwd=ROOT, env=environment, check=True)
        for level_text, metrics in summary.get("best_by_stage", {}).items():
            level = int(level_text)
            if float(metrics["safe_success_rate"]) <= 0.:
                state["videos"].append(dict(level=level+1, status="no_safe_example_in_formal_evaluation"))
                continue
            output = ROOT / f"outputs/reviews/exp009_stage{level+1}_video_v1"
            command = [sys.executable, "-u", "tools/render_baseline_success_videos.py", "--root", str(ROOT),
                       "--output", str(output), "--stages", f"009:{level}", "--env-seed", "65001", "--max-attempts", "20"]
            with (REVIEW / f"video_level{level+1}.log").open("x") as log:
                code = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
            state["videos"].append(dict(level=level+1, status="completed" if code == 0 else "selection_or_render_failed",
                                        exit_code=code, directory=str(output)))
            save(EXPERIMENT / "finalization.json", state)
        state.update(status="completed", finished_at=now(), model_sha256=sha(model))
        save(EXPERIMENT / "finalization.json", state)
        subprocess.run([sys.executable, "tools/report_exp009.py"], cwd=ROOT, env=environment, check=True)
        files = set()
        for directory in (EXPERIMENT, RESULT, REVIEW):
            files.update(path for path in directory.rglob("*") if path.is_file())
        for directory in (ROOT / "outputs/reviews").glob("exp009_stage*_video_v1"):
            files.update(path for path in directory.rglob("*") if path.is_file())
        files.update((ROOT / "docs").glob("baseline_exp009*"))
        files.add(ROOT / "experiments/baseline_exp009.yaml")
        package = ROOT / "docs/baseline_exp009_delivery_v1.zip"
        files.discard(package)
        hashes = {str(path.relative_to(ROOT)): sha(path) for path in sorted(files) if path.is_file()}
        delivery = dict(archive=str(package), files=hashes)
        with zipfile.ZipFile(package, "x", zipfile.ZIP_DEFLATED) as archive:
            for name in hashes:
                archive.write(ROOT / name, name)
            archive.writestr("delivery_manifest.json", json.dumps(delivery, ensure_ascii=False, indent=2))
        delivery.update(archive_sha256=sha(package), archive_bytes=package.stat().st_size)
        save(EXPERIMENT / "delivery.json", delivery)
    except BaseException as error:
        state.update(status="failed", finished_at=now(), error=repr(error))
        save(EXPERIMENT / "finalization.json", state)
        subprocess.run([sys.executable, "tools/report_exp009.py"], cwd=ROOT, env=environment)
        raise


if __name__ == "__main__":
    main()
