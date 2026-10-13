"""Export the verified Xuance sources and civilian extension, without environments or logs."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="New .tar.gz filename")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    upstream = json.loads((root / "UPSTREAM_XUANCE.json").read_text(encoding="utf-8"))
    paths = {root / record["path"] for record in upstream["files"]}
    module = root / "Trajectory Prediction"
    # Sources only: never ship datasets, checkpoints or run logs (the server
    # regenerates the dataset from run_engagement.sh anyway).
    excluded = {"__pycache__", ".pytest_cache", "build", "dist", "outputs", ".workbuddy"}
    paths.update(path for path in module.rglob("*") if path.is_file()
                 and not excluded.intersection(path.parts)
                 and not any(part.endswith(".egg-info") for part in path.parts))
    paths.update(path for path in (root / "xuance/extensions").rglob("*.py") if "__pycache__" not in path.parts)
    paths.update((root / "xuance/configs/trajectory_prediction").rglob("*.yaml"))
    paths.update(root / name for name in ("UPSTREAM_XUANCE.json", "LICENSE.xuance", "LOCAL_XUANCE_CHANGES.md",
                 "bash/train_trajectory_prediction.sh", "bash/run_trajectory_study.sh", "bash/run_civilian_interaction.sh",
                 # military intercept-engagement entry points (the paper port)
                 "bash/run_engagement.sh", "bash/run_paper_targets.sh", "bash/setup_server.sh",
                 "examples/trajectory_prediction_demo.py"))
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("A required source file is missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with tarfile.open(output, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(paths):
            name = "xuance-master/" + path.relative_to(root).as_posix()
            archive.add(path, arcname=name, recursive=False)
            records.append(dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report = dict(archive=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(), files=records,
                  data_included=False, dependencies_included=False)
    manifest = output.with_name(output.name + ".manifest.json")
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(archive=str(output), sha256=report["sha256"], files=len(records)), indent=2))


if __name__ == "__main__":
    main()
