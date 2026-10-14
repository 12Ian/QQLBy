"""One-time authenticated delivery of the authorized experiment's finished artifacts."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "outputs/experiments/exp009"


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def failed_package():
    environment = dict(os.environ, PYTHONPATH=str(ROOT))
    subprocess.run([sys.executable, "tools/report_exp009.py"], cwd=ROOT, env=environment,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    archive = ROOT / "docs/baseline_exp009_failed_delivery_v1.zip"
    if not archive.exists():
        paths = set()
        for directory in (EXPERIMENT, ROOT / "outputs/results/matd3/baseline_exp009",
                          ROOT / "outputs/reviews/exp009_final_v1"):
            paths.update(path for path in directory.rglob("*") if path.is_file())
        paths.update(path for path in (ROOT / "docs").glob("baseline_exp009*")
                     if path.is_file() and path.suffix != ".zip")
        paths.add(ROOT / "experiments/baseline_exp009.yaml")
        manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(paths)}
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as output:
            for name in manifest:
                output.write(ROOT / name, name)
            output.writestr("delivery_manifest.json", json.dumps(dict(files=manifest), indent=2))
    return archive


def main():
    while True:
        delivery = read(EXPERIMENT / "delivery.json")
        process = read(EXPERIMENT / "process.json")
        finalization = read(EXPERIMENT / "finalization.json")
        if delivery:
            archive = Path(delivery["archive"])
            status = "completed"
            break
        if process.get("status") == "failed" or finalization.get("status") == "failed":
            archive = failed_package()
            status = "failed_partial_results"
            break
        time.sleep(30)
    header = dict(status=status, filename=archive.name, bytes=archive.stat().st_size,
                  sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    print(json.dumps(header), flush=True)
    with archive.open("rb") as stream:
        while block := stream.read(3 * 16384):
            print(base64.b64encode(block).decode("ascii"), flush=True)


if __name__ == "__main__":
    main()
