"""Start TensorBoard at an experiment or seed root without selecting a metric folder."""
import argparse
from pathlib import Path
import subprocess
import sys


BASE_LOG_DIR = str(Path(__file__).resolve().parents[1] / "outputs/logs/matd3")


def get_latest_log_dir(base_dir):
    base = Path(base_dir).expanduser()
    if not base.is_dir():
        return None
    # A seed root already contains events; its children may be add_scalars series.
    if any(base.glob("events.out.tfevents.*")):
        return str(base)
    candidates = [path for path in base.iterdir()
                  if path.is_dir() and any(path.rglob("events.out.tfevents.*"))]
    return str(max(candidates, key=lambda path: path.stat().st_mtime)) if candidates else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=BASE_LOG_DIR,
                        help="Experiment, seed, or MATD3 log root")
    parser.add_argument("-n", "--name", default="", help="Explicit child directory of --base-dir")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()
    target = str(Path(args.base_dir) / args.name) if args.name else get_latest_log_dir(args.base_dir)
    if target is None or not Path(target).is_dir():
        parser.error("No TensorBoard log directory found")
    print(f"TensorBoard log root: {Path(target).resolve()}", flush=True)
    try:
        subprocess.run([sys.executable, "-m", "tensorboard.main", "--logdir", target,
                        "--host", args.host, "--port", str(args.port)], check=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
