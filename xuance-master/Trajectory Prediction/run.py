"""Source-checkout entry point; does not import the incomplete Xuance tree."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trajectory_prediction.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
