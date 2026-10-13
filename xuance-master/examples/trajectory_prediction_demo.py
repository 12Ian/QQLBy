"""Run the independent generic forecasting CLI from the examples directory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Trajectory Prediction" / "src"))

from trajectory_prediction.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
