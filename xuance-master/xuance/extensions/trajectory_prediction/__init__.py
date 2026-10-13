"""Native Xuance registry entries for offline civilian trajectory prediction."""
from importlib.util import find_spec
from pathlib import Path
import sys

if find_spec("trajectory_prediction") is None:
    source = Path(__file__).resolve().parents[3] / "Trajectory Prediction/src"
    if not source.is_dir():
        raise ImportError("Install the civilian-trajectory-prediction package before using this extension")
    sys.path.insert(0, str(source))

from xuance.engine import REGISTRY_Runner
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import REGISTRY_Learners
from trajectory_prediction.model import build_forecaster
from .learner import XuanceTrajectoryLearner
from .runner import RunnerTrajectoryPrediction


def register():
    REGISTRY_Policy["CivilianTrajectory"] = build_forecaster
    REGISTRY_Learners["CivilianTrajectoryLearner"] = XuanceTrajectoryLearner
    REGISTRY_Runner["TrajectoryPrediction"] = RunnerTrajectoryPrediction


register()
