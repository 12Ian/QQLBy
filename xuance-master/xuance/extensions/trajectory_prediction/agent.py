from pathlib import Path
import shutil

from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.learners import REGISTRY_Learners
from trajectory_prediction.config import Config
from trajectory_prediction.engine import train, evaluate
from trajectory_prediction.inference import Predictor


class TrajectoryPredictionAgent:
    """Offline supervised Agent interface, without online actions or rewards."""
    def __init__(self, config):
        self.config = config
        values = dict(getattr(config, "trajectory", {}))
        profile = getattr(config, "trajectory_config", None)
        if profile:
            values.update(Config.read(profile).to_dict())
        overrides = getattr(config, "trajectory_overrides", {})
        unknown = set(overrides) - set(Config().to_dict())
        if unknown:
            raise ValueError(f"Unknown trajectory overrides: {sorted(unknown)}")
        values.update(overrides)
        values["device"] = str(config.device)
        self.cfg = Config(**values).validate()
        self.data = getattr(config, "data_path", None)
        self.output = Path(getattr(config, "output_dir", "outputs/trajectory/native"))
        self.closed = False
        self.result = None

    def train(self, **kwargs):
        self.result = train(self.data, self.output, self.cfg,
                            resume=kwargs.get("resume", getattr(self.config, "resume", None)),
                            intention_checkpoint=kwargs.get("intention_checkpoint", getattr(self.config, "intention_checkpoint", None)),
                            policy_factory=REGISTRY_Policy["CivilianTrajectory"],
                            learner_factory=REGISTRY_Learners["CivilianTrajectoryLearner"],
                            integration={"framework": "xuance", "runner": "TrajectoryPrediction",
                                         "policy": "CivilianTrajectory", "learner": "CivilianTrajectoryLearner",
                                         "learning": "offline supervised"})
        return self.result

    def test(self, checkpoint=None, split="test", output_path=None):
        checkpoint = checkpoint or getattr(self.config, "checkpoint", None) or self.output / "best.pt"
        return evaluate(self.data, checkpoint, split, self.cfg.device, output_path,
                        policy_factory=REGISTRY_Policy["CivilianTrajectory"],
                        learner_factory=REGISTRY_Learners["CivilianTrajectoryLearner"])

    def load_model(self, checkpoint=None):
        self.predictor = Predictor(checkpoint or self.output / "best.pt", self.cfg.device)
        self.policy = self.predictor.model
        return self.policy

    def predict(self, history, context=None, context_mask=None):
        if not hasattr(self, "predictor"):
            self.load_model()
        return self.predictor.predict(history, context, context_mask)

    def save_model(self, path):
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.output / "last.pt", destination)

    def finish(self):
        self.closed = True
