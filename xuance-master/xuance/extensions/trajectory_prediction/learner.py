from argparse import Namespace

from xuance.common import BaseCallback
from xuance.torch.learners.learner import Learner
from trajectory_prediction.learner import TrajectoryLearner


class XuanceTrajectoryLearner(TrajectoryLearner, Learner):
    """Xuance Learner with the shared supervised optimizer and objective.

    Iterations are minibatch updates, rather than environment transitions.
    The engine owns full model/optimizer/objective/scaler checkpoint persistence.
    """
    def __init__(self, cfg, policy, device, steps_per_epoch=1):
        values = cfg.to_dict()
        values.update(device=str(device), distributed_training=False,
                      episode_length=cfg.prediction_steps, use_grad_clip=True,
                      grad_clip_norm=cfg.gradient_clip, model_dir=".",
                      running_steps=cfg.epochs * steps_per_epoch, parallels=1)
        Learner.__init__(self, Namespace(**values), policy, BaseCallback())
        TrajectoryLearner.__init__(self, cfg, policy, device, steps_per_epoch)
