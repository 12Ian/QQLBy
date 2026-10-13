"""Enter through real Xuance configuration, registry and Runner dispatch."""
from argparse import Namespace
from pathlib import Path
import sys


def framework():
    checkout = Path(__file__).resolve().parents[3]
    if (checkout / "xuance/__init__.py").exists() and str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    import xuance
    return xuance


def runner(data, output, cfg, resume=None, intention_checkpoint=None, checkpoint=None):
    xuance = framework()
    parser = Namespace(data_path=str(Path(data).resolve()), output_dir=str(Path(output).resolve()),
                       device=cfg.device, trajectory_overrides=cfg.to_dict(),
                       resume=resume, intention_checkpoint=intention_checkpoint, checkpoint=checkpoint)
    return xuance.get_runner(algo="trajectory_prediction", env="civilian_trajectory",
                             env_id="single_track", parser_args=parser)


def train_native(data, output, cfg, resume=None, intention_checkpoint=None):
    return runner(data, output, cfg, resume, intention_checkpoint).run(mode="train")


def evaluate_native(data, checkpoint, cfg, split="test", device="auto", output=None):
    cfg.device = device
    return runner(data, Path(checkpoint).resolve().parent, cfg, checkpoint=str(checkpoint)).run(
        mode="test", split=split, output_path=output)
