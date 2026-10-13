import torch.nn as nn
from torch import Tensor
from torch.nn import Module, ModuleDict, ModuleList
from torch.nn.parallel import DistributedDataParallel
from xuance.torch.utils import set_device, collect_device_info
from importlib import import_module


def __getattr__(name):
    # Supervised extensions need device utilities without online RL loggers.
    modules = {"REGISTRY_Representation": "representations", "REGISTRY_Policy": "policies",
               "REGISTRY_Learners": "learners", "REGISTRY_Agents": "agents"}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(f"xuance.torch.{modules[name]}"), name)
    globals()[name] = value
    return value

__all__ = [
    "nn",
    "Tensor",
    "Module",
    "ModuleDict",
    "ModuleList",
    "DistributedDataParallel",
    "set_device", "collect_device_info",
    "REGISTRY_Representation", "REGISTRY_Policy", "REGISTRY_Learners", "REGISTRY_Agents",
]
