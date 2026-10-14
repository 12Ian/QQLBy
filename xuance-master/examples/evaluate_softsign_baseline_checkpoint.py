"""Evaluate the Softsign variant using the existing held-out evaluation protocol."""
import torch
from xuance.torch.utils import ActivationFunctions
ActivationFunctions['softsign'] = torch.nn.Softsign

from examples.evaluate_baseline_checkpoint import main


if __name__ == '__main__':
    main()
