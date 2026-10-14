"""Explicit Softsign Actor variant of the native MATD3 baseline entrypoint."""
import torch
from xuance.torch.utils import ActivationFunctions

# Registration is confined to this process; historical baseline files stay fixed.
ActivationFunctions['softsign'] = torch.nn.Softsign

from examples.run_baseline import main


if __name__ == '__main__':
    main()
