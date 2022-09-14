__version__ = "0.3.1"

import warnings

from kliff_torch.log import set_logger
from kliff_torch.utils import torch_available

set_logger(level="INFO", stderr=True)

if not torch_available():
    warnings.warn(
        "'PyTorch' not found. All kliff_torch machine learning modules (e.g. NeuralNetwork) "
        "are not imported. Ignore this if you want to use kliff_torch to train "
        "physics-based models."
    )
