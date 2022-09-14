from kliff_torch.models.kim import KIMModel
from kliff_torch.models.lennard_jones import LennardJones
from kliff_torch.models.model import ComputeArguments, Model
from kliff_torch.models.parameter import OptimizingParameters, Parameter
from kliff_torch.utils import torch_available

__all__ = [
    "Parameter",
    "OptimizingParameters",
    "ComputeArguments",
    "Model",
    "LennardJones",
    "KIMModel",
]

if torch_available():
    from kliff_torch.models.linear_regression import LinearRegression
    from kliff_torch.models.model_torch import ModelTorch
    from kliff_torch.models.neural_network import NeuralNetwork

    __all__.extend(["ModelTorch", "NeuralNetwork", "LinearRegression"])
