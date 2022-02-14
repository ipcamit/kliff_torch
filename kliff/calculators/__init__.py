from kliff.utils import torch_available

from .calculator import Calculator
from .calculator_new import CalculatorNew

__all__ = ["Calculator","CalculatorNew"]


if torch_available():
    from .calculator_torch import CalculatorTorch

    __all__.extend(["CalculatorTorch"])
