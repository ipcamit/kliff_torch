import os
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
# import scipy.optimize
from loguru import logger

# from kliff_torch import parallel
# from kliff_torch.calculators.calculator import Calculator, _WrapperCalculator
from kliff_torch.error import report_import_error
from kliff_torch.models.parameter import OptimizingParameters
# from .loss import Loss
from kliff_torch.dataset import Dataset

import torch
from torch.nn import Parameter, Module

try:
    from mpi4py import MPI

    mpi4py_avail = True
except ImportError:
    mpi4py_avail = False

try:
    from geodesicLM import geodesiclm

    geodesicLM_avail = True
except ImportError:
    geodesicLM_avail = False


class OptimizerScipy:
    """

    """

    def __init__(
            self,
            model_fn: Callable,
            parameters: Union[List[Parameter], List[OptimizingParameters]],
            dataset: Dataset,
            weights: Dict = {"energy": 1.0, "forces": 1.0, "stress": 1.0},
            optimizer: Optional[Callable] = None,
            optimizer_kwargs: Optional[Dict] = None,
            max_iter: Optional[int] = 1000,
            target_property: List = ["energy"]
    ):
        # TODO: parallelization of the optimizer based on torch and mpi
        self.model_fn = model_fn if not callable(model_fn) else [model_fn]
        self.parameters = parameters
        self.max_iter = max_iter
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_str = self._get_optimizer(optimizer)
        self.dataset = dataset
        self.weights = weights
        self.loss_agg_func = lambda x, y: np.mean(np.sum((x - y) ** 2))
        self.target_property = target_property

    def _get_optimizer(self, optimizer_str):
        scipy_minimize_methods = [
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ]
        scipy_minimize_methods_not_supported_args = ["bounds"]
        scipy_least_squares_methods = ["trf", "dogbox", "lm", "geodesiclm"]
        scipy_least_squares_methods_not_supported_args = ["bounds"]

        if optimizer_str in scipy_minimize_methods:
            return optimizer_str
        elif optimizer_str == "geodesiclm":
            if geodesicLM_avail:
                return optimizer_str
            else:
                logger.error("GeodesicLM not loaded")
        else:
            logger.warning(f"Optimization method {optimizer_str} not found in scipy, switching to default L-BFGS")
            return "L-BFGS-B"

        raise ValueError("No optimizer provided")

    def update_parameters(self, new_parameters: List[Union[Parameter, OptimizingParameters]]):
        for model in self.model_fn:
            for new_parameter, parameter in zip(new_parameters, self.parameters):
                model.copy_parameters(parameter, new_parameter)

    def loss_fn(self, models, dataset, weights, properties):
        loss = 0.0
        # self.update_parameters(new_parameters)
        for configuration in dataset:
            for i, model in enumerate(models):
                model_eval = model(configuration)
                for property_ in properties:
                    loss += weights[property_] * self.loss_agg_func(model_eval[property_],
                                                                    configuration.__getattribute__(property_))
        return loss

    def _scipy_loss_wrapper(self, new_parameters):
        self.update_parameters(new_parameters)
        loss = self.loss_fn(self.model_fn, self.dataset, self.weights, self.target_property)
        return loss

    def minimize(self, kwargs=None):
        if kwargs:
            kwargs = self.optimizer_kwargs
        else:
            kwargs = {}
        x0 = list(map(lambda x: x[1], self.parameters))
        logger.info(f"Starting with method {self.optimizer_str}")
        result = scipy.optimize.minimize(self._scipy_loss_wrapper, np.array(x0), method=self.optimizer_str, **kwargs)
        return result


class OptimizerTorch:
    """ Optimizer for torch models. This class provides an optimized for the torch models. It is based on the torch
    optimizers and the torch autograd. It can be used for optimizing general pytorch functions with autograd as well.
    The parameters to be are either inferred from the model or can be provided as a list of torch.nn.Parameter objects.
    Dataset has to be an instance of the torch.utils.data.Dataset class or kliff_torch.dataset.Dataset class.
    Weight is expected to be a dictionary with the keys "energy", "forces", and "stress" and the values should be a
    valid torch broadcastable array. The target property is a list of the properties to be optimized. The default is
    energy. The loss_agg_func is a function that takes the model output and the target and returns a scalar value.
    epochs is the number of iterations to be performed.
    Models should either return a named tuple with  three values (energy, forces, stress) or energy if forces are to
    be computed using autograd.
        params:
            model_fn: model function to be optimized. Has to be a torch.nn.Module
            parameters: list of parameters to optimize.
            dataset: dataset to optimize on
            weights: weights for the different properties
            optimizer: optimizer to use
            optimizer_kwargs: kwargs for the optimizer
            epochs: maximum number of iterations
            target_property: list of properties to optimize on

    """

    def __init__(
            self,
            model_fn: Union[Callable, Module],
            dataset: Dataset,
            weights: Dict = None,
            optimizer: Optional[Callable] = "Adam",
            optimizer_kwargs: Optional[Dict] = None,
            epochs: Optional[int] = 100,
            target_property: List = None,
            parameters: Union[List[Parameter], List[OptimizingParameters]] = None
    ):
        # TODO: parallelization of the optimizer based on torch and mpi
        self.model_fn = model_fn
        if parameters:
            self.parameters = parameters
        elif isinstance(model_fn, Module):
            try:
                self.parameters = list(model_fn.parameters())  # If model_fn torch module
            except TypeError:
                self.parameters = list(model_fn.get_parameters())  # if model_fn is a TrainingWheel
        else:
            raise ValueError("No parameters provided")

        if not weights:
            weights = {"energy": 1.0, "forces": 1.0, "stress": 1.0}
        self.weights = weights

        self.epochs = epochs
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = self._get_optimizer(optimizer)
        self.dataset = dataset
        self.weights = weights
        self.loss_agg_func = lambda x, y: torch.mean(torch.sum((x - y) ** 2))
        if not target_property:
            target_property = ["energy"]
        self.target_property = target_property
        self.print_loss = False

    def _get_optimizer(self, optimizer_str):
        torch_minimize_methods = ["Adadelta", "Adagrad", "Adam", "SparseAdam", "Adamax", "ASGD", "LBFGS",
                                  "RMSprop", "Rprop", "SGD"]
        if isinstance(optimizer_str, str):
            if optimizer_str in torch_minimize_methods:
                return getattr(torch.optim, optimizer_str)(self.parameters)
            else:
                logger.warning(f"Optimization method {optimizer_str} not found currently supported list, "
                               f"switching to default Adam")
                return torch.optim.Adam(self.parameters)
        elif "torch.optim" in str(type(optimizer_str)):
            return optimizer_str
        else:
            raise ValueError("No optimizer provided")

    def update_parameters(self, new_parameters: List[Parameter]):
        for new_parameter, parameter in zip(new_parameters, self.parameters):
            with torch.no_grad():
                parameter.copy_(new_parameter)

    def loss_fn(self, model, dataset, weights, properties):
        loss = torch.tensor(0.0)
        for configuration in dataset:
            model_eval = model(configuration)
            for property_ in properties:
                loss += weights[property_] * self.loss_agg_func(model_eval[property_],
                                                            torch.as_tensor(configuration.__getattribute__(property_)))
        return loss

    def minimize(self, kwargs=None):
        if kwargs:
            kwargs = self.optimizer_kwargs
        else:
            kwargs = {}

        logger.info(f"Starting with method {self.optimizer}")
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model_fn, self.dataset, self.weights, self.target_property)
            loss.backward()
            self.optimizer.step()
            if self.print_loss:
                print(f"Epoch {epoch} loss {loss}")
        return self.parameters


def _check_residual_data(data: Dict[str, Any], default: Dict[str, Any]):
    """
    Check whether user provided residual data is valid, and add default values if not
    provided.
    """
    if data is not None:
        for key, value in data.items():
            if key not in default:
                raise LossError(
                    f"Expect the keys of `residual_data` to be one or combinations of "
                    f"{', '.join(default.keys())}; got {key}. "
                )
            else:
                default[key] = value

    return default


def _check_compute_flag(calculator, residual_data):
    """
    Check whether compute flag correctly set when the corresponding weight in residual
    data is 0.
    """
    ew = residual_data["energy_weight"]
    fw = residual_data["forces_weight"]
    sw = residual_data["stress_weight"]
    msg = (
        '"{0}_weight" set to "{1}". Seems you do not want to use {0} in the fitting. '
        'You can set "use_{0}" in "calculator.create()" to "False" to speed up the '
        "fitting."
    )

    if calculator.use_energy and ew < 1e-12:
        logger.warning(msg.format("energy", ew))
    if calculator.use_forces and fw < 1e-12:
        logger.warning(msg.format("forces", fw))
    if calculator.use_stress and sw < 1e-12:
        logger.warning(msg.format("stress", sw))


class LossError(Exception):
    def __init__(self, msg):
        super(LossError, self).__init__(msg)
        self.msg = msg
