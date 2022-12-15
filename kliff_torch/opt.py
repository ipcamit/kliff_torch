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


#
# class OptimizerScipy:
#     """
#
#     """
#     def __init__(
#         self,
#         model_fn: Union[List[Callable], Callable],
#         parameters: Union[List[Parameter], List[OptimizingParameters]],
#         dataset: DatasetAutoloader,
#         weights: Dict = {"energy": 1.0, "forces": 1.0, "stress": 1.0},
#         optimizer: Optional[Callable] = None,
#         optimizer_kwargs: Optional[Dict] = None,
#         max_iter: Optional[int] = 1000,
#         target_property: List =["energy"]
#     ):
#         # TODO: parallelization of the optimizer based on torch and mpi
#         self.model_fn = model_fn if not callable(model_fn) else [model_fn]
#         self.parameters = parameters
#         self.max_iter = max_iter
#         self.optimizer_kwargs = optimizer_kwargs
#         self.optimizer_str = self._get_optimizer(optimizer)
#         self.dataset = dataset
#         self.weights = weights
#         self.loss_agg_func = lambda x, y: np.mean(np.sum((x - y) ** 2))
#         self.target_property = target_property
#
#     def _get_optimizer(self, optimizer_str):
#         scipy_minimize_methods = [
#             "Nelder-Mead",
#             "Powell",
#             "CG",
#             "BFGS",
#             "Newton-CG",
#             "L-BFGS-B",
#             "TNC",
#             "COBYLA",
#              "SLSQP",
#             "trust-constr",
#             "dogleg",
#             "trust-ncg",
#             "trust-exact",
#             "trust-krylov",
#         ]
#         scipy_minimize_methods_not_supported_args = ["bounds"]
#         scipy_least_squares_methods = ["trf", "dogbox", "lm", "geodesiclm"]
#         scipy_least_squares_methods_not_supported_args = ["bounds"]
#
#         if optimizer_str in scipy_minimize_methods:
#             return optimizer_str
#         elif optimizer_str == "geodesiclm":
#             if geodesicLM_avail:
#                 return optimizer_str
#             else:
#                 logger.error("GeodesicLM not loaded")
#         else:
#             logger.warning(f"Optimization method {optimizer_str} not found in scipy, switching to default L-BFGS")
#             return "L-BFGS-B"
#
#         raise ValueError("No optimizer provided")
#
#     def update_parameters(self, new_parameters:  List[Union[Parameter, OptimizingParameters]]):
#         for model in self.model_fn:
#             for new_parameter, parameter in zip(new_parameters, self.parameters):
#                 model.copy_parameters(parameter, new_parameter)
#
#     def loss_fn(self, models, dataset, weights, properties):
#         loss = 0.0
#         # self.update_parameters(new_parameters)
#         for configuration in dataset:
#             for i, model in enumerate(models):
#                 model_eval = model(configuration)
#                 for property_ in properties:
#                     loss += weights[property_] * self.loss_agg_func(model_eval[property_], configuration.__getattribute__(property_))
#         return loss
#
#     def _scipy_loss_wrapper(self, new_parameters):
#         self.update_parameters(new_parameters)
#         loss = self.loss_fn(self.model_fn, self.dataset, self.weights, self.target_property)
#         return loss
#
#     def minimize(self, kwargs=None):
#         if kwargs:
#             kwargs = self.optimizer_kwargs
#         else:
#             kwargs = {}
#         x0 = list(map(lambda x: x[1], self.parameters))
#         logger.info(f"Starting with method {self.optimizer_str}")
#         result = scipy.optimize.minimize(self._scipy_loss_wrapper, np.array(x0), method=self.optimizer_str, **kwargs)
#         return result


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
        target_property: List =["energy"]
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

    def update_parameters(self, new_parameters:  List[Union[Parameter, OptimizingParameters]]):
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
                    loss += weights[property_] * self.loss_agg_func(model_eval[property_], configuration.__getattribute__(property_))
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
                self.parameters = list(model_fn.parameters()) # If model_fn torch module
            except TypeError:
                self.parameters = list(model_fn.get_parameters()) # if model_fn is a TrainingWheel
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
        torch_minimize_methods = [ "Adadelta", "Adagrad", "Adam", "SparseAdam", "Adamax", "ASGD", "LBFGS",
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

    def update_parameters(self, new_parameters:  List[Parameter]):
        for new_parameter, parameter in zip(new_parameters, self.parameters):
            with torch.no_grad():
                parameter.copy_(new_parameter)

    def loss_fn(self, model, dataset, weights, properties):
        loss = torch.tensor(0.0)
        for configuration in dataset:
            model_eval = model(configuration)
            for property_ in properties:
                loss += weights[property_] * self.loss_agg_func(model_eval[property_], configuration.__getattribute__(property_))
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



# class LossPhysicsMotivatedModel:
#     """
#     Loss function class to optimize the physics-based potential parameters.
#
#     Args:
#         calculator: CalculatorNew to compute prediction from atomic configuration using
#             a potential model.
#         nprocs: Number of processes to use..
#         residual_fn: function to compute residual, e.g. :meth:`energy_forces_residual`,
#             :meth:`energy_residual`, and :meth:`forces_residual`. See the documentation
#             of :meth:`energy_forces_residual` for the signature of the function.
#             Default to :meth:`energy_forces_residual`.
#         residual_data: data passed to ``residual_fn``; can be used to fine tune the
#             residual function. Default to
#             {
#                 "energy_weight": 1.0,
#                 "forces_weight": 1.0,
#                 "stress_weight": 1.0,
#                 "normalize_by_natoms": True,
#             }
#             See the documentation of :meth:`energy_forces_residual` for more.
#     """
#
#     scipy_minimize_methods = [
#         "Nelder-Mead",
#         "Powell",
#         "CG",
#         "BFGS",
#         "Newton-CG",
#         "L-BFGS-B",
#         "TNC",
#         "COBYLA",
#         "SLSQP",
#         "trust-constr",
#         "dogleg",
#         "trust-ncg",
#         "trust-exact",
#         "trust-krylov",
#     ]
#     scipy_minimize_methods_not_supported_args = ["bounds"]
#     scipy_least_squares_methods = ["trf", "dogbox", "lm", "geodesiclm"]
#     scipy_least_squares_methods_not_supported_args = ["bounds"]
#
#     def __init__(
#         self,
#         calculator: Calculator,
#         nprocs: int = 1,
#         residual_fn: Optional[Callable] = None,
#         residual_data: Optional[Dict[str, Any]] = None,
#     ):
#
#         default_residual_data = {
#             "energy_weight": 1.0,
#             "forces_weight": 1.0,
#             "stress_weight": 1.0,
#             "normalize_by_natoms": True,
#         }
#
#         residual_data = _check_residual_data(residual_data, default_residual_data)
#         _check_compute_flag(calculator, residual_data)
#
#         self.calculator = calculator
#         self.nprocs = nprocs
#
#         self.residual_fn = (
#             energy_forces_residual if residual_fn is None else residual_fn
#         )
#         self.residual_data = residual_data
#
#         logger.debug(f"`{self.__class__.__name__}` instantiated.")
#
#     def minimize(self, method: str = "L-BFGS-B", **kwargs):
#         """
#         Minimize the loss.
#
#         Args:
#             method: minimization methods as specified at:
#                 https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
#                 https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
#
#             kwargs: extra keyword arguments that can be used by the scipy optimizer
#         """
#         kwargs = self._adjust_kwargs(method, **kwargs)
#
#         logger.info(f"Start minimization using method: {method}.")
#         result = self._scipy_optimize(method, **kwargs)
#         logger.info("Finish minimization using method: {method}.")
#
#         # update final optimized parameters
#         self.calculator.update_model_params(result.x)
#
#         return result
#
#     def _adjust_kwargs(self, method, **kwargs):
#         """
#         Check kwargs and adjust them as necessary.
#         """
#
#         if method in self.scipy_least_squares_methods:
#
#             # check support status
#             for i in self.scipy_least_squares_methods_not_supported_args:
#                 if i in kwargs:
#                     raise LossError(
#                         f"Argument `{i}` should not be set via the `minimize` method. "
#                         "It it set internally."
#                     )
#
#             # adjust bounds
#             if self.calculator.has_opt_params_bounds():
#                 if method in ["trf", "dogbox"]:
#                     bounds = self.calculator.get_opt_params_bounds()
#                     lb = [b[0] if b[0] is not None else -np.inf for b in bounds]
#                     ub = [b[1] if b[1] is not None else np.inf for b in bounds]
#                     bounds = (lb, ub)
#                     kwargs["bounds"] = bounds
#                 else:
#                     raise LossError(f"Method `{method}` cannot handle bounds.")
#
#         elif method in self.scipy_minimize_methods:
#
#             # check support status
#             for i in self.scipy_minimize_methods_not_supported_args:
#                 if i in kwargs:
#                     raise LossError(
#                         f"Argument `{i}` should not be set via the `minimize` method. "
#                         "It it set internally."
#                     )
#
#             # adjust bounds
#             if self.calculator.has_opt_params_bounds():
#                 if method in ["L-BFGS-B", "TNC", "SLSQP"]:
#                     bounds = self.calculator.get_opt_params_bounds()
#                     kwargs["bounds"] = bounds
#                 else:
#                     raise LossError(f"Method `{method}` cannot handle bounds.")
#         else:
#             raise LossError(f"Minimization method `{method}` not supported.")
#
#         return kwargs
#
#     def _scipy_optimize(self, method, **kwargs):
#         """
#         Minimize the loss use scipy.optimize.least_squares or scipy.optimize.minimize
#         methods. A user should not call this function, but should call the ``minimize``
#         method.
#         """
#
#         size = parallel.get_MPI_world_size()
#
#         if size > 1:
#             comm = MPI.COMM_WORLD
#             rank = comm.Get_rank()
#             logger.info(f"Running in MPI mode with {size} processes.")
#
#             if self.nprocs > 1:
#                 logger.warning(
#                     f"Argument `nprocs = {self.nprocs}` provided at initialization is "
#                     f"ignored. When running in MPI mode, the number of processes "
#                     f"provided along with the `mpiexec` (or `mpirun`) command is used."
#                 )
#
#             x = self.calculator.get_opt_params()
#             if method in self.scipy_least_squares_methods:
#                 # geodesic LM
#                 if method == "geodesiclm":
#                     if not geodesicLM_avail:
#                         report_import_error("geodesiclm")
#                     else:
#                         minimize_fn = geodesiclm
#                 else:
#                     minimize_fn = scipy.optimize.least_squares
#                 func = self._get_residual_MPI
#
#             elif method in self.scipy_minimize_methods:
#                 minimize_fn = scipy.optimize.minimize
#                 func = self._get_loss_MPI
#
#             if rank == 0:
#                 result = minimize_fn(func, x, method=method, **kwargs)
#                 # notify other process to break func
#                 break_flag = True
#                 for i in range(1, size):
#                     comm.send(break_flag, dest=i, tag=i)
#             else:
#                 func(x)
#                 result = None
#
#             result = comm.bcast(result, root=0)
#
#             return result
#
#         else:
#             # 1. running MPI with 1 process
#             # 2. running without MPI at all
#             # both cases are regarded as running without MPI
#
#             if self.nprocs == 1:
#                 logger.info("Running in serial mode.")
#             else:
#                 logger.info(
#                     f"Running in multiprocessing mode with {self.nprocs} processes."
#                 )
#
#                 # Maybe one thinks he is using MPI because nprocs is used
#                 if mpi4py_avail:
#                     logger.warning(
#                         "`mpi4y` detected. If you try to run in MPI mode, you should "
#                         "execute your code via `mpiexec` (or `mpirun`). If not, ignore "
#                         "this message."
#                     )
#
#             x = self.calculator.get_opt_params()
#             if method in self.scipy_least_squares_methods:
#                 if method == "geodesiclm":
#                     if not geodesicLM_avail:
#                         report_import_error("geodesiclm")
#                     else:
#                         minimize_fn = geodesiclm
#                 else:
#                     minimize_fn = scipy.optimize.least_squares
#
#                 func = self._get_residual
#             elif method in self.scipy_minimize_methods:
#                 minimize_fn = scipy.optimize.minimize
#                 func = self._get_loss
#
#             result = minimize_fn(func, x, method=method, **kwargs)
#             return result
#
#     def _get_residual(self, x):
#         """
#         Compute the residual in serial or multiprocessing mode.
#
#         This is a callable for optimizing method in scipy.optimize.least_squares,
#         which is passed as the first positional argument.
#
#         Args:
#             x: optimizing parameter values, 1D array
#         """
#
#         # publish params x to predictor
#         self.calculator.update_model_params(x)
#
#         cas = self.calculator.get_compute_arguments()
#
#         # TODO the if else could be combined
#         if isinstance(self.calculator, _WrapperCalculator):
#             calc_list = self.calculator.get_calculator_list()
#             X = zip(cas, calc_list)
#             if self.nprocs > 1:
#                 residuals = parallel.parmap2(
#                     self._get_residual_single_config,
#                     X,
#                     self.residual_fn,
#                     self.residual_data,
#                     nprocs=self.nprocs,
#                     tuple_X=True,
#                 )
#                 residual = np.concatenate(residuals)
#             else:
#                 residual = []
#                 for ca, calc in X:
#                     current_residual = self._get_residual_single_config(
#                         ca, calc, self.residual_fn, self.residual_data
#                     )
#                     residual = np.concatenate((residual, current_residual))
#
#         else:
#             if self.nprocs > 1:
#                 residuals = parallel.parmap2(
#                     self._get_residual_single_config,
#                     cas,
#                     self.calculator,
#                     self.residual_fn,
#                     self.residual_data,
#                     nprocs=self.nprocs,
#                     tuple_X=False,
#                 )
#                 residual = np.concatenate(residuals)
#             else:
#                 residual = []
#                 for ca in cas:
#                     current_residual = self._get_residual_single_config(
#                         ca, self.calculator, self.residual_fn, self.residual_data
#                     )
#                     residual = np.concatenate((residual, current_residual))
#
#         return residual
#
#     def _get_loss(self, x):
#         """
#         Compute the loss in serial or multiprocessing mode.
#
#         This is a callable for optimizing method in scipy.optimize.minimize,
#         which is passed as the first positional argument.
#
#         Args:
#             x: 1D array, optimizing parameter values
#         """
#         residual = self._get_residual(x)
#         loss = 0.5 * np.linalg.norm(residual) ** 2
#         return loss
#
#     def _get_residual_MPI(self, x):
#         def residual_my_chunk(x):
#             # broadcast parameters
#             x = comm.bcast(x, root=0)
#             # publish params x to predictor
#             self.calculator.update_model_params(x)
#
#             residual = []
#             for ca in cas:
#                 current_residual = self._get_residual_single_config(
#                     ca, self.calculator, self.residual_fn, self.residual_data
#                 )
#                 residual.extend(current_residual)
#             return residual
#
#         comm = MPI.COMM_WORLD
#         rank = comm.Get_rank()
#         size = comm.Get_size()
#
#         # get my chunk of data
#         cas = self._split_data()
#
#         while True:
#
#             if rank == 0:
#                 break_flag = False
#                 for i in range(1, size):
#                     comm.send(break_flag, dest=i, tag=i)
#                 residual = residual_my_chunk(x)
#                 all_residuals = comm.gather(residual, root=0)
#                 return np.concatenate(all_residuals)
#             else:
#                 break_flag = comm.recv(source=0, tag=rank)
#                 if break_flag:
#                     break
#                 else:
#                     residual = residual_my_chunk(x)
#                     all_residuals = comm.gather(residual, root=0)
#
#     def _get_loss_MPI(self, x):
#         comm = MPI.COMM_WORLD
#         rank = comm.Get_rank()
#
#         residual = self._get_residual_MPI(x)
#         if rank == 0:
#             loss = 0.5 * np.linalg.norm(residual) ** 2
#         else:
#             loss = None
#
#         return loss
#
#     # NOTE this function can be called only once, no need to call it each time
#     # _get_residual_MPI is called
#     def _split_data(self):
#         comm = MPI.COMM_WORLD
#         rank = comm.Get_rank()
#         size = comm.Get_size()
#
#         # get a portion of data based on rank
#         cas = self.calculator.get_compute_arguments()
#         # random.shuffle(cas)
#
#         rank_size = len(cas) // size
#         # last rank deal with the case where len(cas) cannot evenly divide size
#         if rank == size - 1:
#             cas = cas[rank_size * rank :]
#         else:
#             cas = cas[rank_size * rank : rank_size * (rank + 1)]
#
#         return cas
#
#     @staticmethod
#     def _get_residual_single_config(ca, calculator, residual_fn, residual_data):
#
#         # prediction data
#         calculator.compute(ca)
#         pred = calculator.get_prediction(ca)
#
#         # reference data
#         ref = calculator.get_reference(ca)
#
#         conf = ca.conf
#         identifier = conf.identifier
#         weight = conf.weight
#         natoms = conf.get_num_atoms()
#
#         residual = residual_fn(identifier, natoms, weight, pred, ref, residual_data)
#
#         return residual
#
#
# class LossNeuralNetworkModel(object):
#     """
#     Loss function class to optimize the ML potential parameters.
#
#     This is a wrapper over :class:`LossPhysicsMotivatedModel` and
#     :class:`LossNeuralNetworkModel` to provide a united interface. You can use the two
#     classes directly.
#
#     Args:
#         calculator: CalculatorNew to compute prediction from atomic configuration using
#             a potential model.
#         nprocs: Number of processes to use..
#         residual_fn: function to compute residual, e.g. :meth:`energy_forces_residual`,
#             :meth:`energy_residual`, and :meth:`forces_residual`. See the documentation
#             of :meth:`energy_forces_residual` for the signature of the function.
#             Default to :meth:`energy_forces_residual`.
#         residual_data: data passed to ``residual_fn``; can be used to fine tune the
#             residual function. Default to
#             {
#                 "energy_weight": 1.0,
#                 "forces_weight": 1.0,
#                 "stress_weight": 1.0,
#                 "normalize_by_natoms": True,
#             }
#             See the documentation of :meth:`energy_forces_residual` for more.
#     """
#
#     torch_minimize_methods = [
#         "Adadelta",
#         "Adagrad",
#         "Adam",
#         "SparseAdam",
#         "Adamax",
#         "ASGD",
#         "LBFGS",
#         "RMSprop",
#         "Rprop",
#         "SGD",
#     ]
#
#     def __init__(
#         self,
#         calculator,
#         nprocs: int = 1,
#         residual_fn: Optional[Callable] = None,
#         residual_data: Optional[Dict[str, Any]] = None,
#     ):
#
#         if not torch_avail:
#             report_import_error("pytorch")
#
#         default_residual_data = {
#             "energy_weight": 1.0,
#             "forces_weight": 1.0,
#             "stress_weight": 1.0,
#             "normalize_by_natoms": True,
#         }
#
#         residual_data = _check_residual_data(residual_data, default_residual_data)
#         _check_compute_flag(calculator, residual_data)
#
#         self.calculator = calculator
#         self.nprocs = nprocs
#
#         self.residual_fn = (
#             energy_forces_residual if residual_fn is None else residual_fn
#         )
#         self.residual_data = residual_data
#
#         self.optimizer = None
#         self.optimizer_state_path = None
#
#         logger.debug(f"`{self.__class__.__name__}` instantiated.")
#
#     def minimize(
#         self,
#         method: str = "Adam",
#         batch_size: int = 100,
#         num_epochs: int = 1000,
#         start_epoch: int = 0,
#         **kwargs,
#     ):
#         """
#         Minimize the loss.
#
#         Args:
#             method: PyTorch optimization methods, and available ones are:
#                 [`Adadelta`, `Adagrad`, `Adam`, `SparseAdam`, `Adamax`, `ASGD`, `LBFGS`,
#                 `RMSprop`, `Rprop`, `SGD`]
#                 See also: https://pytorch.org/docs/stable/optim.html
#             batch_size: Number of configurations used in in each minimization step.
#             num_epochs: Number of epochs to carry out the minimization.
#             start_epoch: The starting epoch number. This is typically 0, but if
#                 continuing a training, it is useful to set this to the last epoch number
#                 of the previous training.
#             kwargs: Extra keyword arguments that can be used by the PyTorch optimizer.
#         """
#         if method not in self.torch_minimize_methods:
#             raise LossError("Minimization method `{method}` not supported.")
#
#         self.method = method
#         self.batch_size = batch_size
#         self.num_epochs = num_epochs
#         self.start_epoch = start_epoch
#
#         # model save metadata
#         save_prefix = self.calculator.model.save_prefix
#         save_start = self.calculator.model.save_start
#         save_frequency = self.calculator.model.save_frequency
#
#         if save_prefix is None or save_start is None or save_frequency is None:
#             logger.info(
#                 "Model saving meta data not set by user. Now set it to "
#                 '"prefix=./kliff_saved_model", "start=1", and "frequency=10".'
#             )
#             save_prefix = os.path.join(os.getcwd(), "kliff_saved_model")
#             save_start = 1
#             save_frequency = 10
#             self.calculator.model.set_save_metadata(
#                 save_prefix, save_start, save_frequency
#             )
#
#         logger.info(f"Start minimization using optimization method: {self.method}.")
#
#         # optimizing
#         try:
#             self.optimizer = getattr(torch.optim, method)(
#                 self.calculator.model.parameters(), **kwargs
#             )
#             if self.optimizer_state_path is not None:
#                 self._load_optimizer_stat(self.optimizer_state_path)
#
#         except TypeError as e:
#             print(str(e))
#             idx = str(e).index("argument '") + 10
#             err_arg = str(e)[idx:].strip("'")
#             raise LossError(
#                 f"Argument `{err_arg}` not supported by optimizer `{method}`."
#             )
#
#         # data loader
#         loader = self.calculator.get_compute_arguments(batch_size)
#
#         epoch = 0  # in case never enters loop
#         for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
#
#             # get the loss without any optimization if continue a training
#             if self.start_epoch != 0 and epoch == self.start_epoch:
#                 epoch_loss = self._get_loss_epoch(loader)
#                 print("Epoch = {:<6d}  loss = {:.10e}".format(epoch, epoch_loss))
#
#             else:
#                 epoch_loss = 0
#                 for ib, batch in enumerate(loader):
#
#                     def closure():
#                         self.optimizer.zero_grad()
#                         loss = self._get_loss_batch(batch)
#                         loss.backward()
#                         return loss
#
#                     loss = self.optimizer.step(closure)
#                     # float() such that do not accumulate history, more memory friendly
#                     epoch_loss += float(loss)
#
#                 print("Epoch = {:<6d}  loss = {:.10e}".format(epoch, epoch_loss))
#                 if epoch >= save_start and (epoch - save_start) % save_frequency == 0:
#                     path = os.path.join(save_prefix, "model_epoch{}.pkl".format(epoch))
#                     self.calculator.model.save(path)
#
#         # print loss from final parameter and save last epoch
#         epoch += 1
#         epoch_loss = self._get_loss_epoch(loader)
#         print("Epoch = {:<6d}  loss = {:.10e}".format(epoch, epoch_loss))
#         path = os.path.join(save_prefix, "model_epoch{}.pkl".format(epoch))
#         self.calculator.model.save(path)
#
#         logger.info(f"Finish minimization using optimization method: {self.method}.")
#
#     def _get_loss_epoch(self, loader):
#         epoch_loss = 0
#         for ib, batch in enumerate(loader):
#             loss = self._get_loss_batch(batch)
#             epoch_loss += float(loss)
#         return epoch_loss
#
#     # TODO this is nice since it is simple and gives user the opportunity to provide a
#     #  loss function based on each data point. However, this is slow without
#     #  vectorization. Should definitely modify it and use vectorized loss function.
#     #  The way going forward is to batch all necessary info in dataloader.
#     #  The downsides is that then analytic and machine learning models will have
#     #  different interfaces.
#     def _get_loss_batch(self, batch: List[Any], normalize: bool = True):
#         """
#         Compute the loss of a batch of samples.
#
#         Args:
#             batch: A list of samples.
#             normalize: If `True`, normalize the loss of the batch by the size of the
#                 batch. Note, how to normalize the loss of a single configuration is
#                 determined by the `normalize` flag of `residual_data`.
#         """
#         results = self.calculator.compute(batch)
#         energy_batch = results["energy"]
#         forces_batch = results["forces"]
#         stress_batch = results["stress"]
#
#         if forces_batch is None:
#             forces_batch = [None] * len(batch)
#         if stress_batch is None:
#             stress_batch = [None] * len(batch)
#
#         # Instead of loss_batch = 0 and loss_batch += loss in the loop, the below one may
#         # be faster, considering chain rule it needs to take derivatives.
#         # Anyway, it is minimal. Don't worry about it.
#         losses = []
#         for sample, energy, forces, stress in zip(
#             batch, energy_batch, forces_batch, stress_batch
#         ):
#             loss = self._get_loss_single_config(sample, energy, forces, stress)
#             losses.append(loss)
#         loss_batch = torch.stack(losses).sum()
#         if normalize:
#             loss_batch /= len(batch)
#
#         return loss_batch
#
#     def _get_loss_single_config(self, sample, pred_energy, pred_forces, pred_stress):
#
#         device = self.calculator.model.device
#
#         if self.calculator.use_energy:
#             pred = pred_energy.reshape(-1)  # reshape scalar as 1D tensor
#             ref = sample["energy"].reshape(-1).to(device)
#
#         if self.calculator.use_forces:
#             ref_forces = sample["forces"].to(device)
#             if self.calculator.use_energy:
#                 pred = torch.cat((pred, pred_forces.reshape(-1)))
#                 ref = torch.cat((ref, ref_forces.reshape(-1)))
#             else:
#                 pred = pred_forces.reshape(-1)
#                 ref = ref_forces.reshape(-1)
#
#         if self.calculator.use_stress:
#             ref_stress = sample["stress"].to(device)
#             if self.calculator.use_energy or self.calculator.use_stress:
#                 pred = torch.cat((pred, pred_stress.reshape(-1)))
#                 ref = torch.cat((ref, ref_stress.reshape(-1)))
#             else:
#                 pred = pred_stress.reshape(-1)
#                 ref = ref_stress.reshape(-1)
#
#         conf = sample["configuration"]
#         identifier = conf.identifier
#         weight = conf.weight
#         natoms = conf.get_num_atoms()
#
#         residual = self.residual_fn(
#             identifier, natoms, weight, pred, ref, self.residual_data
#         )
#         loss = torch.sum(torch.pow(residual, 2))
#
#         return loss
#
#     def save_optimizer_state(self, path="optimizer_state.pkl"):
#         """
#         Save the state dict of optimizer to disk.
#         """
#         torch.save(self.optimizer.state_dict(), path)
#
#     def load_optimizer_state(self, path="optimizer_state.pkl"):
#         """
#         Load the state dict of optimizer from file.
#         """
#         self.optimizer_state_path = path
#
#     def _load_optimizer_stat(self, path):
#         self.optimizer.load_state_dict(torch.load(path))
#

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
