import numpy
from typing import Union, List, Callable, Optional

import numpy as np
import torch
from torch.nn import Parameter
from kliff_torch.models.parameter import OptimizingParameters
from kliff_torch.dataset import DatasetAutoloader, AutoloaderConfiguration


class Loss:
    """
    This class defines a callable loss function. A loss function or a loss module will have few necessary components.
    It is compulsory to have computation model or list of computation models, dataset/dataloader object, list of
    parameter objects, and wrapper loss function. If model parameters are not provided then Loss class will assume
    parameters from `model.parameters()`
    """

    def __init__(
        self,
        model_fn: Union[List[Callable], Callable],
        dataset: DatasetAutoloader,
        properties: str = "energy",
        weights: List = None,
        loss_agg_func: Callable = None,
    ):
        self.model_fn = model_fn if not callable(model_fn) else [model_fn]
        self.dataset = dataset
        self.weights = self._get_weights(weights)
        self.iterator_list = None
        self.properties = properties
        if loss_agg_func:
            self.loss_agg_func = loss_agg_func
        else:
            self.loss_agg_func = lambda x, y: np.mean(np.sum((x - y) ** 2))


    def dataset_iterator(
        self,
    ) -> Union[AutoloaderConfiguration, list[AutoloaderConfiguration]]:
        if self.iterator_list:
            unraveled_dataset = next(iter(self.dataset))
            for iter_idx in self.iterator_list:
                yield unraveled_dataset[iter_idx]
        else:
            yield next(iter(self.dataset))

    def evaluate(self):
        """
        This is a barebones function to show a basic loss function implementation. Ideally, any callable function shall
        work.
        Args:
            parameters:

        Returns:

        """
        loss = 0.0
        # self.update_parameters(new_parameters)
        for configuration in self.dataset:
            for i, model in enumerate(self.model_fn):
                loss += self.weights[i] * self.loss_agg_func(model(configuration), configuration.__getattribute__(self.properties))
        return loss

    # TODO: More coherent approach for monkey patching loss function evaluation, or making loss function more general
    # @classmethod
    # def make_loss(cls,  ):
    #     def _decorator(single_loss_fn: Callable):


    # def _get_if_torch_model(self, is_torch_model_val):
    #     if is_torch_model_val:
    #         is_torch_model = False
    #         is_generic_callable = False
    #         for model in self.model_fn:
    #             if isinstance(model, torch.nn.Module):
    #                 is_torch_model = True
    #             else:
    #                 if callable(model):
    #                     is_generic_callable = True
    #         if is_torch_model and is_generic_callable:
    #             raise ValueError("Models contain both torch.nn.Module and generic function, no optimizer can propagate losses through them yet. Please reimplemnt model in single framework")
    #         if (not is_torch_model) and (not is_generic_callable):
    #             raise ValueError("Invalid model; model functions should be callable")
    #         return is_torch_model
    #     else:
    #         return is_torch_model_val

    # def _get_parameters(self, parameters):
    #     parameters_list = []
    #     if parameters:
    #         parameters_list = parameters
    #     else:
    #         if isinstance(self.model_fn, list):
    #             for model in self.model_fn:
    #                 parameters_list = parameters_list.extend(model.parameters())
    #         else:
    #             parameters_list.extend(self.model_fn.parameters())
    #     return parameters_list
