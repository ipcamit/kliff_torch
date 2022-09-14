"""
.. _tut_linear_regression:

Train a linear regression potential
===================================

In this tutorial, we train a linear regression model on the descriptors obtained using the
symmetry functions.
"""

from kliff_torch.calculators import CalculatorTorch
from kliff_torch.dataset import Dataset
from kliff_torch.descriptors import SymmetryFunction
from kliff_torch.models import LinearRegression
from kliff_torch.utils import download_dataset

descriptor = SymmetryFunction(
    cut_name="cos", cut_dists={"Si-Si": 5.0}, hyperparams="set30", normalize=True
)


model = LinearRegression(descriptor)

# training set
dataset_path = download_dataset(dataset_name="Si_training_set")
dataset_path = dataset_path.joinpath("varying_alat")
tset = Dataset(dataset_path)
configs = tset.get_configs()

# calculator
calc = CalculatorTorch(model)
calc.create(configs, reuse=False)


##########################################################################################
# We can train a linear regression model by minimizing a loss function as discussed in
# :ref:`tut_nn`. But linear regression model has analytic solutions, and thus we can train
# the model directly by using this feature. This can be achieved by calling the ``fit()``
# function of its calculator.
#

# fit the model
calc.fit()


# save model
model.save("linear_model.pkl")
