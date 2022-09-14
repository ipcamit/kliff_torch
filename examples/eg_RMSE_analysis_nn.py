"""
Compute the root-mean-square error (RMSE) of model prediction and reference values from
the dataset.
"""

from kliff_torch import nn
from kliff_torch.analyzers import EnergyForcesRMSE
from kliff_torch.calculators import CalculatorTorch
from kliff_torch.dataset import Dataset
from kliff_torch.descriptors import SymmetryFunction
from kliff_torch.models import NeuralNetwork
from kliff_torch.utils import download_dataset

# model
descriptor = SymmetryFunction(
    cut_name="cos", cut_dists={"Si-Si": 5.0}, hyperparams="set30", normalize=True
)

N1 = 10
N2 = 10
model = NeuralNetwork(descriptor)
model.add_layers(
    # first hidden layer
    nn.Linear(descriptor.get_size(), N1),
    nn.Tanh(),
    # second hidden layer
    nn.Linear(N1, N2),
    nn.Tanh(),
    # output layer
    nn.Linear(N2, 1),
)

# load the trained model back
# model.load("./saved_model.pkl", mode="eval")

# dataset
dataset_path = download_dataset(dataset_name="Si_training_set")
tset = Dataset(dataset_path)
configs = tset.get_configs()

# calculator
calc = CalculatorTorch(model)
calc.create(configs, reuse=False)

# analyzer
analyzer = EnergyForcesRMSE(calc)
analyzer.run(verbose=2, sort="energy")
