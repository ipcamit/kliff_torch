"""
Compute the root-mean-square error (RMSE) of a model prediction and reference values in
the dataset.
"""

from kliff_torch.analyzers import EnergyForcesRMSE
from kliff_torch.calculators import Calculator
from kliff_torch.dataset import Dataset
from kliff_torch.models import KIMModel
from kliff_torch.utils import download_dataset

model = KIMModel(model_name="SW_StillingerWeber_1985_Si__MO_405512056662_006")

# load the trained model back
# model.load("kliff_model.yaml")


dataset_path = download_dataset(dataset_name="Si_training_set_4_configs")
tset = Dataset(dataset_path)
configs = tset.get_configs()

calc = Calculator(model)
calc.create(configs)

analyzer = EnergyForcesRMSE(calc)
analyzer.run(verbose=2, sort="energy")
