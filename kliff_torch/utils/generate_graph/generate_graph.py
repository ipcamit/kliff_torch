import torch
from kliff_torch.utils.generate_graph import tg
from kliff_torch.dataset import AutoloaderConfiguration

class KIMTorchGraph:
    def __init__(self, cutoff, n_layers):
        self.cutoff = cutoff
        self.n_layers = n_layers
        self.infl_dist = n_layers * cutoff
        self._tg = tg

    def generate_graph(self, configuration:AutoloaderConfiguration):
        graph = tg.get_complete_graph(
            self.n_layers,
            self.cutoff,
            configuration.species,
            configuration.coords,
            configuration.cell,
            configuration.PBC
        )

        graph.energy = torch.as_tensor(configuration.energy)
        graph.forces = torch.as_tensor(configuration.forces)

        return graph