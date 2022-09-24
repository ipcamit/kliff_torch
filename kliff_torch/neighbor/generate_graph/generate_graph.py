import torch
from kliff_torch.neighbor.generate_graph import tg
from kliff_torch.dataset import Configuration
from torch_geometric.data import Data


class KIMTorchGraph:
    def __init__(self, cutoff, n_layers, as_torch_geometric_data=False):
        self.cutoff = cutoff
        self.n_layers = n_layers
        self.infl_dist = n_layers * cutoff
        self._tg = tg
        self.as_torch_geometric_data = as_torch_geometric_data

    def generate_graph(self, configuration:Configuration):
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

        if self.as_torch_geometric_data:
            torch_geom_graph = Data()
            torch_geom_graph.energy = graph.energy
            torch_geom_graph.forces = graph.forces
            torch_geom_graph.n_layers = graph.n_layers
            torch_geom_graph.pos = graph.pos
            torch_geom_graph.images = graph.images
            torch_geom_graph.elements = graph.elements
            torch_geom_graph.contributions = graph.contributions
            for i in range(graph.n_layers):
                torch_geom_graph.__setattr__(f"edge_index{i}", graph.edge_index[i])
            return torch_geom_graph

        return graph