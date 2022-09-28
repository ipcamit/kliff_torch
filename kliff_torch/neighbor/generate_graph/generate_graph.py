import torch
from kliff_torch.neighbor.generate_graph import tg
from kliff_torch.dataset import Configuration
from torch_geometric.data import Data


class KIMTorchGraph(Data):
    def __init__(self):
        super(KIMTorchGraph, self).__init__()
        self.energy = None
        self.forces = None
        self.n_layers = None
        self.pos = None
        self.images = None
        self.species = None
        self.contributions = None

    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        elif 'contributions' in key:
            return 2
        elif 'images' in key:
            return torch.max(value) + 1
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return 1
        else:
            return 0


class KIMTorchGraphGenerator:
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
            torch_geom_graph = KIMTorchGraph()
            torch_geom_graph.energy = graph.energy
            torch_geom_graph.forces = graph.forces
            torch_geom_graph.n_layers = graph.n_layers
            torch_geom_graph.pos = graph.pos
            torch_geom_graph.images = graph.images
            torch_geom_graph.species = graph.species
            torch_geom_graph.contributions = graph.contributions
            for i in range(graph.n_layers):
                torch_geom_graph.__setattr__(f"edge_index{i}", graph.edge_index[i])
            return torch_geom_graph

        return graph

    def collate_fn(self, config_list):
        graph_list = []
        for conf in config_list:
            graph = self.generate_graph(conf)
            graph_list.append(graph)
        return graph_list

    def collate_fn_single_conf(self, config_list):
        graph = self.generate_graph(config_list[0])
        return graph
