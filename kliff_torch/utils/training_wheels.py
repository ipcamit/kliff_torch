import torch
from enum import Enum
from kliff_torch.neighbor import KIMTorchGraphGenerator, KIMTorchGraph
from torch_scatter import scatter

# Todo: link it with AvailableDescriptors in libdescriptor
class DescriptorFunctions(Enum):
    SymmetryFunctions = 1
    Bispectrum = 2

class KIMTorchDescriptorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, SymFunCtx, configuration, coordinates):
        """
        Coordinates tensor needs to be passed separately .
        Args:
            ctx:
            SymFunCtx:
            coordinates: Coordinate tensor to accumulate gradients

        Returns:

        """
        ctx.SymFunCtx = SymFunCtx
        ctx.configuration = configuration
        descriptor_tensor = SymFunCtx.forward(configuration)
        descriptor_tensor = torch.from_numpy(descriptor_tensor)
        descriptor_tensor.requires_grad_(True)
        return descriptor_tensor

    @staticmethod
    def backward(ctx, grad_outputs):
        SymFunCtx = ctx.SymFunCtx
        configuration = ctx.configuration
        dE_dzeta = grad_outputs.numpy()
        dE_dr = SymFunCtx.backward(configuration, dE_dzeta)
        dE_dr = torch.from_numpy(dE_dr)
        dE_dr.requires_grad_(True)
        return None, None, dE_dr


# registered_descriptor_fn = {"SymmetryFunction": TorchSymmetryFunction}

class TorchGraphFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, GraphCtx:KIMTorchGraphGenerator, configuration, coordinate:torch.Tensor):
        ctx.GraphCtx = GraphCtx
        graph:KIMTorchGraph = GraphCtx.generate_graph(configuration)
        outputs = [graph.species, graph.coords]
        for i in range(graph.n_layers):
            outputs.append(graph.__getattr__(f"edge_index{i}"))
        outputs.append(graph.contributions)
        ctx.graph = graph
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        graph = ctx.graph
        images = graph.images
        d_coords = grad_outputs[1]
        d_coords = scatter(d_coords, images, 0)
        return None, None, d_coords


class TrainingWheels(torch.nn.Module):
    @classmethod
    def init_graph(cls, model, cutoff, n_layers):
        kgg = KIMTorchGraphGenerator(cutoff, n_layers,as_torch_geometric_data=True)
        preprocessor = TorchGraphFunction()
        return cls(model, preprocessor, generator_ctx=kgg)

    @classmethod
    def init_descriptor(cls, model, cutoff, descriptor_kind:DescriptorFunctions):
        pass

    def __init__(self, model, preprocessor=None, generator_ctx=None):
        super(TrainingWheels, self).__init__()
        if preprocessor:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = lambda x: x
        self.generator_ctx = generator_ctx
        self.model = model
        self.parameters = model.parameters()

    def forward(self, configuration):
        # coordinate_tensor = torch.from_numpy(self.descriptor_ctx.get_padded_coordinates(configuration))
        coordinate_tensor = torch.from_numpy(configuration.coords)
        coordinate_tensor.requires_grad_(True)
        model_inputs = self.preprocessor.apply(self.generator_ctx, configuration, coordinate_tensor)
        energy = self.model(*model_inputs)
        energy = energy.sum()
        forces, = torch.autograd.grad([energy], [coordinate_tensor], retain_graph=True, allow_unused=True)
        return {"energy": energy, "forces": -forces, "stress": None}
