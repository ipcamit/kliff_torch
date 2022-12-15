import torch
from enum import Enum
from kliff_torch.neighbor import KIMTorchGraphGenerator, KIMTorchGraph
from torch_scatter import scatter
from kliff_torch.descriptors import Descriptor
from kliff_torch.libdescriptor import Descriptor as LibDescriptor
import os

# Todo: link it with AvailableDescriptors in libdescriptor
class DescriptorFunctions(Enum):
    SymmetryFunctions = 1
    Bispectrum = 2


class KIMTorchDescriptorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, DescFunCtx, configuration, coordinates):
        """
        Coordinates tensor needs to be passed separately .
        Args:
            ctx:
            DescFunCtx:
            coordinates: Coordinate tensor to accumulate gradients

        Returns:

        """
        ctx.DescFunCtx = DescFunCtx
        ctx.configuration = configuration
        descriptor_tensor = DescFunCtx.forward(configuration)
        descriptor_tensor = torch.from_numpy(descriptor_tensor)
        descriptor_tensor.requires_grad_(True)
        return descriptor_tensor

    @staticmethod
    def backward(ctx, grad_outputs):
        DescFunCtx = ctx.DescFunCtx
        configuration = ctx.configuration
        dE_dzeta = grad_outputs.numpy()
        dE_dr = DescFunCtx.backward(configuration, dE_dzeta)
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
    def init_graph(cls, model, cutoff, n_layers, elements, model_returns_forces=False):
        kgg = KIMTorchGraphGenerator(elements, cutoff, n_layers,as_torch_geometric_data=True)
        preprocessor = TorchGraphFunction()
        return cls(model, preprocessor, generator_ctx=kgg,
                   model_returns_forces=model_returns_forces)

    @classmethod
    def init_descriptor(cls, model, cutoff, descriptor_kind: DescriptorFunctions, elements, model_returns_forces=False,
                        use_libdescriptor=False, **kwargs):
        if use_libdescriptor:
            descriptor_ctx = LibDescriptor(cutoff, descriptor_kind)
        else:
            descriptor_ctx = Descriptor(cutoff, descriptor_kind, **kwargs)
        preprocessor = KIMTorchDescriptorFunction()
        return cls(model, preprocessor, generator_ctx=descriptor_ctx,
                   model_returns_forces=model_returns_forces)

    def __init__(self, model, preprocessor=None, generator_ctx=None, model_returns_forces=False):
        super(TrainingWheels, self).__init__()
        if preprocessor:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = lambda x: x
        self.generator_ctx = generator_ctx
        self.model = model
        self.parameters = model.parameters()
        self.model_returns_forces = model_returns_forces

    def forward(self, configuration):
        # coordinate_tensor = torch.from_numpy(self.descriptor_ctx.get_padded_coordinates(configuration))
        coordinate_tensor = torch.from_numpy(configuration.coords)
        coordinate_tensor.requires_grad_(True)
        model_inputs = self.preprocessor.apply(self.generator_ctx, configuration, coordinate_tensor)
        if self.model_returns_forces:
            energy, forces = self.model(*model_inputs)
            energy = energy.sum()
        else:
            energy = self.model(*model_inputs)
            energy = energy.sum()
            forces, = torch.autograd.grad([energy], [coordinate_tensor], retain_graph=True, allow_unused=True)
        return {"energy": energy, "forces": -forces, "stress": None}

    def get_parameters(self):
        return self.parameters

    def save_kim_model(self, model_name):
        try:
            os.mkdir(model_name)
        except FileExistsError:
            pass
        model_jit = torch.jit.script(self.model)
        model_jit.save(os.path.join(model_name, "model.pt"))

        if type(self.generator_ctx) == KIMTorchGraphGenerator:
            self.generator_ctx.save_kim_model(model_name, "model.pt")
            file_names = ["kim_model.param", "model.pt"]
        else:
            raise NotImplementedError("Saving KIM model is only supported for graph models.")
        self._write_cmake_file(model_name, file_names)

    @staticmethod
    def _write_cmake_file(model_name, file_names):
        with open(f"{model_name}/CMakeLists.txt", "w") as f:
            f.write("cmake_minimum_required(VERSION 3.10)\n\n")
            f.write("list(APPEND CMAKE_PREFIX_PATH $ENV{KIM_API_CMAKE_PREFIX_DIR})\n")
            f.write("find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)\n")
            f.write('kim_api_items_setup_before_project(ITEM_TYPE "portableModel")\n\n')

            f.write(f"project({model_name})\n\n")
            f.write(f'kim_api_items_setup_after_project(ITEM_TYPE "portableModel")\n')

            f.write('add_kim_api_model_library(\n')
            f.write('  NAME            ${PROJECT_NAME}\n')
            f.write('  DRIVER_NAME     "TorchMLModelDriver"\n')
            f.write('  PARAMETER_FILES  ')
            for file in file_names:
                f.write(f' "{file}" ')
            f.write(')\n')
