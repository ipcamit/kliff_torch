import torch
from kliff_torch.descriptors_new.descriptor_module import SymmetryFunction

class TorchSymmetryFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, SymFunCtx, configuration, coordinates):
        """
        Coordinates tensor needs to be passed separately .
        Args:
            ctx:
            SymFunCtx:
            coordinates: Coordintate tensor to accumulate gradients

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


registered_descriptor_fn = {"SymmetryFunction": TorchSymmetryFunction}


class TrainingWheels(torch.nn.Module):
    def __init__(self, descriptor_ctx, model):
        super(TrainingWheels, self).__init__()
        self.descriptor_fn = registered_descriptor_fn[type(descriptor_ctx).__name__]
        self.model = model
        self.parameters = model.parameters()
        self.descriptor_ctx = descriptor_ctx
        self.model.descriptor = type(descriptor_ctx).__name__


    def forward(self, configuration):
        # coordinate_tensor = torch.from_numpy(self.descriptor_ctx.get_padded_coordinates(configuration))
        coordinate_tensor = torch.from_numpy(configuration.coords)
        coordinate_tensor.requires_grad_(True)
        descriptor = self.descriptor_fn.apply(self.descriptor_ctx, configuration, coordinate_tensor)
        # descriptor = torch.from_numpy(descriptor)
        # descriptor.requires_grad_(True)
        energy = self.model(descriptor)
        energy = energy.sum()
        forces, = torch.autograd.grad([energy], [coordinate_tensor], retain_graph=True, allow_unused=True)
        return {"energy": energy, "forces": -forces, "stress": None}
