from collections import OrderedDict
from typing import List, Dict

import kliff_torch.libdescriptor.libdescriptor as lds
from kliff_torch.neighbor import NeighborList
from kliff_torch.dataset import Configuration
import numpy as np


class AvailableDescriptors:
    def __init__(self):
        i = 0
        while True:
            desc_type = lds.AvailableDescriptors(i)
            i += 1
            if desc_type.name == "???":
                break
            else:
                setattr(self, desc_type.name, desc_type)


class Descriptor:
    @staticmethod
    def show_available_descriptors():
        print("--------------------------------------------------------------------------------------------------")
        print("Descriptors below are currently available, select them as AvailableDescriptors.<NAME SHOWN BELOW>:")
        print("--------------------------------------------------------------------------------------------------")
        for key in AvailableDescriptors.__dict__.keys():
            print(f"{key}")

    def __init__(self, cutoff: float, species: List[str], descriptor_kind: AvailableDescriptors, hyperparameters: Dict,
                 cutoff_function: str = "cos", nl_ctx:NeighborList=None):
        self.cutoff = cutoff
        self.species = species
        self.descriptor_kind = descriptor_kind
        self.width = -1
        self.hyperparameters = self._set_hyperparams(hyperparameters)
        self.cutoff_function = cutoff_function
        self._cdesc, self.width = self._init_descriptor_from_kind()
        if nl_ctx:
            self.nl_ctx = nl_ctx
            self.external_nl_ctx = True
        else:
            self.external_nl_ctx = False

    def _set_hyperparams(self, hyperparameters):
        if isinstance(hyperparameters, str):
            if hyperparameters == "set51":
                return get_set51()
            elif hyperparameters == "set30":
                return get_set30()
            else:
                raise ValueError("Hyperparameter set not found")
        elif isinstance(hyperparameters, OrderedDict):
            return hyperparameters
        else:
            raise TypeError("Hyperparameters must be either a string or an OrderedDict")

    def _init_descriptor_from_kind(self):
        if self.descriptor_kind == AvailableDescriptors.SymmetryFunctions:
            cutoff_array = np.ones((len(self.species), len(self.species))) * self.cutoff
            symmetry_function_types = list(self.hyperparameters.keys())
            symmetry_function_sizes = []

            symmetry_function_param_matrices = []
            param_num_elem = 0
            width = 0
            for function in symmetry_function_types:
                if function.lower() not in ["g1", "g2", "g3", "g4", "g5"]:
                    ValueError("Symmetry Function provided, not supported")

                if function.lower() == "g1":
                    rows = 1
                    cols = 1
                    params_mat = np.zeros((1, 1), dtype=np.double)
                else:
                    params = self.hyperparameters[function]
                    rows = len(params)
                    cols = len(list(params[0].keys()))
                    params_mat = np.zeros((rows, cols), dtype=np.double)

                    for i in range(rows):
                        if function.lower() == "g2":
                            params_mat[i, 0] = params[i]["eta"]
                            params_mat[i, 1] = params[i]["Rs"]
                        elif function.lower() == "g3":
                            params_mat[i, 0] = params[i]["kappa"]
                        elif function.lower() == "g4":
                            params_mat[i, 0] = params[i]["zeta"]
                            params_mat[i, 1] = params[i]["lambda"]
                            params_mat[i, 2] = params[i]["eta"]
                        elif function.lower() == "g5":
                            params_mat[i, 0] = params[i]["zeta"]
                            params_mat[i, 1] = params[i]["lambda"]
                            params_mat[i, 2] = params[i]["eta"]
                symmetry_function_sizes.extend([rows, cols])
                symmetry_function_param_matrices.append(params_mat)
                param_num_elem += rows + cols
                width += rows

            symmetry_function_param = np.zeros((param_num_elem,), dtype=np.double)
            k = 0
            for i in range(len(symmetry_function_types)):
                symmetry_function_param[k: k + symmetry_function_sizes[2 * i] * symmetry_function_sizes[2 * i + 1]] = \
                    symmetry_function_param_matrices[i].reshape(1, -1)
                k += symmetry_function_sizes[2 * i] * symmetry_function_sizes[2 * i + 1]

            return lds.DescriptorKind(self.desciptor_kind, self.species, self.cutoff_function, cutoff_array,
                                      symmetry_function_types, symmetry_function_sizes, symmetry_function_param), width
        elif self.desciptor_kind == AvailableDescriptors.Bispectrum:
            raise ValueError("Descriptor kind not supported yet")
        else:
            raise ValueError("Descriptor kind not supported yet")

    def _map_species_to_int(self, species):
        return [self.species.index(s) for s in species]

    def forward(self, configuration: Configuration):
        if not self.external_nl_ctx:
            self.nl_ctx = NeighborList(configuration, self.cutoff)
        n_atoms = configuration.get_num_atoms()
        descriptors = np.zeros((n_atoms, self.width))
        species = np.array(self._map_species_to_int(self.nl_ctx.species),np.intc)
    
        for i in range(n_atoms):
            neigh_list, _, _ = self.nl_ctx.get_neigh(i)
            # TODO Implement and use compute function for faster evaluation
            descriptors[i, :] = lds.compute_single_atom(self._cdesc, i, species, np.array(neigh_list, dtype=np.intc), self.nl_ctx.coords)
        return descriptors

    def backward(self, configuration: Configuration, dE_dZeta: np.ndarray):
        if not self.external_nl_ctx:
            self.nl_ctx = NeighborList(configuration, self.cutoff)
        n_atoms = configuration.get_num_atoms()
        derivatives_unrolled = np.zeros(self.nl_ctx.coords.shape)
        species = np.array(self._map_species_to_int(self.nl_ctx.species), dtype=np.intc)

        descriptor = np.zeros(self.width)

        for i in range(n_atoms):
            neigh_list, _, _ = self.nl_ctx.get_neigh(i)
            descriptors_derivative = lds.gradient_single_atom(self._cdesc, i, species, np.array(neigh_list, dtype=np.intc), self.nl_ctx.coords,descriptor, dE_dZeta[i,:])
            derivatives_unrolled += descriptors_derivative.reshape(-1, 3)

        derivatives = np.zeros(configuration.coords.shape)
        neigh_images = self.nl_ctx.get_image()
        for i, atom in enumerate(neigh_images):
            derivatives[atom,:] += derivatives_unrolled[i,:]

        return derivatives


def get_set51():
    r"""Hyperparameters for symmetry functions, as discussed in:
    Nongnuch Artrith and Jorg Behler. "High-dimensional neural network potentials for
    metal surfaces: A prototype study for copper." Physical Review B 85, no. 4 (2012):
    045439.
    """
    return OrderedDict([('g2',
                         [{'eta': 0.0035710676725828126, 'Rs': 0.0},
                          {'eta': 0.03571067672582813, 'Rs': 0.0},
                          {'eta': 0.07142135345165626, 'Rs': 0.0},
                          {'eta': 0.12498736854039845, 'Rs': 0.0},
                          {'eta': 0.21426406035496876, 'Rs': 0.0},
                          {'eta': 0.3571067672582813, 'Rs': 0.0},
                          {'eta': 0.7142135345165626, 'Rs': 0.0},
                          {'eta': 1.428427069033125, 'Rs': 0.0}]),
                        ('g4',
                         [{'zeta': 1, 'lambda': -1, 'eta': 0.00035710676725828126},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.00035710676725828126},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.00035710676725828126},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.00035710676725828126},
                          {'zeta': 1, 'lambda': -1, 'eta': 0.010713203017748437},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.010713203017748437},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.010713203017748437},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.010713203017748437},
                          {'zeta': 1, 'lambda': -1, 'eta': 0.0285685413806625},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.0285685413806625},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.0285685413806625},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.0285685413806625},
                          {'zeta': 1, 'lambda': -1, 'eta': 0.05356601508874219},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.05356601508874219},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.05356601508874219},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.05356601508874219},
                          {'zeta': 4, 'lambda': -1, 'eta': 0.05356601508874219},
                          {'zeta': 4, 'lambda': 1, 'eta': 0.05356601508874219},
                          {'zeta': 16, 'lambda': -1, 'eta': 0.05356601508874219},
                          {'zeta': 16, 'lambda': 1, 'eta': 0.05356601508874219},
                          {'zeta': 1, 'lambda': -1, 'eta': 0.08927669181457032},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.08927669181457032},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.08927669181457032},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.08927669181457032},
                          {'zeta': 4, 'lambda': -1, 'eta': 0.08927669181457032},
                          {'zeta': 4, 'lambda': 1, 'eta': 0.08927669181457032},
                          {'zeta': 16, 'lambda': -1, 'eta': 0.08927669181457032},
                          {'zeta': 16, 'lambda': 1, 'eta': 0.08927669181457032},
                          {'zeta': 1, 'lambda': -1, 'eta': 0.16069804526622655},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.16069804526622655},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.16069804526622655},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.16069804526622655},
                          {'zeta': 4, 'lambda': -1, 'eta': 0.16069804526622655},
                          {'zeta': 4, 'lambda': 1, 'eta': 0.16069804526622655},
                          {'zeta': 16,'lambda': -1, 'eta': 0.16069804526622655},
                          {'zeta': 16, 'lambda': 1, 'eta': 0.16069804526622655},
                          {'zeta': 1, 'lambda': -1, 'eta': 0.28568541380662504},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.28568541380662504},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.28568541380662504},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.28568541380662504},
                          {'zeta': 4, 'lambda': -1, 'eta': 0.28568541380662504},
                          {'zeta': 4, 'lambda': 1, 'eta': 0.28568541380662504},
                          {'zeta': 16, 'lambda': 1, 'eta': 0.28568541380662504}])])


def get_set30():
    r"""Hyperparameters for symmetry functions, as discussed in:
    Artrith, N., Hiller, B. and Behler, J., 2013. Neural network potentials for metals and
    oxides–First applications to copper clusters at zinc oxide. physica status solidi (b),
    250(6), pp.1191-1203.
    """
    return OrderedDict([('g2',
                         [{'eta': 0.003213960905324531, 'Rs': 0.0},
                          {'eta': 0.03571067672582813, 'Rs': 0.0},
                          {'eta': 0.07142135345165626, 'Rs': 0.0},
                          {'eta': 0.12498736854039845, 'Rs': 0.0},
                          {'eta': 0.21426406035496876, 'Rs': 0.0},
                          {'eta': 0.3571067672582813, 'Rs': 0.0},
                          {'eta': 0.7142135345165626, 'Rs': 0.0},
                          {'eta': 1.428427069033125, 'Rs': 0.0}]),
                        ('g4',
                         [{'zeta': 1, 'lambda': -1, 'eta': 0.00035710676725828126},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.00035710676725828126},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.00035710676725828126},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.00035710676725828126},
                          {'zeta': 1, 'lambda': -1, 'eta': 0.010713203017748437},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.010713203017748437},
                          {'zeta': 2, 'lambda': -1, 'eta': 0.010713203017748437},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.010713203017748437},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.0285685413806625},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.0285685413806625},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.05356601508874219},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.05356601508874219},
                          {'zeta': 4, 'lambda': 1, 'eta': 0.05356601508874219},
                          {'zeta': 16, 'lambda': 1, 'eta': 0.05356601508874219},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.08927669181457032},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.08927669181457032},
                          {'zeta': 4, 'lambda': 1, 'eta': 0.08927669181457032},
                          {'zeta': 16, 'lambda': 1, 'eta': 0.08927669181457032},
                          {'zeta': 1, 'lambda': 1, 'eta': 0.16069804526622655},
                          {'zeta': 2, 'lambda': 1, 'eta': 0.16069804526622655},
                          {'zeta': 4, 'lambda': 1, 'eta': 0.16069804526622655},
                          {'zeta': 16, 'lambda': 1, 'eta': 0.16069804526622655}])])


