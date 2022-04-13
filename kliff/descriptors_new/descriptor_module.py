from collections import namedtuple, Counter
from kliff.descriptors_new import descriptors as ds
from kliff.neighbor import NeighborList
import numpy as np

class SymmetryFunction:

    def __init__(self, cut_dists, hyperparams, neighbor_list_builder=None):
        self.cut_dists = cut_dists
        self.hyperparams = hyperparams
        self.data_container =  ds.SymmetryFunctionParams()
        self.data_container.set_cutoff("cos", np.array([[self.cut_dists]], dtype=np.double))
        self.descriptor_width = None
        self._set_hyperparams()
        if neighbor_list_builder:
            self.nl_builder = neighbor_list_builder
        else:
            self.nl_builder = NeighborList

    def _set_hyperparams(self):
        if isinstance(self.hyperparams, str):
            name = self.hyperparams.lower()
            if name == "set51":
                hparams = get_set51()
                self.descriptor_width = 51
            elif name == "set30":
                hparams = get_set30()
                self.descriptor_width = 30
            else:
                raise ValueError("NO HYP PARAM SET FOUND")

        # hyperparams of descriptors
        for key, values in zip(hparams._fields, hparams):
            if key.lower() not in ["g1", "g2", "g3", "g4", "g5"]:
                raise ValueError("NO SYM FUNCTION FOUND")

            # g1 needs no hyperparams, put a placeholder
            name = key.lower()
            if name == "g1":
                # it has no hyperparams, zeros([1,1]) for placeholder
                params = np.zeros([1, 1], dtype=np.double)
                self.data_container.add_descriptor("g1", np.array(params, dtype=np.double))
            else:
                self.data_container.add_descriptor(name, values)

    def forward(self, configuration):
        nl = self.nl_builder(configuration, self.cut_dists)
        descriptors = np.zeros((configuration.get_num_atoms(), self.descriptor_width))
        element_dict = {}
        for i, element in enumerate(Counter(nl.species)):
            element_dict[element] =  i
        species = list(map(lambda x: element_dict[x], nl.species))

        for i in range(configuration.get_num_atoms()):
            neigh_list, _, _ = nl.get_neigh(i)
            descriptors[i,:] = ds.symmetry_function_atomic(
                                i,
                                configuration.coords,
                                np.array(species, np.intc),
                                np.array(neigh_list, np.intc),
                                self.descriptor_width,
                                self.data_container)
            # print(i, descriptors[i,0:3])
        return descriptors

    def backward(self, configuration, dE_dZeta):
        nl = self.nl_builder(configuration, self.cut_dists)
        derivatives = np.zeros(configuration.coords.shape)
        element_dict = {}
        for i, element in enumerate(Counter(nl.species)):
            element_dict[element] =  i
        species = list(map(lambda x: element_dict[x], nl.species))

        for i in range(configuration.get_num_atoms()):
            neigh_list, _, _ = nl.get_neigh(i)
            descriptors_derivative =  ds.grad_symmetry_function_atomic(
                                i,
                                configuration.coords,
                                np.array(species, np.intc),
                                np.array(neigh_list, np.intc),
                                self.descriptor_width,
                                self.data_container,
                                dE_dZeta[i,:])
            derivatives += descriptors_derivative.reshape(-1,3)
        return derivatives

    def compute_neighbours(configuration):
        pass

def get_set51():
    r"""Hyperparameters for symmetry functions, as discussed in:
    Nongnuch Artrith and Jorg Behler. "High-dimensional neural network potentials for
    metal surfaces: A prototype study for copper." Physical Review B 85, no. 4 (2012):
    045439.
    """
    hyperparameters = namedtuple("hyperparameters", "g2, g4")
    # g2 = [eta Rs], g4 = [zeta, lambda, eta]
    bhor2ang = 0.529177
    g2 = np.array([[0.001,  0.0],
                 [0.01,  0.0],
                 [0.02,  0.0],
                 [0.035,  0.0],
                 [0.06,  0.0],
                 [0.1,  0.0],
                 [0.2,  0.0],
                 [0.4,  0.0]], dtype=np.double)
    g4 = np.array([[1, -1, 0.0001],
                [1, 1, 0.0001],
                [2, -1, 0.0001],
                [2, 1, 0.0001],
                [1, -1, 0.003],
                [1, 1, 0.003],
                [2, -1, 0.003],
                [2, 1, 0.003],
                [1, -1, 0.008],
                [1, 1, 0.008],
                [2, -1, 0.008],
                [2, 1, 0.008],
                [1, -1, 0.015],
                [1, 1, 0.015],
                [2, -1, 0.015],
                [2, 1, 0.015],
                [4, -1, 0.015],
                [4, 1, 0.015],
                [16, -1, 0.015],
                [16, 1, 0.015],
                [1, -1, 0.025],
                [1, 1, 0.025],
                [2, -1, 0.025],
                [2, 1, 0.025],
                [4, -1, 0.025],
                [4, 1, 0.025],
                [16, -1, 0.025],
                [16, 1, 0.025],
                [1, -1, 0.045],
                [1, 1, 0.045],
                [2, -1, 0.045],
                [2, 1, 0.045],
                [4, -1, 0.045],
                [4, 1, 0.045],
                [16, -1, 0.045],
                [16, 1, 0.045],
                [1, -1, 0.08],
                [1, 1, 0.08],
                [2, -1, 0.08],
                [2, 1, 0.08],
                [4, -1, 0.08],
                [4, 1, 0.08],
                [16, 1, 0.08]], dtype=np.double)
    g2[:,0] /= bhor2ang
    g4[:,2] /= bhor2ang

    params = hyperparameters(g2, g4)
    # transfer units from bohr to angstrom
    return params

def get_set30():
    r"""Hyperparameters for symmetry functions, as discussed in:
    Artrith, N., Hiller, B. and Behler, J., 2013. Neural network potentials for metals and
    oxides–First applications to copper clusters at zinc oxide. physica status solidi (b),
    250(6), pp.1191-1203.
    """
    hyperparameters = namedtuple("hyperparameters", "g2, g4")
    # g2 = [eta Rs], g4 = [zeta, lambda, eta]
    bhor2ang = 0.529177

    g2 = np.array([[0.0009, 0.0],
        [0.01, 0.0],
        [0.02, 0.0],
        [0.035, 0.0],
        [0.06, 0.0],
        [0.1, 0.0],
        [0.2, 0.0],
        [0.4, 0.0]],dtype=np.double)

    g4 = np.array([[1,  1, 0.0001],
        [1, 1, 0.0001],
        [2,  1, 0.0001],
        [2, 1, 0.0001],
        [1,  1, 0.003],
        [1, 1, 0.003],
        [2,  1, 0.003],
        [2, 1, 0.003],
        [1, 1, 0.008],
        [2, 1, 0.008],
        [1, 1, 0.015],
        [2, 1, 0.015],
        [4, 1, 0.015],
        [16, 1, 0.015],
        [1, 1, 0.025],
        [2, 1, 0.025],
        [4, 1, 0.025],
        [16, 1, 0.025],
        [1, 1, 0.045],
        [2, 1, 0.045],
        [4, 1, 0.045],
        [16, 1, 0.045]], dtype=np.double)

    # transfer units from bohr to angstrom
    g2[:,0] /= bhor2ang
    g4[:,2] /= bhor2ang

    params = hyperparameters(g2, g4)
    return params

