from collections import namedtuple, Counter
from kliff_torch.descriptors_new import descriptors as ds
from kliff_torch.neighbor import NeighborList
import numpy as np

class SymmetryFunction:

    def __init__(self, cut_dists, hyperparams, neighbor_list_builder=None):
        self.cut_dists = cut_dists
        self.cut_matrix = self._get_full_cutoff_matrix()
        self.hyperparams = hyperparams
        self.data_container =  ds.SymmetryFunctionParams()
        self.data_container.set_cutoff("cos",self.cut_matrix)
        self.descriptor_width = None
        self._set_hyperparams()
        if neighbor_list_builder:
            self.nl_builder = neighbor_list_builder
        else:
            self.nl_builder = NeighborList

    def _get_full_cutoff_matrix(self):
        if type(self.cut_dists) == float:
            return np.array([[self.cut_dists]], dtype=np.double)
        elif type(self.cut_dists) == np.ndarray:
            return self.cut_dists.astype(np.double)
        elif type(self.cut_dists) == dict:
            num_elem = 0
            elem_dict = {}
            for keys in self.cut_dists.keys():
                elem1, elem2 = keys.split('-')
                if elem1 == elem2:
                    elem_dict[elem1] = num_elem
                    num_elem += 1
            cut_array = np.ones((num_elem, num_elem), dtype=np.double) *-1.0
            for keys, values in self.cut_dists.items():
                elem1, elem2 = keys.split('-')
                cut_array[elem_dict[elem1], elem_dict[elem2]] = values
            missing_vals = np.where(cut_array == -1)
            for i,j in zip(*missing_vals):
                cut_array[i,j] = cut_array[j,i]
            if -1. in cut_array:
                missing_vals = np.where(cut_array == -1)
                missing_pairs = [f"{elem1}-{elem2}" for elem1, elem2 in zip(*missing_vals)]
                raise ValueError(f"Incomplete Cutoff array\n missing pairs: {missing_pairs}")
            return cut_array.astype(np.double)


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
        infl_dist = np.amax(self.cut_matrix)
        nl = self.nl_builder(configuration, infl_dist)
        descriptors = np.zeros((configuration.get_num_atoms(), self.descriptor_width))
        element_dict = {}
        for i, element in enumerate(Counter(nl.species)):
            element_dict[element] =  i
        species = list(map(lambda x: element_dict[x], nl.species))

        for i in range(configuration.get_num_atoms()):
            neigh_list, _, _ = nl.get_neigh(i)
            descriptors[i,:] = ds.symmetry_function_atomic(
                                i,
                                nl.coords,
                                np.array(species, np.intc),
                                np.array(neigh_list, np.intc),
                                self.descriptor_width,
                                self.data_container)
        return descriptors

    def backward(self, configuration, dE_dZeta):
        infl_dist = np.amax(self.cut_matrix)
        nl = self.nl_builder(configuration, infl_dist)
        derivatives_unrolled = np.zeros(nl.coords.shape)
        element_dict = {}
        for i, element in enumerate(Counter(nl.species)):
            element_dict[element] =  i
        species = list(map(lambda x: element_dict[x], nl.species))

        for i in range(configuration.get_num_atoms()):
            neigh_list, _, _ = nl.get_neigh(i)
            descriptors_derivative =  ds.grad_symmetry_function_atomic(
                                i,
                                nl.coords,
                                np.array(species, np.intc),
                                np.array(neigh_list, np.intc),
                                self.descriptor_width,
                                self.data_container,
                                dE_dZeta[i,:])
            derivatives_unrolled += descriptors_derivative.reshape(-1,3)

        derivatives = np.zeros(configuration.coords.shape)
        neigh_images = nl.get_image()
        for i, atom in enumerate(neigh_images):
            derivatives[atom,:] += derivatives_unrolled[i,:]

        return derivatives

    def get_padded_coordinates(self, configuration):
        infl_dist = np.amax(self.cut_matrix)
        nl = self.nl_builder(configuration, infl_dist)
        return nl.coords


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
    g2[:,0] /= bhor2ang ** 2
    g4[:,2] /= bhor2ang ** 2

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

