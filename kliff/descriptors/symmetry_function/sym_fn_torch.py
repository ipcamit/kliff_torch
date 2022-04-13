import os
import pickle
from collections import OrderedDict, namedtuple

import numpy as np
import torch
from traitlets import Any

from kliff.descriptors.descriptor import (
    Descriptor,
    generate_full_cutoff,
    generate_species_code,
    generate_unique_cutoff_pairs,
)
from kliff.descriptors.symmetry_function import sf  # C extension
from kliff.neighbor import NeighborList
from loguru import logger

class SymmetryFunctionTorch:
    def __init__(self, hyperparam, cutoff=10.0, cut_name="cos", fit_forces=False):
        super(SymmetryFunctionTorch, self).__init__()
        self.hyperpram = hyperparam
        self.cutoff = cutoff
        self.cut_name = cut_name
        self.fit_forces = fit_forces
        self.descriptor_function = sf.Descriptor()

    def _set_cutoff(self):
        supported = ["cos"]
        if self.cut_name is None:
            self.cut_name = supported[0]
        if self.cut_name not in supported:
            spd = ['"{}", '.format(s) for s in supported]
            raise SymmetryFunctionError(
                'Cutoff "{}" not supported by this descriptor. Use {}.'.format(
                    self.cut_name, spd
                )
            )

        self.cutoff = generate_full_cutoff(self.cut_dists)
        self.species_code = generate_species_code(self.cut_dists)
        num_species = len(self.species_code)

        rcutsym = np.zeros([num_species, num_species], dtype=np.double)
        for si, i in self.species_code.items():
            for sj, j in self.species_code.items():
                rcutsym[i][j] = self.cutoff[si + "-" + sj]
        self.descriptor_function.set_cutoff(self.cut_name, rcutsym)

    def _set_hyperparams(self):
        if isinstance(self.hyperparams, str):
            name = self.hyperparams.lower()
            if name == "set51":
                self.hyperparams = get_set51_torch()
            elif name == "set30":
                self.hyperparams = get_set30_torch()
            else:
                raise SymmetryFunctionError(
                    'hyperparams "{}" unrecognized.'.format(name)
                )
        # if not isinstance(self.hyperparams, OrderedDict):
        #     self.hyperparams = OrderedDict(self.hyperparams)

        # hyperparams of descriptors
        for idx, key in enumerate(self.hyperparams._fields):
            if key.lower() not in ["g1", "g2", "g3", "g4", "g5"]:
                raise SymmetryFunctionError(
                    'Symmetry function "{}" unrecognized.'.format(key)
                )

            # g1 needs no hyperparams, put a placeholder
            name = key.lower()
            if name == "g1":
                # it has no hyperparams, zeros([1,1]) for placeholder
                params = np.zeros([1, 1], dtype=np.double)
            else:
                params = self.hyperparams[idx]
            # store cutoff values in both this python and cpp class
            # self._desc[name] = params
            self.descriptor_function.add_descriptor(name, params)

    # def _return_autograd_padded_descriptor(self):
    #     class autograd_padded_descriptor(torch.autograd.Function):
    #         @staticmethod
    #         def forward(ctx,coords,func):
    #



def get_set51_torch():
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

def get_set30_torch():
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


class SymmetryFunctionTransform:
    r"""Atom-centered symmetry functions descriptor as discussed in [Behler2011]_."""
    def __init__(
        self, cut_dists, cut_name, hyperparams, dtype=np.float32, fit_forces=False, fit_stress=False, normalize=False, reuse=False, mean_std_file=None,dataset=None
    ):
        self._desc = OrderedDict()
        self._cdesc = sf.Descriptor()
        self.cut_name = cut_name
        self.cut_dists = cut_dists
        self.hyperparams = hyperparams
        self.dtype = dtype
        self._set_cutoff()
        self._set_hyperparams()
        self.size = self.get_width()
        self.fit_forces = fit_forces
        self.fit_stress = fit_stress
        self.normalize = normalize
        self.mean_std_file = mean_std_file


        logger.debug(f"`{self.__class__.__name__}` descriptor initialized.")

    def transform(self, conf):
        r"""Transform atomic coords to atomic environment descriptor values.
        """

        # create neighbor list
        infl_dist = max(self.cutoff.values())
        nei = NeighborList(conf, infl_dist, padding_need_neigh=False)

        coords = nei.coords
        image = nei.image
        species = np.asarray([self.species_code[i] for i in nei.species], dtype=np.intc)

        Ncontrib = conf.get_num_atoms()
        Ndesc = self.get_width()

        grad = self.fit_forces or self.fit_stress

        zeta_config = []
        dzetadr_forces_config = []
        dzetadr_stress_config = []

        for i in range(Ncontrib):
            neigh_indices, _, _ = nei.get_neigh(i)
            neighlist = np.asarray(neigh_indices, dtype=np.intc)
            zeta, dzetadr = self._cdesc.generate_one_atom(
                i, coords, species, neighlist, grad
            )

            zeta_config.append(zeta)

            if grad:
                # last 3 elements dzetadr is associated with atom i
                atom_ids = np.concatenate((neigh_indices, [i]))
                dzetadr = dzetadr.reshape(Ndesc, -1, 3)

            if self.fit_forces:
                dzetadr_forces = np.zeros((Ndesc, Ncontrib, 3))
                for ii, idx in enumerate(atom_ids):
                    org_idx = image[idx]
                    dzetadr_forces[:, org_idx, :] += dzetadr[:, ii, :]
                dzetadr_forces_config.append(dzetadr_forces.reshape(Ndesc, -1))
                # self.grads = np.asarray(dzetadr_forces_config)

            if self.fit_stress:
                dzetadr_stress = np.zeros((Ndesc, 6))
                for ii, idx in enumerate(atom_ids):
                    dzetadr_stress[:, 0] += dzetadr[:, ii, 0] * coords[idx][0]
                    dzetadr_stress[:, 1] += dzetadr[:, ii, 1] * coords[idx][1]
                    dzetadr_stress[:, 2] += dzetadr[:, ii, 2] * coords[idx][2]
                    dzetadr_stress[:, 3] += dzetadr[:, ii, 1] * coords[idx][2]
                    dzetadr_stress[:, 4] += dzetadr[:, ii, 2] * coords[idx][0]
                    dzetadr_stress[:, 5] += dzetadr[:, ii, 0] * coords[idx][1]
                dzetadr_stress_config.append(dzetadr_stress)

        zeta_config = torch.tensor(zeta_config)
        if self.fit_forces:
            dzetadr_forces_config = torch.tensor(dzetadr_forces_config)
        else:
            dzetadr_forces_config = None
        if self.fit_stress:
            dzetadr_stress_config = torch.tensor(dzetadr_stress_config)
        else:
            dzetadr_stress_config = None

        msg = (
            "=" * 25
            + "descriptor values (no normalization)"
            + "=" * 25
            + f"\nconfiguration name: {conf.identifier}"
            + "\natom id    descriptor values ..."
        )
        logger.debug(msg)

        for i, line in enumerate(zeta_config):
            s = f"\n{i}    "
            for j in line:
                s += f"{j:.15g} "
            logger.debug(s)

        return {"zeta": zeta_config, "dzetadr_forces": dzetadr_forces_config, "dzetadr_stress": dzetadr_stress_config}

    def _set_cutoff(self):
        supported = ["cos"]
        if self.cut_name is None:
            self.cut_name = supported[0]
        if self.cut_name not in supported:
            spd = ['"{}", '.format(s) for s in supported]
            raise SymmetryFunctionError(
                'Cutoff "{}" not supported by this descriptor. Use {}.'.format(
                    self.cut_name, spd
                )
            )

        self.cutoff = generate_full_cutoff(self.cut_dists)
        self.species_code = generate_species_code(self.cut_dists)
        num_species = len(self.species_code)

        rcutsym = np.zeros([num_species, num_species], dtype=np.double)
        for si, i in self.species_code.items():
            for sj, j in self.species_code.items():
                rcutsym[i][j] = self.cutoff[si + "-" + sj]
        self._cdesc.set_cutoff(self.cut_name, rcutsym)

    def _set_hyperparams(self):
        if isinstance(self.hyperparams, str):
            name = self.hyperparams.lower()
            if name == "set51":
                self.hyperparams = get_set51()
            elif name == "set30":
                self.hyperparams = get_set30()
            else:
                raise SymmetryFunctionError(
                    'hyperparams "{}" unrecognized.'.format(name)
                )
        if not isinstance(self.hyperparams, OrderedDict):
            self.hyperparams = OrderedDict(self.hyperparams)

        # hyperparams of descriptors
        for key, values in self.hyperparams.items():
            if key.lower() not in ["g1", "g2", "g3", "g4", "g5"]:
                raise SymmetryFunctionError(
                    'Symmetry function "{}" unrecognized.'.format(key)
                )

            # g1 needs no hyperparams, put a placeholder
            name = key.lower()
            if name == "g1":
                # it has no hyperparams, zeros([1,1]) for placeholder
                params = np.zeros([1, 1], dtype=np.double)
            else:
                rows = len(values)
                cols = len(values[0])
                params = np.zeros([rows, cols], dtype=np.double)
                for i, line in enumerate(values):
                    if name == "g2":
                        params[i][0] = line["eta"]
                        params[i][1] = line["Rs"]
                    elif name == "g3":
                        params[i][0] = line["kappa"]
                    elif key == "g4":
                        params[i][0] = line["zeta"]
                        params[i][1] = line["lambda"]
                        params[i][2] = line["eta"]
                    elif key == "g5":
                        params[i][0] = line["zeta"]
                        params[i][1] = line["lambda"]
                        params[i][2] = line["eta"]

            # store cutoff values in both this python and cpp class
            self._desc[name] = params
            self._cdesc.add_descriptor(name, params)

    def get_width(self):
        N = 0
        for key in self._desc:
            N += len(self._desc[key])
        return N

    def get_hyperparams(self):
        return self._desc

    def _set_mean_std(self):

        if not self.normalize:
            self.mean = 1.0
            self.std = 1.0
        elif self.mean_std_file:
            data = pickle.load(open(self.mean_std_file, "rb"))
            try:
                mean = data["mean"]
                stdev = data["stdev"]
                size = data["size"]
            except Exception as e:
                raise ValueError(f"Corrupted state dict for descriptor: {str(e)}")

            # more checks on data integrity
            if mean is not None and stdev is not None and size is not None:
                if len(mean.shape) != 1 or mean.shape[0] != size:
                    raise ValueError(f"Corrupted descriptor mean.")

                if len(stdev.shape) != 1 or stdev.shape[0] != size:
                    raise ValueError("Corrupted descriptor standard deviation.")

            self.mean = mean
            self.stdev = stdev
            self.size = size


    def _calc_zeta_dzetadr(self, configs, fit_forces, fit_stress, nprocs=1):
        """
        Calculate the fingerprints and maybe its gradients w.r.t the atomic coords.
        """
        if nprocs == 1:
            zeta = []
            dzetadr_forces = []
            dzetadr_stress = []
            for conf in configs:
                z, dzdr_f, dzdr_s = self.transform(conf, fit_forces, fit_stress)
                zeta.append(z)
                dzetadr_forces.append(dzdr_f)
                dzetadr_stress.append(dzdr_s)
        # else:
        #     rslt = parallel.parmap1(
        #         self.transform, configs, fit_forces, fit_stress, nprocs=nprocs
        #     )
        #     zeta = [pair[0] for pair in rslt]
        #     dzetadr_forces = [pair[1] for pair in rslt]
        #     dzetadr_stress = [pair[2] for pair in rslt]

        return zeta, dzetadr_forces, dzetadr_stress

# def SymmetryFunctionDataset:
#     def __init__(self, filename: Path, transform: Optional[Callable] = None):
#         self.fp = load_fingerprints(filename)
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.fp)
#
#     def __getitem__(self, index):
#         sample = self.fp[index]
#         if self.transform:
#             sample = self.transform(sample)
#         return sample
#


class SymmetryFunctionError(Exception):
    def __init__(self, msg):
        super(SymmetryFunctionError, self).__init__(msg)
        self.msg = msg
