import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from kliff_torch.dataset.extxyz import read_extxyz, write_extxyz
from kliff_torch.utils import to_path
from loguru import logger

# map from file_format to file extension
SUPPORTED_FORMAT = {"xyz": ".xyz"}

try:
    from colabfit.tools.database import MongoDatabase
except ModuleNotFoundError:
    logger.info("colabfit module not found.")


def Z_to_Symbol(Z: int) -> str:
    if Z == 14:
        return "Si"


class Configuration:
    r"""
    Class of atomic configuration.

    This is used to store the information of an atomic configuration, e.g. supercell,
    species, coords, energy, and forces.

    Args:
        cell: A 3x3 matrix of the lattice vectors. The first, second, and third rows are
            :math:`a_1`, :math:`a_2`, and :math:`a_3`, respectively.
        species: A list of N strings giving the species of the atoms, where N is the
            number of atoms.
        coords: A Nx3 matrix of the coordinates of the atoms, where N is the number of
            atoms.
        PBC: A list with 3 components indicating whether periodic boundary condition
            is used along the directions of the first, second, and third lattice vectors.
        energy: energy of the configuration.
        forces: A Nx3 matrix of the forces on atoms, where N is the number of atoms.
        stress: A list with 6 components in Voigt notation, i.e. it returns
            :math:`\sigma=[\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{yz},\sigma_{xz},
            \sigma_{xy}]`. See: https://en.wikipedia.org/wiki/Voigt_notation
        weight: weight of the configuration in the loss function.
        identifier: a (unique) identifier of the configuration

    """

    def __init__(
        self,
        cell: np.ndarray,
        species: List[str],
        coords: np.ndarray,
        PBC: List[bool],
        energy: float = None,
        forces: Optional[np.ndarray] = None,
        stress: Optional[List[float]] = None,
        weight: float = 1.0,
        identifier: Optional[Union[str, Path]] = None,
        is_colabfit_dataset: Optional[bool] = None,
        database_client: Optional[MongoDatabase] = None,
        property_id: Optional[str] = None,
        configuration_id: Optional[str] = None,
    ):
        self._cell = cell
        self._species = species
        self._coords = coords
        self._PBC = PBC
        self._energy = energy
        self._forces = forces
        self._stress = stress
        self._weight = weight
        self._identifier = identifier
        self._path = None
        self._is_colabfit_dataset = is_colabfit_dataset
        self._colabfit_dataset = database_client
        self.property_id = property_id
        self.configuration_id = configuration_id

    # TODO enable config weight read in from file
    @classmethod
    def from_file(cls, filename: Path, file_format: str = "xyz"):
        """
        Read configuration from file.

        Args:
            filename: Path to the file that stores the configuration.
            file_format: Format of the file that stores the configuration (e.g. `xyz`).
        """

        if file_format == "xyz":
            cell, species, coords, PBC, energy, forces, stress = read_extxyz(filename)
        else:
            raise ConfigurationError(
                f"Expect data file_format to be one of {list(SUPPORTED_FORMAT.keys())}, "
                f"got: {file_format}."
            )

        cell = np.asarray(cell)
        species = [str(i) for i in species]
        coords = np.asarray(coords)
        PBC = [bool(i) for i in PBC]
        energy = float(energy) if energy is not None else None
        forces = np.asarray(forces) if forces is not None else None
        stress = [float(i) for i in stress] if stress is not None else None

        self = cls(
            cell, species, coords, PBC, energy, forces, stress, identifier=str(filename)
        )
        self._path = to_path(filename)

        return self

    @classmethod
    def from_colabfit(
        cls, database_client: MongoDatabase, configuration_ids: str, property_ids: str
    ):
        """
        Read configuration from colabfit database .

        Args:
            database_name: Name of the that stores the configuration.
            kim_property: kim property for forces and energy (e.g. `xyz`).
        """
        #
        # TODO forces and energy properties shall only be fetched when needed, else keep value none.
        # this should intended use for pytorch dataloader

        configuration_query = database_client.get_data(
            "configurations",
            fields=["cell", "pbc", "atomic_numbers", "positions"],
            query={"_id": configuration_ids},
        )
        property_query = database_client.get_data(
            "properties",
            fields=["energy-forces.energy", "energy-forces.forces"],
            query={"_id": property_ids},
        )

        cell = configuration_query["cell"][0]
        species = list(map(Z_to_Symbol, configuration_query["atomic_numbers"][0]))
        PBC = [bool(i) for i in configuration_query["pbc"][0]]
        coords = configuration_query["positions"][0]
        stress = None

        PBC = [bool(i) for i in PBC]

        energy = (
            property_query["energy-forces.energy"][0][0]
            if property_query["energy-forces.energy"]
            else None
        )
        forces = (
            property_query["energy-forces.forces"][0]
            if property_query["energy-forces.forces"]
            else None
        )
        # stress = [float(i) for i in stress] if stress is not None else None

        self = cls(
            cell,
            species,
            coords,
            PBC,
            energy,
            forces,
            stress,
            is_colabfit_dataset=True,
            database_client=database_client,
            configuration_id=configuration_ids,
            property_id=property_ids,
        )

        return self

    def to_file(self, filename: Path, file_format: str = "xyz"):
        """
        Write the configuration to file.

        Args:
            filename: Path to the file that stores the configuration.
            file_format: Format of the file that stores the configuration (e.g. `xyz`).
        """
        filename = to_path(filename)
        if file_format == "xyz":
            write_extxyz(
                filename,
                self.cell,
                self.species,
                self.coords,
                self.PBC,
                self._energy,
                self._forces,
                self._stress,
            )
        else:
            raise ConfigurationError(
                f"Expect data file_format to be one of {list(SUPPORTED_FORMAT.keys())}, "
                f"got: {file_format}."
            )

    @property
    def cell(self) -> np.ndarray:
        """
        3x3 matrix of the lattice vectors of the configurations.
        """
        return self._cell

    @property
    def PBC(self) -> List[bool]:
        """
        A list with 3 components indicating whether periodic boundary condition
        is used along the directions of the first, second, and third lattice vectors.
        """
        return self._PBC

    @property
    def species(self) -> List[str]:
        """
        Species string of all atoms.
        """
        return self._species

    @property
    def coords(self) -> np.ndarray:
        """
        A Nx3 matrix of the Cartesian coordinates of all atoms.
        """
        return self._coords

    @property
    def energy(self) -> Union[float, None]:
        """
        Potential energy of the configuration.
        """
        if self._energy is None:
            raise ConfigurationError("Configuration does not contain energy.")
        return self._energy

    @property
    def forces(self) -> np.ndarray:
        """
        Return a `Nx3` matrix of the forces on each atoms.
        """
        if self._forces is None:
            raise ConfigurationError("Configuration does not contain forces.")
        return self._forces

    @property
    def stress(self) -> List[float]:
        r"""
        Stress of the configuration.

        The stress is given in Voigt notation i.e
        :math:`\sigma=[\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{yz},\sigma_{xz},
        \sigma_{xy}]`.

        """
        if self._stress is None:
            raise ConfigurationError("Configuration does not contain stress.")
        return self._stress

    @property
    def weight(self):
        """
        Get the weight of the configuration if the loss function.
        """
        return self._weight

    @weight.setter
    def weight(self, weight: float):
        """
        Set the weight of the configuration if the loss function.
        """
        self._weight = weight

    @property
    def identifier(self) -> str:
        """
        Return identifier of the configuration.
        """
        return self._identifier

    @identifier.setter
    def identifier(self, identifier: str):
        """
        Set the identifier of the configuration.
        """
        self._identifier = identifier

    @property
    def path(self) -> Union[Path, None]:
        """
        Return the path of the file containing the configuration. If the configuration
        is not read from a file, return None.
        """
        return self._path

    def get_num_atoms(self) -> int:
        """
        Return the total number of atoms in the configuration.
        """
        return len(self.species)

    def get_num_atoms_by_species(self) -> Dict[str, int]:
        """
        Return a dictionary of the number of atoms with each species.
        """
        return self.count_atoms_by_species()

    def get_volume(self) -> float:
        """
        Return volume of the configuration.
        """
        return abs(np.dot(np.cross(self.cell[0], self.cell[1]), self.cell[2]))

    def count_atoms_by_species(
        self, symbols: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Count the number of atoms by species.

        Args:
            symbols: species to count the occurrence. If `None`, all species present
                in the configuration are used.

        Returns:
            {specie, count}: with `key` the species string, and `value` the number of
                atoms with each species.
        """

        unique, counts = np.unique(self.species, return_counts=True)
        symbols = unique if symbols is None else symbols

        natoms_by_species = dict()
        for s in symbols:
            if s in unique:
                natoms_by_species[s] = counts[list(unique).index(s)]
            else:
                natoms_by_species[s] = 0

        return natoms_by_species

    def order_by_species(self):
        """
        Order the atoms according to the species such that atoms with the same species
        have contiguous indices.
        """
        if self.forces is not None:
            species, coords, forces = zip(
                *sorted(
                    zip(self.species, self.coords, self.forces),
                    key=lambda pair: pair[0],
                )
            )
            self._species = np.asarray(species).tolist()
            self._coords = np.asarray(coords)
            self._forces = np.asarray(forces)
        else:
            species, coords = zip(
                *sorted(zip(self.species, self.coords), key=lambda pair: pair[0])
            )
            self._species = np.asarray(species)
            self._coords = np.asarray(coords)


class Dataset:
    """
    A dataset of multiple configurations (:class:`~kliff_torch.dataset.Configuration`).

    Args:
        path: Path of a file storing a configuration or filename to a directory containing
            multiple files. If given a directory, all the files in this directory and its
            subdirectories with the extension corresponding to the specified file_format
            will be read.
        file_format: Format of the file that stores the configuration, e.g. `xyz`.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        file_format="xyz",
        colabfit_database=None,
        kim_property=None,
        colabfit_dataset=None,
    ):
        self.file_format = file_format

        if path is not None:
            self.configs = self._read(path, file_format)

        elif colabfit_database is not None:
            if colabfit_dataset is not None:
                # open link to the mongo
                if "colabfit" in sys.modules:
                    self.mongo_client = MongoDatabase(colabfit_database)
                    self.colabfit_dataset = colabfit_dataset
                    self.kim_property = kim_property
                    self.configs = self._read_colabfit(
                        self.mongo_client, colabfit_dataset, kim_property
                    )
                else:
                    logger.error(f"colabfit tools not installed.")
                    raise ModuleNotFoundError(
                        f"You are trying to read configuration from colabfit dataset"
                        " but colabfit-tools module is not installed."
                        " Please do `pip install colabfit` first"
                    )
            else:
                logger.error(
                    f"colabfit database provided ({colabfit_database}) but not dataset."
                )
                raise DatasetError(f"No dataset name given.")

        else:
            self.configs = []

    def add_configs(self, path: Path):
        """
        Read configurations from filename and added them to the existing set of
        configurations.

        This is a convenience function to read configurations from different directory
        on disk.

        Args:
            path: Path the directory (or filename) storing the configurations.
        """

        configs = self._read(path, self.file_format)
        self.configs.extend(configs)

    def get_configs(self) -> List[Configuration]:
        """
        Get the configurations.
        """
        return self.configs

    def get_num_configs(self) -> int:
        """
        Return the number of configurations in the dataset.
        """
        return len(self.configs)

    @staticmethod
    def _read(path: Path, file_format: str = "xyz"):
        """
        Read atomic configurations from path.
        """
        try:
            extension = SUPPORTED_FORMAT[file_format]
        except KeyError:
            raise DatasetError(
                f"Expect data file_format to be one of {list(SUPPORTED_FORMAT.keys())}, "
                f"got: {file_format}."
            )

        path = to_path(path)

        if path.is_dir():
            parent = path
            all_files = []
            for root, dirs, files in os.walk(parent):
                for f in files:
                    if f.endswith(extension):
                        all_files.append(to_path(root).joinpath(f))
            all_files = sorted(all_files)
        else:
            parent = path.parent
            all_files = [path]

        configs = [Configuration.from_file(f, file_format) for f in all_files]

        if len(configs) <= 0:
            raise DatasetError(
                f"No dataset file with file format `{file_format}` found at {parent}."
            )

        logger.info(f"{len(configs)} configurations read from {path}")

        return configs

    @staticmethod
    def _read_colabfit(client: MongoDatabase, dataset_name: str, kim_property: str):
        """
        Read atomic configurations from path.
        """

        # get configuration and property ID and send it to load configuration
        dataset_id_query = client.get_data(
            "datasets", fields=["_id"], query={"name": dataset_name}
        )
        if not dataset_id_query:
            logger.error(f"{dataset_name} is either empty or does not exist")
            raise DatasetError(f"{dataset_name} is either empty or does not exist")

        colabfit_dataset = client.get_dataset(dataset_id_query[0][0])
        configuration_ids, property_ids = (
            colabfit_dataset["dataset"].configuration_set_ids[0],
            colabfit_dataset["dataset"].property_ids,
        )
        configs = [
            Configuration.from_colabfit(client, ids[0], ids[1])
            for ids in zip(configuration_ids, property_ids)
        ]

        if len(configs) <= 0:
            raise DatasetError(
                f"No dataset file with in {dataset_name} dataset."
            )

        logger.info(f"{len(configs)} configurations read from {dataset_name}")

        return configs


class ConfigurationError(Exception):
    def __init__(self, msg):
        super(ConfigurationError, self).__init__(msg)
        self.msg = msg


class DatasetError(Exception):
    def __init__(self, msg):
        super(DatasetError, self).__init__(msg)
        self.msg = msg
