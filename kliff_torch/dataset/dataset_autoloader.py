import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset as TorchDataset

import colabfit.tools.configuration
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


class AutoloaderConfiguration:
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

        If colabfit-tools is identified as the source of the configuration data then the configuration
        getter and setter functions would a shallow wrapper over mongodb datacalls.

    """

    def __init__(
        self,
        cell: np.ndarray = None,
        species: List[str] = None,
        coords: np.ndarray = None,
        PBC: List[bool] = None,
        energy: float = None,
        forces: Optional[np.ndarray] = None,
        stress: Optional[List[float]] = None,
        weight: float = 1.0,
        identifier: Optional[Union[str, Path]] = None,
        is_colabfit_dataset: Optional[bool] = None,
        database_client: Optional[MongoDatabase] = None,
        property_id: Optional[str] = None,
        configuration_id: Optional[str] = None,
        aux_property_fields: List[str] = None,
        dynamic_load=False
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
        self.is_colabfit_dataset = is_colabfit_dataset
        self.colabfit_dataclient = database_client
        self.property_id = property_id
        self.configuration_id = configuration_id
        self.aux_property_fields = aux_property_fields
        self._is_initialized = False
        self.descriptor = None
        self.dynamic_load = dynamic_load

        if self.aux_property_fields:
            self._set_aux_properties()

        if dynamic_load and is_colabfit_dataset:
            self._load_at_once()


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
        cls,
        database_client: MongoDatabase,
        configuration_id: str,
        property_ids: str,
        aux_property_fields: List[str] = None,
        dynamic_load = False
    ):
        """
        Read configuration from colabfit database .

        Args:
            database_client: Instance of connected MongoDatabase client, which can be used to
            fetch database from colabfit-tools dataset.
            configuration_id: ID of the configuration instance to be collected from the collection
            "configuration" in colabfit-tools.
            property_ids: ID of the property instance to be associated with current configuration.
            Usually properties would be trained against. Each associated property "field" will be
            matched against provided list of aux_property_fields.
            aux_property_fields: associated colabfit-tools property fields to be associated.
             Default is energy, forces, and stress but more can be added. Provided property field will be
             available under the "property" field.
        """
        self = cls(
            is_colabfit_dataset=True,
            database_client=database_client,
            configuration_id=configuration_id,
            property_id=property_ids,
            aux_property_fields=aux_property_fields,
            dynamic_load=dynamic_load
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
        if not self.is_colabfit_dataset:
            # For normal configuration data is already there
            return self._cell
        else:
            # If it is colabfit dataset then initially it only contains ids with `_is_initialized` flag = False
            # if it is already instantiated then return values else call instantiate method first
            if self._is_initialized:
                return self._cell
            else:
                self._initialize_from_colabfit()
                return self._cell

    @property
    def PBC(self) -> List[bool]:
        """
        A list with 3 components indicating whether periodic boundary condition
        is used along the directions of the first, second, and third lattice vectors.
        """
        if not self.is_colabfit_dataset:
            return self._PBC
        else:
            if self._is_initialized:
                return self._PBC
            else:
                self._initialize_from_colabfit()
                return self._PBC

    @property
    def species(self) -> List[str]:
        """
        Species string of all atoms.
        """
        if not self.is_colabfit_dataset:
            return self._species
        else:
            if self._is_initialized:
                return self._species
            else:
                self._initialize_from_colabfit()
                return self._species

    @property
    def coords(self) -> np.ndarray:
        """
        A Nx3 matrix of the Cartesian coordinates of all atoms.
        """
        if not self.is_colabfit_dataset:
            return self._coords
        else:
            if self._is_initialized:
                return self._coords
            else:
                self._initialize_from_colabfit()
                return self._coords

    @property
    def energy(self) -> Union[float, None]:
        """
        Potential energy of the configuration.
        """
        if self._energy is None:
            # check if it colabfit-datasbase and hence needs initialization
            if self.is_colabfit_dataset:
                energy = self.get_colabfit_property("energy")
                self._energy = energy
                return self._energy
            raise ConfigurationError("Configuration does not contain energy.")
        return self._energy

    @property
    def forces(self) -> np.ndarray:
        """
        Return a `Nx3` matrix of the forces on each atoms.
        """
        if self._forces is None:
            # check if it colabfit-datasbase and hence needs initialization
            if self.is_colabfit_dataset:
                forces = self.get_colabfit_property("forces")
                if forces is not None:
                    self._forces = np.array(forces)
                    return self._forces
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
        # TODO avoid reloading the property if it is explicitly set to None
        if self._stress is None:
            # check if it colabfit-datasbase and hence needs initialization
            if self.is_colabfit_dataset:
                stress = self.get_colabfit_property("stress")
                if stress is not None:
                    self._stress = stress
                    return self._stress
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

    def _initialize_from_colabfit(self):
        """
        Time to fill up the data. To minimize Mongo calls, entire dataset is initialized at once.
        Returns:

        """
        try:
            fetched_configuration: colabfit.tools.configuration.Configuration = (
                self.colabfit_dataclient.get_configuration(self.configuration_id)
            )
        except:
            raise ConnectionError(
                "Looks like Mongo database did not return appropriate response. "
                f"Please run db.configurations.find('_id':{self.configuration_id}) to verify response. "
                f"Or try running the following in separate Python terminal:\n",
                "from colabfit.tools.database import MongoDatabase\n"
                f"client = MongoDatabase({self.colabfit_dataclient.database_name})\n"
                f"client.get_configuration({self.configuration_id})\n"
                " \n"
                "Above shall return a Configuration object with ASE Atoms format.",
            )
        # print("loading coordinates")
        self._coords = fetched_configuration.arrays["positions"]
        self._species = fetched_configuration.get_chemical_symbols()
        self._cell = np.array(fetched_configuration.cell.todict()["array"])
        self._PBC = fetched_configuration.pbc
        self._is_initialized = True

    def _set_aux_properties(self):
        """
        Setup any extra property to be read from property definitions, specially in case of colabfit-tools.
        This routine shall set attributes of the class based on `aux_property_fields`. The fields will be accessed
        by first finding the property type using `type` field in `client.aggregate_property_info`, then suffixing
        the aux_property_fields for field query `<type>.<aux_property_fields>`. By default three properties are defined,
        Energy, forces and stress. This subroutine adds the attributes to the class. As these properties
        will be accessed outside directly and will be determined at runtime, they will be public member.
        Also as of now aux attributes get initialized with the initialization of the Configuration itself.
        Might change this behaviour in future
        Returns: None

        """
        for new_property in self.aux_property_fields:
            property_value = self.get_colabfit_property(new_property)
            if property_value is not None:
                self.__setattr__(new_property, property_value)
            else:
                raise ConfigurationError(
                    f"Configuration does not contain {new_property}."
                )

    def get_colabfit_property(self, property_name: str):
        """
        Returns colabfit-property. workaround till we get proper working get_property routine
        Args:
            property_name: name of the property to fetch

        Returns: fetched value, None if query comes empty

        """
        # print(f"loading {property_name}")
        property_info = self.colabfit_dataclient.aggregate_property_info(
            self.property_id
        )
        property_value = self.colabfit_dataclient.get_data(
            "properties",
            fields=[f"{property_info['types'][0]}.{property_name}"],
            query={"_id": self.property_id},
        )
        if property_value:
            # Temporary workaround against https://github.com/colabfit/colabfit-tools/issues/9
            try:
                property_value = property_value[0].item()
            except AttributeError:
                property_value = property_value[0]
            except ValueError:
                property_value = property_value[0]
            return property_value
        else:
            return None

    def _load_at_once(self):
        """
        This method initializes and loads the properties during object creation only.
        Can be disabled by setting dynamic_load flag to True.
        Returns:
        #TODO: more performant way of initializing from colabfit
        """
        self._initialize_from_colabfit()
        _ = self.energy
        _ = self.forces


    # TODO set up `getattribute` member to raise ConfigurationError for Nonetype properties
    # Currently it was getting in infinite loop

    # def __getattr__(self, name):
    #     """
    #     If none of the above attributes are matched then most likely the property attribute is from `property_list`.
    #     In that case initialize them here. If attribute is not found in `property_list` then it raises
    #     AttributeError
    #     Args:
    #         name: name of the attribute
    #
    #     Returns: return the initialized value of the attribute
    #
    #     """
    #     if hasattr(self, name):
    #         attr_val = self.__getattr__(name)
    #         if attr_val is None:
    #             property_info = self.colabfit_dataclient.aggregate_property_info(self.property_id)
    #             property_value = self.colabfit_dataclient.get_data("properties",
    #                                                                fields=[f"{property_info['types'][0]}.{name}"],
    #                                                                query={"_id": self.property_id})
    #             if property_value:
    #                 self.__setattr__(name, property_value[0])
    #             else:
    #                 raise ConfigurationError(f"Configuration does not contain {name}.")
    #         else:
    #             return attr_val
    #     else:
    #         raise AttributeError(f"Configuration does not contain property: {name}")


class DatasetAutoloader(TorchDataset):
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
        descriptor=None,
    ):
        self.file_format = file_format
        self.descriptor = descriptor
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
            self.configs: List[AutoloaderConfiguration] = []

        self.train_on = None

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

    def get_configs(self) -> List[AutoloaderConfiguration]:
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

        configs = [AutoloaderConfiguration.from_file(f, file_format) for f in all_files]

        if len(configs) <= 0:
            raise DatasetError(
                f"No dataset file with file format `{file_format}` found at {parent}."
            )

        logger.info(f"{len(configs)} configurations read from {path}")

        return configs

    @staticmethod
    def _read_colabfit(
        client: MongoDatabase, dataset_name: str, aux_property_fields: str = None
    ):
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

        colabfit_dataset = client.get_dataset(dataset_id_query[0])
        configuration_ids = []
        for cs in colabfit_dataset["dataset"].configuration_set_ids:
            cos = client.get_configuration_set(cs)["configuration_set"].configuration_ids
            configuration_ids.extend(cos)
        property_ids = colabfit_dataset["dataset"].property_ids

        if len(configuration_ids) != len(property_ids):
            raise ConfigurationError("Number of Configurations and Properties do no match")

        configs = [
            AutoloaderConfiguration.from_colabfit(
                client, ids[0], ids[1], aux_property_fields=aux_property_fields
            )
            for ids in zip(configuration_ids, property_ids)
        ]

        if len(configs) <= 0:
            raise DatasetError(f"No dataset file with in {dataset_name} dataset.")

        logger.info(f"{len(configs)} configurations read from {dataset_name}")

        return configs

    def __len__(self) -> int:
        return len(self.configs)

    def __getitem__(self, idx):
        if self.descriptor and not self.configs[idx].descriptor:
            self.configs[idx].descriptor = self.descriptor.transform(self.configs[idx])
        else:
            return self.configs[idx]
        # if self.train_on:
        #     # If train_on flag is set then only fetch that property
        #     query_dict = {"configuration": self.configs[idx].coords}
        #     if isinstance(self.train_on, list):
        #         for train_on_property in self.train_on:
        #             query_dict[train_on_property] = self.configs[idx].__getattribute__(
        #                 train_on_property
        #             )
        #     elif isinstance(self.train_on, str):
        #         query_dict[self.train_on] = self.configs[idx].__getattribute__(
        #             self.train_on
        #         )
        #     else:
        #         raise ValueError(
        #             f"Expected type `str` or `list` of properties to train on, got {type(self.train_on)}"
        #         )
        #     return query_dict
        # else:
        #     # Else return energy and forces
        #     return {
        #         "configuration": self.configs[idx].coords,
        #         "energy": self.configs[idx].energy,
        #         "forces": self.configs[idx].forces,
        #     }


class ConfigurationError(Exception):
    def __init__(self, msg):
        super(ConfigurationError, self).__init__(msg)
        self.msg = msg


class DatasetError(Exception):
    def __init__(self, msg):
        super(DatasetError, self).__init__(msg)
        self.msg = msg


# TODO: Test for arbitrary properties using aux_property_fields
# TODO: Stress handling
# TODO: Automate property lookup from kim-property
# TODO: Documentation and cleanup
