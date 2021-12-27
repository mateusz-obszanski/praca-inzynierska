from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass, field
from pathlib import Path
import json

from marshmallow.exceptions import ValidationError
import numpy as np


from .base import BaseDataclass
from .exceptions import MapDataError, PopulationDataError


@dataclass
class ExperimentBase(BaseDataclass, ABC):
    environment_path: str = field(metadata={"is_path": True})
    initial_population_path: str = field(metadata={"is_path": True})
    mutators: list[str]
    crossovers: list[str]
    fixers: list[str]
    cost_calcs: list[str]
    rng_seed: int

    def pre_load_and_validate(self, data, **_) -> dict[str, Any]:
        self.check_numeric_fields(data)
        data = self.load_enums(data)
        self.validate_paths(data)
        return data

    def parse_paths(self, data, **_) -> dict[str, Any]:
        data = self.convert_to_type(data)
        data = self.convert_paths(data)
        return self.load_from_registers(data)

    def load_map_data(self, data: dict[str, Any]) -> dict[str, Any]:
        path: Path = self.environment_path  # type: ignore
        try:
            file_data = self.load_json(path)
        except ValidationError as e:
            raise MapDataError(f"loading JSON data from {path} failed") from e
        try:
            keys = ("dyn_costs", "distance_mx")
            # in file -> in dict
            mapped_keys = (("generation_setup", "map_gen_info"),)
            ndarrays = ("dyn_costs", "distance_mx")
            data.update((k, file_data[k]) for k in keys)
            data.update((mk, file_data[k]) for k, mk in mapped_keys)
            for a in ndarrays:
                data[a] = np.array(data[a])
        except KeyError as e:
            raise MapDataError(f"key `{e.args[0]}` not present in file {path}") from e
        return data

    def load_initial_population(self, data: dict[str, Any]) -> dict[str, Any]:
        path: Path = self.initial_population_path  # type: ignore
        try:
            file_data = self.load_json(path)
        except ValidationError as e:
            raise MapDataError(f"loading JSON data from {path} failed") from e
        try:
            keys = ("population",)
            mapped_keys = ("generation_setup", "pop_gen_info")
            pop_ndarrays = ("vx_seq", "quantities")
            data.update((k, file_data[k]) for k in keys)
            data.update((mk, file_data[k]) for k, mk in mapped_keys)
            for individual in data["population"].values():
                for a in pop_ndarrays:
                    if a not in individual:
                        continue
                    individual[a] = np.array(individual)
        except KeyError as e:
            raise PopulationDataError(
                f"key `{e.args[0]}` not present in file {path}"
            ) from e
        return data

    def load_json(self, path: Path) -> dict[str, Any]:
        try:
            with path.open("r") as f:
                map_data_raw = f.read()
        except FileNotFoundError as e:
            raise ValidationError(f"file {path} does not exist") from e
        try:
            return json.loads(map_data_raw)
        except json.JSONDecodeError as e:
            raise ValidationError(f"JSON parsing error of file {path}") from e

    @abstractmethod
    def load_callables(self, data: dict[str, Any]) -> dict[str, Any]:
        ...
