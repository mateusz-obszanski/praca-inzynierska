from typing import Any, Literal, Union, overload
from pathlib import Path
import yaml
import json
import pickle

import pandas as pd
from libs.schemas.exp_funcs_map import ExperimentType
from libs.schemas.experiment_base import ExperimentBase
from libs.schemas.experiment_irp import ExperimentConfigIRP
from libs.schemas.experiment_tsp import ExperimentConfigTSP
from libs.schemas.experiment_vrp import ExperimentConfigVRP
from libs.schemas.experiment_vrpp import ExperimentConfigVRPP
from libs.schemas import (
    ExperimentIRPSchema,
    ExperimentTSPSchema,
    ExperimentVRPPSchema,
    ExperimentVRPSchema,
)

from libs.utils.iteration import iterate_dataclass


_FILE_READER = {
    "yaml": yaml.full_load,
    "json": json.load,
    "pkl": pickle.load,
    "pickle": pickle.load,
    "csv": pd.read_csv,
}

_READ_IN_BYTE_MODE = {"pickle": True, "pkl": True}


def load_data(path: Union[str, Path]) -> Any:
    path = Path(path)
    extension = path.suffix[1:]
    reader = _FILE_READER[extension]
    read_mode = f"r{'b' if _READ_IN_BYTE_MODE.get(extension, False) else ''}"
    with path.open(read_mode) as f:
        return reader(f)


def load_from_dataclass_paths(instance) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for name, path, ftype in iterate_dataclass(instance):
        if not issubclass(ftype, (Path, str)):
            continue
        loaded[name] = load_data(path)
    return loaded


@overload
def get_experiment_config(
    exp_t: Literal[ExperimentType.TSP], path: Union[str, Path]
) -> ExperimentConfigTSP:
    ...


@overload
def get_experiment_config(
    exp_t: Literal[ExperimentType.VRP], path: Union[str, Path]
) -> ExperimentConfigVRP:
    ...


@overload
def get_experiment_config(
    exp_t: Literal[ExperimentType.VRPP], path: Union[str, Path]
) -> ExperimentConfigVRPP:
    ...


@overload
def get_experiment_config(
    exp_t: Literal[ExperimentType.IRP], path: Union[str, Path]
) -> ExperimentConfigIRP:
    ...


def get_experiment_config(
    exp_t: ExperimentType, path: Union[str, Path]
) -> ExperimentBase:
    schema_map = {
        ExperimentType.TSP: ExperimentTSPSchema,
        ExperimentType.VRP: ExperimentVRPSchema,
        ExperimentType.VRPP: ExperimentVRPPSchema,
        ExperimentType.IRP: ExperimentIRPSchema,
    }
    exp_schema = schema_map[exp_t]()
    with Path(path).open("r") as f:
        exp_data_to_validate = yaml.full_load(f)
    exp_config_data: ExperimentBase = exp_schema.load(exp_data_to_validate)
    return exp_config_data


def configure_experiment_data(exp_conf: ExperimentConfigTSP) -> dict[str, Any]:
    data = load_from_dataclass_paths(exp_conf)
    data.update(
        (name, val) for name, val, _ in iterate_dataclass(exp_conf) if name not in data
    )
    return data
