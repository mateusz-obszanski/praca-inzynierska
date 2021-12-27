from enum import auto, Enum
from typing import Any, Callable, Union
from pathlib import Path
import yaml
import json
import pickle

import pandas as pd
from libs.data_loading.exceptions import FuncNotFoundError

from libs.utils.iteration import iterate_dataclass


class ExperimentType(Enum):
    TSP = auto()
    VRP = auto()
    VRPP = auto()
    IRP = auto()


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


def get_f_by_name(name: str, fs: list[Callable]) -> Callable:
    try:
        return next(f for f in fs if f.__name__ == name)
    except StopIteration:
        raise FuncNotFoundError(name)
