from typing import Any
from dataclasses import dataclass, field

from marshmallow import pre_load, post_load

from .experiment_base import ExperimentBase


@dataclass
class ExperimentConfigTSP(ExperimentBase):
    """
    Values or paths to pregenerated data
    """

    mutator

    @pre_load
    def pre_load_and_validate(self, data, **_) -> dict[str, Any]:
        ExperimentConfigTSP.check_numeric_fields(data)
        data = ExperimentConfigTSP.load_enums(data)
        ExperimentConfigTSP.validate_paths(data)
        return data

    @post_load
    def parse_paths(self, data, **_) -> dict[str, Any]:
        data = ExperimentConfigTSP.convert_to_type(data)
        data = ExperimentConfigTSP.convert_paths(data)
        return ExperimentConfigTSP.load_from_registers(data)
