from typing import Any
from dataclasses import dataclass, field

from marshmallow import pre_load, post_load

from .base import BaseDataclass


from enum import Enum, auto


class DummyEnum(Enum):
    A = auto()


@dataclass
class ExperimentConfigTSP(BaseDataclass):
    """
    Values or paths to pregenerated data
    """

    # population: Path
    # cost_mx: Path
    maka_from_path: str = field(metadata={"is_path": True})
    dummy_enum: DummyEnum

    @pre_load
    def pre_load_and_validate(self, data, **_) -> dict[str, Any]:
        ExperimentConfigTSP.check_numeric_fields(data)
        data = ExperimentConfigTSP.load_enums(data)
        ExperimentConfigTSP.validate_paths(data)
        return ExperimentConfigTSP.load_from_registers(data)

    @post_load
    def parse_paths(self, data, **_) -> dict[str, Any]:
        data = ExperimentConfigTSP.convert_to_type(data)
        return ExperimentConfigTSP.convert_paths(data)
