from typing import Any
from dataclasses import dataclass, field

from marshmallow import pre_load, post_load

from libs.schemas.exp_funcs_map import EXP_ALLOWED_FUNCS, ExperimentType

from .experiment_base import ExperimentBase
from .exceptions import LoadCallablesError


@dataclass
class ExperimentConfigTSP(ExperimentBase):
    """
    Values or paths to pregenerated data
    """

    @pre_load
    def pre_load_and_validate(self, data: dict[str, Any], **_) -> dict[str, Any]:
        data = super().pre_load_and_validate(data)
        return data

    @post_load
    def parse_paths(self, data: dict[str, Any], **_) -> dict[str, Any]:
        data = super().parse_paths(data)
        data = super().load_map_data(data)
        data = super().load_initial_population(data)
        data = self.load_callables(data)
        return data

    def load_callables(self, data: dict[str, Any]) -> dict[str, Any]:
        allowed = EXP_ALLOWED_FUNCS[ExperimentType.TSP]
        keys = ("mutators", "crossovers", "cost_calcs", "fixers")
        callables = {k: [c for c in allowed[k] if c.__name__ in data[k]] for k in keys}
        if any(not cs or len(data[k]) != len(cs) for k, cs in callables.items()):
            raise LoadCallablesError(
                "some callables were not correctly recognized in config file data"
            )
        data.update(callables)
        return data
