from dataclasses import dataclass, field

from .base import BaseDataclass


@dataclass
class ExperimentBase(BaseDataclass):
    environment_path: str = field(metadata={"is_path": True})
    initial_population_path: str = field(metadata={"is_path": True})
