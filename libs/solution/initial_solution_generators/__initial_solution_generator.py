from typing import Protocol
from . import SolutionRepresentation, SolutionRepresentationTSP, Environment, EnvironmentTSPSimple


class SolutionCreator(Protocol):
    @staticmethod
    def create(environment: Environment, *args, **kwargs) -> SolutionRepresentation:
        ...


class SolutionCreatorTSP(SolutionCreator):
    ...


class SolutionCreatorTSPSimple:
    @staticmethod
    def create(environment: EnvironmentTSPSimple, initial_vx: int) -> SolutionRepresentationTSP:
        ...
