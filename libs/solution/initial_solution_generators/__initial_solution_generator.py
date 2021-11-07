from abc import ABC, abstractmethod
from ..representation import SolutionRepresentation, SolutionRepresentationTSP
from ...environment import Environment, EnvironmentTSPSimple


class SolutionCreator(ABC):
    """
    Abstract base class.
    """

    @staticmethod
    @abstractmethod
    def create(environment: Environment, *args, **kwargs) -> SolutionRepresentation:
        ...


class SolutionCreatorTSP(SolutionCreator, ABC):
    """
    Abstract base class.
    """


class SolutionCreatorTSPSimple(ABC):
    """
    Abstract base class.
    """

    @staticmethod
    @abstractmethod
    def create(
        environment: EnvironmentTSPSimple, initial_vx: int
    ) -> SolutionRepresentationTSP:
        ...
