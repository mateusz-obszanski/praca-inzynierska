from typing import Protocol, Generator, Any

from . import SolutionRepresentation
from . import Environment


CostT = float


class CostFunctor(Protocol):
    @staticmethod
    def calculate(solution: SolutionRepresentation, environment: Environment) -> Generator[CostT, Any, Any]:
        ...
