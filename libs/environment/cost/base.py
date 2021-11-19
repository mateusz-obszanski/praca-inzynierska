from abc import ABC, abstractmethod
from typing import Callable, Generator, Any
from functools import wraps
from collections import deque

from ...environment import Environment
from ...solution.representation import SolutionRepresentation
from ...utils.iteration import exhaust_iterator


CostT = float


class CostCalculator(ABC):
    """
    Abstract base class. Requires overwriting `_stepper` method; for calculation use `stepper`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.initial_cost: CostT
        self.total_cost: CostT
        self.step_cost: deque[CostT] = deque()

    @abstractmethod
    def _stepper(
        self, solution: SolutionRepresentation, environment: Environment
    ) -> Generator[CostT, Any, Any]:
        """
        Generates cost step-by-step. Should be decorated with `CostCalculator._stae_updater`.
        """

    def stepper(
        self,
        solution: SolutionRepresentation,
        environment: Environment,
        initial_cost: CostT = 0,
    ) -> Generator[CostT, None, None]:
        cost_generator = self.__state_updater(self._stepper)
        yield from cost_generator(solution, environment, initial_cost)

    def calculate_total(
        self,
        solution: SolutionRepresentation,
        environment: Environment,
        initial_cost: CostT = 0,
    ) -> tuple[CostT, deque[CostT]]:
        cost_generator = self.stepper(solution, environment, initial_cost)
        exhaust_iterator(cost_generator)
        return self.total_cost, self.step_cost

    def __state_updater(
        self,
        f: Callable[
            [SolutionRepresentation, Environment], Generator[CostT, None, None]
        ],
    ) -> Callable[
        [SolutionRepresentation, Environment, CostT], Generator[CostT, None, None]
    ]:
        """
        Decorator for stepper for state updating.
        """

        @wraps(f)
        def decorator_generator(
            solution: SolutionRepresentation,
            environment: Environment,
            initial_cost: CostT = 0,
        ) -> Generator[CostT, None, None]:
            self.initial_cost = initial_cost
            self.total_cost = initial_cost

            cost_generator = f(solution, environment)

            for next_cost in cost_generator:
                self.total_cost += next_cost
                self.step_cost.append(next_cost)
                yield next_cost

        return decorator_generator
