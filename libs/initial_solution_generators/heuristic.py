"""
Heuristic solution generators.
"""


from abc import ABC, abstractmethod

from . import SolutionGenerator, Solution
from . import Environment


class Heuristic(SolutionGenerator, ABC):
    @staticmethod
    @abstractmethod
    def __call__(environment: Environment) -> Solution:
        ...
