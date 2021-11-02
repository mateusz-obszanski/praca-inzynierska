"""
Random solution generators.
"""


from abc import ABC, abstractmethod

from . import SolutionGenerator, Solution
from . import Environment


class Random(SolutionGenerator, ABC):
    @staticmethod
    @abstractmethod
    def __call__(environment: Environment) -> Solution:
        ...
