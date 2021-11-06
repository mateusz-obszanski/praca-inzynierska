from abc import ABC, abstractmethod

from . import Solution


class SolutionGenerator(ABC):
    @staticmethod
    @abstractmethod
    def __call__(*args, **kwargs) -> Solution:
        ...
