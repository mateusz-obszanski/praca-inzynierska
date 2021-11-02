from abc import ABC, abstractmethod

from . import Population


class Selector(ABC):
    @staticmethod
    @abstractmethod
    def __call__(population: Population) -> Population:
        ...
