from abc import ABC, abstractmethod

from . import Population


class Selector(ABC):
    """
    Abstract base class.
    """

    @staticmethod
    @abstractmethod
    def __call__(population: Population) -> Population:
        ...
