"""
New population generators.
"""


from abc import ABC, abstractmethod

from . import Population


class Generator(ABC):
    @staticmethod
    @abstractmethod
    def __call__(old_population: Population, *args, **kwargs) -> Population:
        """
        Generates new population from the old one.
        """
