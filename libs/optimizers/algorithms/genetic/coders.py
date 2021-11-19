"""
Encoders and decoders for chromosome representations.
"""


from abc import ABC, abstractmethod
from typing import Any

from ....solution.representation import SolutionRepresentation
from ...algorithms.genetic.chromosomes import Chromosome


class Encoder(ABC):
    """
    Abstract base class.
    """

    @staticmethod
    @abstractmethod
    def __call__(solution: SolutionRepresentation) -> Chromosome:
        ...


class Decoder(ABC):
    """
    Abstract base class.
    """

    @staticmethod
    @classmethod
    def __call__(chromosome: Chromosome) -> SolutionRepresentation:
        ...
