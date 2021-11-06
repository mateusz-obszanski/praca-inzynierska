"""
Encoders and decoders for chromosome representations.
"""


from abc import ABC, abstractmethod
from typing import Any

from ...algorithms import SolutionRepresentation
from ...algorithms.genetic.chromosomes import Chromosome


class Encoder(ABC):
    @staticmethod
    @abstractmethod
    def __call__(solution: SolutionRepresentation) -> Chromosome:
        ...


class Decoder(ABC):
    @staticmethod
    @classmethod
    def __call__(chromosome: Chromosome) -> SolutionRepresentation:
        ...
