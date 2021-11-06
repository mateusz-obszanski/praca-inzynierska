"""
Encoders and decoders for chromosome representations.
"""


from abc import ABC, abstractmethod
from typing import Any

from ...algorithms import Solution
from ...algorithms.genetic.chromosomes import Chromosome


class Encoder(ABC):
    @staticmethod
    @abstractmethod
    def __call__(solution: Solution) -> Chromosome:
        ...


class Decoder(ABC):
    @staticmethod
    @classmethod
    def __call__(chromosome: Chromosome) -> Solution:
        ...
