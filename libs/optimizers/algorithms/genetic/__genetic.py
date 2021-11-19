"""
Algorithms.
"""


from abc import ABC, abstractmethod
from .. import Algorithm
from .chromosomes import Chromosome
from .coders import Encoder, Decoder
from .population.generators import PopulationGenerator, PopulationGenerationData


class Genetic(Algorithm, ABC):
    @abstractmethod
    def step(self):
        ...


class GeneticTSP(Genetic):
    def step(self):
        ...
