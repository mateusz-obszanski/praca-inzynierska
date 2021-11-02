"""
Algorithms.
"""


from .. import Algorithm
from .chromosomes import Chromosome
from .coders import Encoder, Decoder
from .operators import Mutation, Crossover


class Genetic(Algorithm):
    ...
