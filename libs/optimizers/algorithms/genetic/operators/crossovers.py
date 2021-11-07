from abc import ABC, abstractmethod
import numpy as np
import itertools as it
import more_itertools as mit
from copy import copy


from .base import GeneticOperator
from ..chromosomes import Chromosome, ChromosomeHomogenousVector
from .....types import PositiveInt
from .....utils.iteration import chunkify_randomly_indices, iterate_zigzag


class Crossover(GeneticOperator, ABC):
    """
    Abstract base class.
    """

    @abstractmethod
    def execute(
        self, chromosome1: Chromosome, chromosome2: Chromosome
    ) -> tuple[Chromosome, ...]:
        ...

        # Problem - crossover jest losowy, mutacja pewnych
        # fragmentów chromosomu jest nieopłacalna. Wiadomo
        # o tym dopiero po wielu iteracjach
        # Rozwiązanie - adaptacja. Zapamiętuj, gdzie były
        # zmiany i czy dana zmiana poprawiła wynik.
        # Od ustalonej iteracji losuj punkty crossoveru
        # i mutacji z updatowanym rozkładem prawdopodobieństwa
        # opłacalności...


class CrossoverKPoint(Crossover, ABC):
    """
    Abstract base class.
    Crossover with k loci.
    """

    @abstractmethod
    def execute(
        self, chromosome1: Chromosome, chromosome2: Chromosome, k: PositiveInt = 1
    ) -> tuple[Chromosome, ...]:
        ...


class CrossoverHomogenousVector(Crossover):
    """
    Crossover with locus drawn from uniform distribution (excluding ends).
    """

    def execute(
        self,
        chromosome1: ChromosomeHomogenousVector,
        chromosome2: ChromosomeHomogenousVector,
    ) -> tuple[ChromosomeHomogenousVector, ChromosomeHomogenousVector]:
        """
        Crossover with locus drawn from uniform distribution (excluding ends).
        """

        vxs1 = chromosome1.vertex_sequence
        vxs2 = chromosome2.vertex_sequence

        vx_n = len(vxs1)

        locus = np.random.choice(range(1, vx_n - 1))

        new_vxs1 = vxs1[locus:] + vxs2[:locus]
        new_vxs2 = vxs2[locus:] + vxs1[:locus]

        return ChromosomeHomogenousVector(new_vxs1), ChromosomeHomogenousVector(
            new_vxs2
        )


class CrossoverHomogenousVectorKPoint(CrossoverKPoint):
    """
    Crossover with k loci drawn from uniform distribution (excluding ends).
    """

    def execute(
        self,
        chromosome1: ChromosomeHomogenousVector,
        chromosome2: ChromosomeHomogenousVector,
        k: int,
    ) -> tuple[ChromosomeHomogenousVector, ChromosomeHomogenousVector]:
        """
        Crossover with k loci drawn from uniform distribution (excluding ends).
        """

        vxs1 = chromosome1.vertex_sequence
        vxs2 = chromosome2.vertex_sequence

        vx_n = len(vxs1)

        loci = chunkify_randomly_indices(vxs1, k + 1)

        vx1_chunks = [
            vxs1[i:j] for i, j in mit.windowed(mit.value_chain(0, loci, vx_n), 2)
        ]
        vx2_chunks = [
            vxs2[i:j] for i, j in mit.windowed(mit.value_chain(0, loci, vx_n), 2)
        ]

        new_vxs1_chunks = (chunk for chunk in iterate_zigzag(vx1_chunks, vx2_chunks))
        new_vxs2_chunks = (chunk for chunk in iterate_zigzag(vx2_chunks, vx1_chunks))

        new_vxs1 = list(it.chain.from_iterable(new_vxs1_chunks))
        new_vxs2 = list(it.chain.from_iterable(new_vxs2_chunks))

        return ChromosomeHomogenousVector(new_vxs1), ChromosomeHomogenousVector(
            new_vxs2
        )
