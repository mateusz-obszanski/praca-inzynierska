from abc import ABC, abstractmethod
from typing import TypeVar
import numpy as np
import itertools as it
import more_itertools as mit

from ..chromosomes import Chromosome, ChromosomeHomogenousVector
from .....types import PositiveInt
from .....utils.iteration import chunkify_randomly_indices, iterate_zigzag


T = TypeVar("T")


def _crossover_k_locus(
    parent1: list[T], parent2: list[T], k: int
) -> tuple[list[T], list[T]]:
    """
    Assumes that `parent1` and `parent2` have the same length.
    """

    parent_len = len(parent1)

    loci = chunkify_randomly_indices(parent1, k + 1)

    parent1_chunks = [
        parent1[i:j] for i, j in mit.windowed(mit.value_chain(0, loci, parent_len), 2)
    ]
    parent2_chunks = [
        parent2[i:j] for i, j in mit.windowed(mit.value_chain(0, loci, parent_len), 2)
    ]

    new_parent1_chunks = (
        chunk for chunk in iterate_zigzag(parent1_chunks, parent2_chunks)
    )
    new_parent2_chunks = (
        chunk for chunk in iterate_zigzag(parent2_chunks, parent1_chunks)
    )

    new_parent1 = list(it.chain.from_iterable(new_parent1_chunks))
    new_parent2 = list(it.chain.from_iterable(new_parent2_chunks))

    return new_parent1, new_parent2


class Crossover(ABC):
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

        vxs1 = chromosome1.sequence
        vxs2 = chromosome2.sequence

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

        vxs1 = chromosome1.sequence
        vxs2 = chromosome2.sequence

        new_vxs1, new_vxs2 = _crossover_k_locus(vxs1, vxs2, k)

        return ChromosomeHomogenousVector(new_vxs1), ChromosomeHomogenousVector(
            new_vxs2
        )


class CrossoverHomogenousVectorKPointPoisson(CrossoverKPoint):
    """
    Crossover with varying locus number drawn from Poisson distribution.
    """

    def __init__(self, lam: float) -> None:
        """
        Crossover with `k` loci (`k >= 1`) where `k - 1` is drawn from Poisson distribution.

        :param lam float: lambda parameter for Poisson distribution
        """

        super().__init__()
        self.lam = lam

    def execute(
        self,
        chromosome1: ChromosomeHomogenousVector,
        chromosome2: ChromosomeHomogenousVector,
    ) -> tuple[ChromosomeHomogenousVector, ChromosomeHomogenousVector]:
        k = 1 + np.random.poisson(self.lam)

        vxs1 = chromosome1.sequence
        vxs2 = chromosome2.sequence

        new_vxs1, new_vxs2 = _crossover_k_locus(vxs1, vxs2, k)

        return ChromosomeHomogenousVector(new_vxs1), ChromosomeHomogenousVector(
            new_vxs2
        )
