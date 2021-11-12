from abc import ABC, abstractmethod
from typing import TypeVar
from copy import deepcopy
import numpy as np
import itertools as it

from libs.utils.iteration import chunkify_randomly

from ..chromosomes import Chromosome, ChromosomeHomogenousVector


DidMutate = bool


class Mutator(ABC):
    """
    Abstract base class.
    """

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    ChromosomeT = TypeVar("ChromosomeT", bound=Chromosome)

    @abstractmethod
    def mutate(
        self, chromosome: ChromosomeT, inplace: bool = False
    ) -> tuple[ChromosomeT, DidMutate]:
        """
        Abstract method.
        """

        chromosome = chromosome if inplace else deepcopy(chromosome)
        return chromosome, False


class MutatorHomogenousVectorSwap(Mutator):
    """
    Swaps genes k times, where k is drawn from Poisson distribution.
    """

    ChromosomeHomogenousVectorT = TypeVar(
        "ChromosomeHomogenousVectorT", bound=ChromosomeHomogenousVector
    )

    def __init__(self, probability: float, lam: float) -> None:
        """
        :param lam float: lambda parameter for random k choice by Poisson distribution
        """

        super().__init__(probability)
        self.lam = lam

    def mutate(
        self,
        chromosome: ChromosomeHomogenousVectorT,
        inplace: bool = False,
    ) -> tuple[ChromosomeHomogenousVectorT, DidMutate]:
        """
        Swaps genes 1 + k times, where k is drawn from Poisson distribution.
        """

        chromosome, _ = super().mutate(chromosome, inplace)

        probability = self.probability

        do_it = np.random.choice([True, False], p=[probability, 1 - probability])

        if not do_it:
            return chromosome, False

        vx_sequence = chromosome.sequence

        k = np.random.poisson(self.lam)

        ixs = list(range(len(vx_sequence)))
        ix_set = set(ixs)

        indices_to_swap = np.random.choice(ixs, size=1 + k)
        second_indices = [np.random.choice(list(ix_set - {i})) for i in indices_to_swap]

        for i, j in zip(indices_to_swap, second_indices):
            vx_sequence[i], vx_sequence[j] = vx_sequence[j], vx_sequence[i]

        return chromosome, True


class MutatorHomogenousVectorShuffle(Mutator):
    """
    Shuffles genes in ranges determined by split indices drawn from Poisson distribution.
    """

    ChromosomeHomogenousVectorT = TypeVar(
        "ChromosomeHomogenousVectorT", bound=ChromosomeHomogenousVector
    )

    def __init__(self, probability: float, lam: float) -> None:
        """
        :param probability float: probability of gene mutation
        "param lam float: lambda parameter for shuffle indices choice by Poisson
            distribution
        """

        super().__init__(probability)
        self.lam = lam

    def mutate(
        self,
        chromosome: ChromosomeHomogenousVectorT,
        inplace: bool = False,
    ) -> tuple[ChromosomeHomogenousVectorT, DidMutate]:
        """
        Shuffles genes in ranges determinde by indices drawn from Poisson distribution.
        `probability` - probability of mutation occuring at all
        `lam` - lambda parameter for Poisson distribution.
        """

        MAX_ITERATION = 100

        probability = self.probability

        # ! `chromosome.vertex_sequence` is not mutated, hence this is OK
        chromosome = chromosome if inplace else type(chromosome)(chromosome.sequence)

        do_it = np.random.choice([True, False], p=[probability, 1 - probability])

        if not do_it:
            return chromosome, False

        vx_sequence = chromosome.sequence
        split_ix_n = np.random.poisson(self.lam) + 1

        chunks = chunkify_randomly(vx_sequence, split_ix_n + 1)

        chunk_n = len(chunks)
        chunk_permutation_ixs = list(range(chunk_n))

        # guarantee mutation
        for _ in range(MAX_ITERATION):
            new_permutation: list[int] = np.random.permutation(chunk_n).tolist()
            if new_permutation != chunk_permutation_ixs:
                chunk_permutation_ixs = new_permutation
                break
        else:
            # executes if loop did not break
            return chromosome, False

        shuffled_vx_sequence = list(
            it.chain.from_iterable(chunks[i] for i in chunk_permutation_ixs)
        )

        chromosome.sequence = shuffled_vx_sequence

        return chromosome, True
