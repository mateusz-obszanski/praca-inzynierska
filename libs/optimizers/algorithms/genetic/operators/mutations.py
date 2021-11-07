from abc import ABC, abstractmethod
from typing import TypeVar
from copy import deepcopy
import numpy as np
import itertools as it
import more_itertools as mit
import random

from .base import GeneticOperator
from ..chromosomes import Chromosome, ChromosomeTSP


class Mutator(GeneticOperator):
    """
    Abstract base class.
    """

    ChromosomeT = TypeVar("ChromosomeT", bound=Chromosome)

    @abstractmethod
    def mutate(self, chromosome: ChromosomeT, inplace: bool = False) -> ChromosomeT:
        """
        Alias for `self.execute`.
        """

        chromosome = chromosome if inplace else deepcopy(chromosome)
        return chromosome

    def execute(self, chromosome: ChromosomeT, inplace: bool = False) -> ChromosomeT:
        """
        For compatibility with `GeneticOperator` ABC.
        """

        return self.mutate(chromosome, inplace)


class MutatorSwap(Mutator, ABC):
    """
    Abstract base class. Swaps to single genes.
    """


class MutatorTSP(Mutator, ABC):
    """
    Abstract base class.
    """


class MutatorTSPSwap(MutatorTSP):
    """
    Swaps genes k times, where k is drawn from Poisson distribution.
    """

    def mutate(
        self, chromosome: ChromosomeTSP, inplace: bool = False, lam: float = 1
    ) -> ChromosomeTSP:
        """
        Swaps genes k times, where k is drawn from Poisson distribution.
        `lam` - lambda parameter for Poisson distribution.
        """
        chromosome = super().mutate(chromosome, inplace)

        vx_sequence = chromosome.vertex_sequence

        k = np.random.poisson(lam)

        ixs = list(range(len(vx_sequence)))
        ix_set = set(ixs)

        indices_to_swap = np.random.choice(ixs, size=k)
        second_indices = [np.random.choice(list(ix_set - {i})) for i in indices_to_swap]

        for i, j in zip(indices_to_swap, second_indices):
            vx_sequence[i], vx_sequence[j] = vx_sequence[j], vx_sequence[i]

        return chromosome


class MutatorTSPShuffle(MutatorTSP):
    """
    Shuffles genes in ranges determined by split indices drawn from poisson distribution.
    """

    ChromosomeTSPT = TypeVar("ChromosomeTSPT", bound=ChromosomeTSP)

    def mutate(
        self,
        chromosome: ChromosomeTSPT,
        inplace: bool = False,
        probability: float = 1,
        lam: float = 1,
    ) -> ChromosomeTSPT:
        """
        Shuffles genes in ranges determinde by indices drawn from exponential distribution.
        `probability` - probability of mutation occuring at all
        `lam` - lambda parameter for Poisson distribution.
        """

        # FIXME optimization - this unnecesarily copies vertex_sequence (which is not modified later)
        chromosome = super().mutate(chromosome, inplace)
        do_it = np.random.choice([True, False], p=[probability, 1 - probability])

        if not do_it:
            return chromosome

        vx_sequence = chromosome.vertex_sequence
        ix_num = len(vx_sequence)
        split_ix_n = np.random.poisson(lam)

        split_ix_set: set[int] = set()
        ix_pool = set(range(ix_num)) - {0, ix_num - 1}

        for _ in range(split_ix_n):
            if not ix_pool:
                break

            new_split_ix = np.random.choice(list(ix_pool))
            ix_pool.difference_update(
                [new_split_ix - 1, new_split_ix, new_split_ix + 1]
            )
            split_ix_set.add(new_split_ix)

        split_ixs = sorted(split_ix_set)

        chunks = [
            vx_sequence[i:j]
            for i, j in mit.windowed(mit.value_chain(0, split_ixs, ix_num), 2)
        ]
        chunk_n = len(chunks)
        chunk_permutation_ixs = list(range(chunk_n))

        # guarantee mutation
        while True:
            new_permutation: list[int] = np.random.permutation(chunk_n).tolist()
            if new_permutation != chunk_permutation_ixs:
                chunk_permutation_ixs = new_permutation
                break

        shuffled_vx_sequence = list(
            it.chain.from_iterable(chunks[i] for i in chunk_permutation_ixs)
        )
        chromosome.vertex_sequence = shuffled_vx_sequence

        return chromosome