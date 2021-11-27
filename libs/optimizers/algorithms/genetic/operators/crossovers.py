from typing import TypeVar, Protocol
import numpy as np
import itertools as it
import more_itertools as mit

from .....utils.iteration import random_chunk_range_indices, iterator_alternating


T = TypeVar("T")
Rng = TypeVar("Rng", bound=np.random.Generator)


class Crossover(Protocol):
    def __call__(
        self, p1: list[T], p2: list[T], rng: Rng, *args, **kwargs
    ) -> tuple[list[T], list[T], Rng]:
        """
        Assumes that `p1` and `p2` have the same length.
        """


def crossover(p1: list[T], p2: list[T], rng: Rng) -> tuple[list[T], list[T], Rng]:
    """
    Crossover with locus drawn from uniform distribution (excluding ends).
    Assumes that `p1` and `p2` have the same length.
    """

    assert len(p1) == len(p2)

    locus = rng.randint(1, len(p1) - 1)

    c1 = p1[locus:] + p2[:locus]
    c2 = p2[locus:] + p1[:locus]

    return c1, c2, rng


def crossover_k_locus(
    p1: list[T], p2: list[T], rng: Rng, k: int
) -> tuple[list[T], list[T], Rng]:
    """
    Assumes that `p1` and `p2` have the same length.
    """

    assert len(p1) == len(p2)

    parent_len = len(p1)

    loci, rng = random_chunk_range_indices(p1, k + 1, rng)

    parent1_chunks = [
        p1[i:j] for i, j in mit.windowed(mit.value_chain(0, loci, parent_len), 2)
    ]
    parent2_chunks = [
        p2[i:j] for i, j in mit.windowed(mit.value_chain(0, loci, parent_len), 2)
    ]

    new_parent1_chunks = (
        chunk for chunk in iterator_alternating(parent1_chunks, parent2_chunks)
    )
    new_parent2_chunks = (
        chunk for chunk in iterator_alternating(parent2_chunks, parent1_chunks)
    )

    new_parent1 = list(it.chain.from_iterable(new_parent1_chunks))
    new_parent2 = list(it.chain.from_iterable(new_parent2_chunks))

    return new_parent1, new_parent2, rng


class CrossoverKLociPoisson:
    """
    Crossover with varying locus number drawn from Poisson distribution.
    """

    def __init__(self, lam: float) -> None:
        """
        Crossover with `k` loci (`k >= 1`) where `k - 1` is drawn from Poisson
        distribution.

        :param lam float: lambda parameter for Poisson distribution
        """

        self.lam = lam

    def __call__(
        self, p1: list[T], p2: list[T], rng: Rng
    ) -> tuple[list[T], list[T], Rng]:
        k = 1 + rng.poisson(self.lam)
        return crossover_k_locus(p1, p2, k, rng)
