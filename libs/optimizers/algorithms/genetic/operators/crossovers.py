from typing import TypeVar, Protocol, Generator
from collections.abc import Iterable, Sequence
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

    c1 = p1[:locus] + p2[locus:]
    c2 = p2[:locus] + p1[locus:]

    return c1, c2, rng


def crossover_with_inversion(
    p1: list[T], p2: list[T], rng: Rng, inversion_p: float
) -> tuple[list[T], list[T], Rng]:
    """
    Crossover with locus drawn from uniform distribution (excluding ends).
    Assumes that `p1` and `p2` have the same length. Each chunk can be inverted
    with probability `inversion_p`.
    """

    assert len(p1) == len(p2)
    assert 0 <= inversion_p <= 1

    locus = rng.randint(1, len(p1) - 1)

    c1_chunks = (p1[:locus], p2[locus:])
    c2_chunks = (p2[:locus], p1[locus:])

    c1 = list(
        it.chain.from_iterable(
            __reverse_chunks_with_prob(c1_chunks, inversion_p, rng)[0]
        )
    )
    c2 = list(
        it.chain.from_iterable(
            __reverse_chunks_with_prob(c2_chunks, inversion_p, rng)[0]
        )
    )

    return c1, c2, rng


def crossover_k_loci(
    p1: list[T], p2: list[T], rng: Rng, k: int
) -> tuple[list[T], list[T], Rng]:
    """
    Assumes that `p1` and `p2` have the same length.
    """

    assert len(p1) == len(p2)

    c1_chunks, c2_chunks, rng = __chunkify_parents(p1, p2, rng, k)

    c1 = list(it.chain.from_iterable(c1_chunks))
    c2 = list(it.chain.from_iterable(c2_chunks))

    return c1, c2, rng


def crossover_k_loci_with_inversion(
    p1: list[T], p2: list[T], rng: Rng, k: int, inversion_p: float
) -> tuple[list[T], list[T], Rng]:
    """
    Assumes that `p1` and `p2` have the same length. Inverts chunk with probability
    `inversion_p`
    """

    assert len(p1) == len(p2)
    assert 0 <= inversion_p <= 1

    c1_chunks, c2_chunks, rng = __chunkify_parents(p1, p2, rng, k)

    c1 = list(
        it.chain.from_iterable(
            __reverse_chunks_with_prob(list(c1_chunks), inversion_p, rng)[0]
        )
    )
    c2 = list(
        it.chain.from_iterable(
            __reverse_chunks_with_prob(list(c2_chunks), inversion_p, rng)[0]
        )
    )

    return c1, c2, rng


def crossover_k_loci_poisson(
    p1: list[T], p2: list[T], rng: Rng, lam: float
) -> tuple[list[T], list[T], Rng]:
    """
    Crossover with `k` loci (`k >= 1`) where `k - 1` is drawn from Poisson
    distribution.

    :param lam float: lambda parameter for Poisson distribution
    """

    k = 1 + rng.poisson(lam)
    return crossover_k_loci(p1, p2, k, rng)


def crossover_k_loci_poisson_with_inversion(
    p1: list[T], p2: list[T], rng: Rng, lam: float, inversion_p: float
) -> tuple[list[T], list[T], Rng]:
    """
    Crossover with `k` loci (`k >= 1`) where `k - 1` is drawn from Poisson
    distribution. Each segment between two loci can be inversed with probability
    `inversion_p`.

    :param lam float: lambda parameter for Poisson distribution
    :param inversion_p float:
    """
    k = 1 + rng.poisson(lam)
    return crossover_k_loci_with_inversion(p1, p2, rng, k, inversion_p)


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
        return crossover_k_loci(p1, p2, k, rng)


def __chunkify_parents(
    p1: list[T], p2: list[T], rng: Rng, k: int
) -> tuple[Generator[list[T], None, None], Generator[list[T], None, None], Rng]:
    """
    Chunkifies parents into `k` chunks each.
    """

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

    return new_parent1_chunks, new_parent2_chunks, rng


def __reverse_chunks_with_prob(
    chunks: Sequence[Sequence[T]], p: float, rng: Rng
) -> tuple[Generator[Iterable[T], None, None], Rng]:
    do_reverse: Sequence[bool] = rng.choice(
        [True, False], p=(p, 1 - p), size=len(chunks)
    )
    return (
        reversed(chunk) if do_reverse[i] else chunk for i, chunk in enumerate(chunks)
    ), rng
