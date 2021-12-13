from typing import TypeVar, Protocol, Generator
from collections.abc import Iterable, Sequence, Iterator
import numpy as np
import itertools as it
import more_itertools as mit

from libs.utils.iteration import random_chunk_range_indices, iterator_alternating


T = TypeVar("T")
Rng = TypeVar("Rng", bound=np.random.Generator)


class Crossover(Protocol):
    def __call__(
        self, p1: list[T], p2: list[T], rng: Rng, *args, **kwargs
    ) -> tuple[list[T], list[T], Rng]:
        """
        Assumes that `p1` and `p2` have the same length.
        """
        ...


class CrossoverNDArray(Protocol):
    def __call__(
        self, p1: np.ndarray, p2: np.ndarray, rng: Rng, *args, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, Rng]:
        """
        Assumes that `p1` and `p2` have the same length.
        """
        ...


def crossover(p1: list[T], p2: list[T], rng: Rng) -> tuple[list[T], list[T], Rng]:
    """
    Crossover with locus drawn from uniform distribution (excluding ends).
    Assumes that `p1` and `p2` have the same length.
    """

    locus = rng.randint(1, len(p1) - 1)

    c1 = p1[:locus] + p2[locus:]
    c2 = p2[:locus] + p1[locus:]

    return c1, c2, rng


def crossover_ndarray(
    p1: np.ndarray, p2: np.ndarray, rng: Rng
) -> tuple[np.ndarray, np.ndarray, Rng]:
    """
    Crossover with locus drawn from uniform distribution (excluding ends).
    Assumes that `p1` and `p2` have the same length.
    """

    locus = rng.randint(1, p1.shape[0] - 1)

    c1 = np.concatenate((p1[:locus], p2[locus:]))
    c2 = np.concatenate((p2[:locus], p1[locus:]))

    return c1, c2, rng


def crossover_with_inversion(
    p1: list[T], p2: list[T], rng: Rng, inversion_p: float
) -> tuple[list[T], list[T], Rng]:
    """
    Crossover with locus drawn from uniform distribution (excluding ends).
    Assumes that `p1` and `p2` have the same length. Each chunk can be inverted
    with probability `inversion_p`.
    """

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


def crossover_with_inversion_ndarray(
    p1: np.ndarray, p2: np.ndarray, rng: Rng, inversion_p: float
) -> tuple[np.ndarray, np.ndarray, Rng]:
    """
    Crossover with locus drawn from uniform distribution (excluding ends).
    Assumes that `p1` and `p2` have the same length. Each chunk can be inverted
    with probability `inversion_p`.
    """

    locus = rng.randint(1, p1.shape[0] - 1)

    c1_chunks = (p1[:locus], p2[locus:])
    c2_chunks = (p2[:locus], p1[locus:])

    c1, c2 = (
        np.concatenate(tuple(__reverse_chunks_with_prob_ndarray(cs, inversion_p, rng)))
        for cs in (c1_chunks, c2_chunks)
    )

    return c1, c2, rng


def crossover_k_loci(
    p1: list[T], p2: list[T], rng: Rng, k: int
) -> tuple[list[T], list[T], Rng]:
    """
    Assumes that `p1` and `p2` have the same length.
    """

    c1_chunks, c2_chunks, rng = __chunkify_parents(p1, p2, rng, k)

    c1 = list(it.chain.from_iterable(c1_chunks))
    c2 = list(it.chain.from_iterable(c2_chunks))

    return c1, c2, rng


def crossover_k_loci_ndarray(
    p1: np.ndarray, p2: np.ndarray, rng: Rng, k: int
) -> tuple[np.ndarray, np.ndarray, Rng]:
    """
    Assumes that `p1` and `p2` have the same length.
    """

    c1_chunks, c2_chunks, rng = __chunkify_parents_ndarray(p1, p2, rng, k)
    c1, c2 = (
        np.concatenate(tuple(it.chain.from_iterable(cs)))
        for cs in (c1_chunks, c2_chunks)
    )

    return c1, c2, rng


def crossover_k_loci_with_inversion(
    p1: list[T], p2: list[T], rng: Rng, k: int, inversion_p: float
) -> tuple[list[T], list[T], Rng]:
    """
    Assumes that `p1` and `p2` have the same length. Inverts chunk with probability
    `inversion_p`
    """

    c1_chunks, c2_chunks, rng = __chunkify_parents(p1, p2, rng, k)

    c1 = list(
        it.chain.from_iterable(
            __reverse_chunks_with_prob(tuple(c1_chunks), inversion_p, rng)[0]
        )
    )
    c2 = list(
        it.chain.from_iterable(
            __reverse_chunks_with_prob(tuple(c2_chunks), inversion_p, rng)[0]
        )
    )

    return c1, c2, rng


def crossover_k_loci_with_inversion_ndarray(
    p1: np.ndarray, p2: np.ndarray, rng: Rng, k: int, inversion_p: float
) -> tuple[np.ndarray, np.ndarray, Rng]:
    """
    Assumes that `p1` and `p2` have the same length. Inverts chunk with probability
    `inversion_p`
    """

    c1_chunks, c2_chunks, rng = __chunkify_parents_ndarray(p1, p2, rng, k)

    c1, c2 = (
        np.concatenate(rcs)
        for rcs in (
            __reverse_chunks_with_prob_ndarray(tuple(cs), inversion_p, rng)[0]
            for cs in (c1_chunks, c2_chunks)
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


def crossover_k_loci_poisson_ndarray(
    p1: np.ndarray, p2: np.ndarray, rng: Rng, lam: float
) -> tuple[np.ndarray, np.ndarray, Rng]:
    """
    Crossover with `k` loci (`k >= 1`) where `k - 1` is drawn from Poisson
    distribution.

    :param lam float: lambda parameter for Poisson distribution
    """

    k = 1 + rng.poisson(lam)
    return crossover_k_loci_ndarray(p1, p2, k, rng)


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


def crossover_k_loci_poisson_with_inversion_ndarray(
    p1: np.ndarray, p2: np.ndarray, rng: Rng, lam: float, inversion_p: float
) -> tuple[np.ndarray, np.ndarray, Rng]:
    """
    Crossover with `k` loci (`k >= 1`) where `k - 1` is drawn from Poisson
    distribution. Each segment between two loci can be inversed with probability
    `inversion_p`.

    :param lam float: lambda parameter for Poisson distribution
    :param inversion_p float:
    """
    k = 1 + rng.poisson(lam)
    return crossover_k_loci_with_inversion_ndarray(p1, p2, rng, k, inversion_p)


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
) -> tuple[Iterator[list[T]], Iterator[list[T]], Rng]:
    """
    Chunkifies parents into `k` chunks each.
    """

    parent_len = len(p1)

    loci, rng = random_chunk_range_indices(parent_len, k + 1, rng)

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


def __chunkify_parents_ndarray(
    p1: np.ndarray, p2: np.ndarray, rng: Rng, k: int
) -> tuple[Iterator[np.ndarray], Iterator[np.ndarray], Rng]:
    p_len = p1.shape[0]
    ixs, rng = random_chunk_range_indices(p_len, k, rng)
    ixs = it.chain((0,), ixs, (p_len,))
    p1_chunks, p2_chunks = (
        tuple(p[i:j] for i, j in mit.windowed(ixs, n=2)) for p in (p1, p2)
    )
    c1_chunks, c2_chunks = (
        iterator_alternating(cs1, cs2)
        for cs1, cs2 in ((p1_chunks, p2_chunks), (p2_chunks, p1_chunks))
    )
    return c1_chunks, c2_chunks, rng


def __reverse_chunks_with_prob(
    chunks: Sequence[Sequence[T]], p: float, rng: Rng
) -> tuple[Iterator[Iterable[T]], Rng]:
    do_reverse: Sequence[bool] = rng.random(size=len(chunks)) < p
    return (
        reversed(chunk) if do_reverse[i] else chunk for i, chunk in enumerate(chunks)
    ), rng


def __reverse_chunks_with_prob_ndarray(
    chunks: Sequence[np.ndarray], p: float, rng: Rng
) -> tuple[Iterator[np.ndarray], Rng]:
    do_reverse: Sequence[bool] = rng.random(size=len(chunks)) < p
    return (
        np.flip(chunk, axis=0) if do_reverse[i] else chunk
        for i, chunk in enumerate(chunks)
    ), rng
