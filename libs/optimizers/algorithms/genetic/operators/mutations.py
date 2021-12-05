from typing import TypeVar, Protocol
from collections.abc import Sequence

import numpy as np
import itertools as it

from libs.utils.random import randomly_chunkify


T = TypeVar("T")
Rng = TypeVar("Rng", bound=np.random.Generator)


class Mutator(Protocol):
    def __call__(self, seq: Sequence[T], p: float, rng: Rng) -> tuple[list[T], Rng]:
        ...


def mutate_swap(seq: Sequence[T], p: float, rng: Rng) -> tuple[list[T], Rng]:
    """
    Shuffles `seq` on indices marked with probability `p`.
    """

    assert 0 <= p <= 1

    seq_array = np.array(seq)
    marks = np.random.choice([True, False], size=len(seq_array), p=[p, 1 - p])
    marked_elems = seq_array[marks]
    rng.shuffle(marked_elems)
    seq_array[marks] = marked_elems

    return seq_array.tolist(), rng


def mutate_shuffle_ranges(seq: Sequence[T], p: float, rng: Rng) -> tuple[list[T], Rng]:
    """
    Shuffles `seq`'s slices sliced at loci generated with probability `p`.
    """

    chunks, rng = randomly_chunkify(seq, p, rng)
    rng.shuffle(chunks)

    return list(it.chain.from_iterable(chunks)), rng


def mutate_reverse_ranges(seq: Sequence[T], p: float, rng: Rng) -> tuple[list[T], Rng]:
    """
    Chunkifies `seq` at loci generated with probability `p` and reverses chunks
    with probability `p`.
    """

    chunks, rng = randomly_chunkify(seq, p, rng)
    should_reverse = rng.choice([True, False], p=[p, 1 - p], size=len(chunks))
    return (
        list(
            it.chain.from_iterable(
                reversed(c) if should_reverse[i] else c for i, c in enumerate(chunks)
            )
        ),
        rng,
    )


def mutate_reverse_ranges_alternately(
    seq: Sequence[T], p: float, rng: Rng
) -> tuple[list[T], Rng]:
    """
    Chunkifies `seq` at loci generated with probability `p` and reverses
    every second chunk starting randomly at first or second chunk.
    """

    chunks, rng = randomly_chunkify(seq, p, rng)
    reverse_odd = rng.choice([True, False])
    transformed_chunks = (
        reversed(chunk) if bool(i % 2) == reverse_odd else chunk
        for i, chunk in enumerate(chunks)
    )

    return list(it.chain.from_iterable(transformed_chunks)), rng
