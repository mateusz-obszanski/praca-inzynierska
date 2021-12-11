from typing import Any, TypeVar, Protocol
from collections.abc import Sequence

import numpy as np
import itertools as it

from libs.utils.random import randomly_chunkify


T = TypeVar("T")
Rng = TypeVar("Rng", bound=np.random.Generator)


class Mutator(Protocol):
    def __call__(
        self, c: np.ndarray, p: float, rng: Rng, *args
    ) -> tuple[np.ndarray, Rng]:
        ...


def mutate_swap(c: np.ndarray, p: float, rng: Rng) -> tuple[np.ndarray, Rng]:
    """
    Shuffles `seq` on indices marked with probability `p`.
    """

    marks = rng.random(size=c.shape[0]) < p
    marked_elems = c[marks]
    rng.shuffle(marked_elems)
    c[marks] = marked_elems

    return c, rng


def mutate_insert(
    c: np.ndarray,
    p: float,
    rng: Rng,
    rand_range: tuple[int, int],
    ini_and_dummy_vxs: set[int],
    fillval: int,
) -> tuple[np.ndarray, Rng]:
    """
    `p` probability of insertion at index - resulting seq will have random val
    inserted at that index.
    `rand_range` - range [min, max) to draw values from.
    """

    c_len = c.shape[0]
    marks: np.ndarray = rng.random(size=c_len) < p
    fillvals: np.ndarray = c == fillval
    repl_marks: np.ndarray = fillvals & marks
    choice_pool: np.ndarray = np.fromiter(
        x for x in range(*rand_range) if x not in ini_and_dummy_vxs
    )
    v_inplace_fv: np.ndarray = rng.choice(
        choice_pool, size=np.count_nonzero(repl_marks)
    )
    c[repl_marks] = v_inplace_fv
    ins_marks = marks.copy()
    ins_marks[repl_marks] = False
    mutated = np.empty(shape=(c_len + np.count_nonzero(ins_marks)), dtype=np.int64)
    mutated[ins_marks] = rng.choice(choice_pool, size=ins_marks.shape)
    mutated[~marks] = c

    return mutated, rng


def mutate_insert_irp(
    c: np.ndarray,
    p: float,
    rng: Rng,
    rand_vx_range: tuple[int, int],
    ini_and_dummy_vxs: set[int],
    fillval: int,
    rand_quantity_range: tuple[float, float],
) -> tuple[np.ndarray, Rng]:
    """
    `p` probability of insertion at index - resulting seq will have random val
    inserted at that index.
    `rand_range` - range [min, max) to draw values from.
    """

    c_len = c.shape[0]
    marks: np.ndarray = rng.random(size=c_len) < p
    fillvals: np.ndarray = c[:, 0] == fillval
    repl_marks: np.ndarray = fillvals & marks
    vx_choice_pool: np.ndarray = np.fromiter(
        x for x in range(*rand_vx_range) if x not in ini_and_dummy_vxs
    )
    v_inplace_fv: np.ndarray = rng.choice(
        vx_choice_pool, size=np.count_nonzero(repl_marks)
    )
    c[repl_marks] = v_inplace_fv
    ins_marks = marks.copy()
    ins_marks[repl_marks] = False
    mutated = np.empty(shape=(c_len + np.count_nonzero(ins_marks)), dtype=np.int64)
    rand_vxs = rng.choice(vx_choice_pool, size=ins_marks.shape)
    rand_qs = rng.uniform(*rand_quantity_range, size=rand_vxs.shape)
    mutated[ins_marks] = np.stack((rand_vxs, rand_qs), axis=1)
    mutated[~marks] = c

    return mutated, rng


def mutate_del(
    c: np.ndarray, p: float, rng: Rng, fillval: Any
) -> tuple[np.ndarray, Rng]:
    """
    `p` - probability of deletion at each index by inserting `fillval`.
    """

    seq_len = c.shape[0]
    marks = rng.random(size=seq_len) < p
    c[marks] = fillval
    return c, rng


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
    should_reverse = rng.random(size=len(chunks)) < p
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
    reverse_odd = rng.random() < 0.5
    transformed_chunks = (
        reversed(chunk) if bool(i % 2) == reverse_odd else chunk
        for i, chunk in enumerate(chunks)
    )

    return list(it.chain.from_iterable(transformed_chunks)), rng
