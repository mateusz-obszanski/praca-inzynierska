from typing import Any, Iterable, TypeVar, Protocol
from collections.abc import Sequence

import numpy as np
import itertools as it
import more_itertools as mit

from libs.utils.iteration import join_sequences
from libs.utils.random import randomly_chunkify, shuffle_ensure_change


T = TypeVar("T")
Rng = TypeVar("Rng", bound=np.random.Generator)


class Mutator(Protocol):
    def __call__(
        self, c: np.ndarray, p: float, rng: Rng, *args, **kwargs
    ) -> tuple[np.ndarray, Rng]:
        ...


def mutate_swap(c: np.ndarray, p: float, rng: Rng) -> tuple[np.ndarray, Rng]:
    """
    Shuffles `seq` on indices marked with probability `p`.
    """

    marks = rng.random(size=c.shape[0]) < p
    marked_elems = c[marks]
    marked_elems, rng = shuffle_ensure_change(marked_elems, rng, max_iter=10)
    c[marks] = marked_elems

    return c, rng


def mutate_swap_irp(
    c: np.ndarray, p: float, rng: Rng, quantities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, Rng]:
    """
    Shuffles `seq` on indices marked with probability `p`.
    """

    marks = rng.random(size=c.shape[0]) < p
    mark_ixs = np.fromiter((i for i, m in enumerate(marks) if m), dtype=np.int64)
    mark_ixs, rng = shuffle_ensure_change(mark_ixs, rng, max_iter=10)
    c[marks] = c[mark_ixs]
    quantities[marks] = quantities[mark_ixs]

    return c, quantities, rng


# def mutate_insert(
#     c: np.ndarray,
#     p: float,
#     rng: Rng,
#     rand_range: tuple[int, int],
#     ini_and_dummy_vxs: set[int],
#     fillval: int,
# ) -> tuple[np.ndarray, Rng]:
#     """
#     `p` probability of insertion at index - resulting seq will have random val
#     inserted at that index.
#     `rand_range` - range [min, max) to draw values from.
#     """

#     c_len = c.shape[0]
#     marks: np.ndarray = rng.random(size=c_len) < p
#     fillvals: np.ndarray = c == fillval
#     repl_marks: np.ndarray = fillvals & marks
#     choice_pool: np.ndarray = np.fromiter(
#         (x for x in range(*rand_range) if x not in ini_and_dummy_vxs), dtype=np.int64
#     )
#     v_inplace_fv: np.ndarray = rng.choice(
#         choice_pool, size=np.count_nonzero(repl_marks)
#     )
#     c[repl_marks] = v_inplace_fv
#     ins_marks = marks.copy()
#     ins_marks[repl_marks] = False

#     chunk_ranges: tuple[tuple[int, int]] = tuple(mit.windowed(  # type: ignore
#         (0, *(i for i, m in enumerate(ins_marks) if m), ins_marks.shape[0]), n=2
#     ))

#     ins_marks_chunks = [ins_marks[x:y] for x, y in chunk_ranges]
#     marks_chunks = [marks[x:y] for x, y in chunk_ranges]

#     ins_marks = np.array(join_sequences(ins_marks_chunks, val=False), dtype=np.bool8)
#     marks = np.array(join_sequences(marks_chunks, val=False))

#     mutated = np.empty(shape=(c_len + np.count_nonzero(ins_marks)), dtype=np.int64)
#     mutated[ins_marks] = rng.choice(choice_pool, size=ins_marks.shape)
#     mutated[~marks] = c

#     return mutated, rng


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
        (x for x in range(*rand_range) if x not in ini_and_dummy_vxs), dtype=np.int64
    )
    v_inplace_fv: np.ndarray = rng.choice(
        choice_pool, size=np.count_nonzero(repl_marks)
    )
    c[repl_marks] = v_inplace_fv
    ins_marks = marks & ~repl_marks
    nonzero_ins_marks = np.count_nonzero(ins_marks)
    insert_ixs = np.fromiter((i for i, m in enumerate(ins_marks) if m), dtype=np.int64)
    c = np.insert(c, insert_ixs, rng.choice(choice_pool, size=nonzero_ins_marks))

    return c, rng


# def mutate_insert_irp(
#     c: np.ndarray,
#     p: float,
#     rng: Rng,
#     rand_vx_range: tuple[int, int],
#     ini_and_dummy_vxs: set[int],
#     fillval: int,
#     rand_quantity_range: tuple[float, float],
# ) -> tuple[np.ndarray, Rng]:
#     """
#     `p` probability of insertion at index - resulting seq will have random val
#     inserted at that index.
#     `rand_range` - range [min, max) to draw values from.
#     """

#     c_len = c.shape[0]
#     marks: np.ndarray = rng.random(size=c_len) < p
#     fillvals: np.ndarray = c[:, 0] == fillval
#     repl_marks: np.ndarray = fillvals & marks
#     vx_choice_pool: np.ndarray = np.fromiter(
#         (x for x in range(*rand_vx_range) if x not in ini_and_dummy_vxs), dtype=np.int64
#     )
#     v_inplace_fv: np.ndarray = rng.choice(
#         vx_choice_pool, size=np.count_nonzero(repl_marks)
#     )
#     c[repl_marks] = v_inplace_fv
#     ins_marks = marks.copy()
#     ins_marks[repl_marks] = False
#     mutated = np.empty(shape=(c_len + np.count_nonzero(ins_marks)), dtype=np.int64)
#     rand_vxs = rng.choice(vx_choice_pool, size=ins_marks.shape)
#     rand_qs = rng.uniform(*rand_quantity_range, size=rand_vxs.shape)
#     mutated[ins_marks] = np.stack((rand_vxs, rand_qs), axis=1)
#     mutated[~marks] = c

#     return mutated, rng


def mutate_insert_irp(
    c: np.ndarray,
    p: float,
    rng: Rng,
    rand_vx_range: tuple[int, int],
    ini_and_dummy_vxs: set[int],
    fillval: int,
    rand_quantity_range: tuple[float, float],
    quantities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Rng]:
    """
    `c` is a sequence of vertices - ints
    `quantities` is a list of quantities left at `c`'s vertices
    `p` probability of insertion at index - resulting seq will have random val
    inserted at that index.
    `rand_range` - range [min, max) to draw values from.

    Returns vx sequence, quantities and rng.
    """

    c_len = c.shape[0]
    marks: np.ndarray = rng.random(size=c_len) < p
    fillvals: np.ndarray = c == fillval
    repl_marks: np.ndarray = fillvals & marks
    vx_choice_pool: np.ndarray = np.fromiter(
        (x for x in range(*rand_vx_range) if x not in ini_and_dummy_vxs), dtype=np.int64
    )
    true_repl_marks_n = np.count_nonzero(repl_marks)
    v_inplace_fv: np.ndarray = rng.choice(vx_choice_pool, size=true_repl_marks_n)
    c[repl_marks] = v_inplace_fv
    quantities[repl_marks] = rng.uniform(*rand_quantity_range, size=true_repl_marks_n)

    ins_marks = marks & ~repl_marks
    nonzero_ins_marks = np.count_nonzero(ins_marks)
    insert_ixs = np.fromiter((i for i, m in enumerate(ins_marks) if m), dtype=np.int64)
    c = np.insert(c, insert_ixs, rng.choice(vx_choice_pool, size=nonzero_ins_marks))
    quantities = np.insert(
        c, insert_ixs, rng.uniform(*rand_quantity_range, size=nonzero_ins_marks)
    )

    return c, quantities, rng


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


def mutate_del_irp(
    c: np.ndarray, p: float, rng: Rng, fillval: Any, quantities: np.ndarray
) -> tuple[np.ndarray, Rng]:
    """
    `p` - probability of deletion at each index by inserting `fillval`.
    """

    seq_len = c.shape[0]
    marks = rng.random(size=seq_len) < p
    c[marks] = fillval
    quantities[marks] = 0.0
    return c, rng


def mutate_shuffle_ranges(seq: Sequence[T], p: float, rng: Rng) -> tuple[list[T], Rng]:
    """
    Shuffles `seq`'s slices sliced at loci generated with probability `p`.
    """

    chunks, rng = randomly_chunkify(seq, p, rng)
    chunks, rng = shuffle_ensure_change(chunks, rng, max_iter=10)

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
