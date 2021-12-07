from typing import Optional, Union, TypeVar
from collections.abc import Sequence, Iterable
import itertools as it

import numpy as np
import more_itertools as mit


T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
Rng = TypeVar("Rng", bound=np.random.Generator)


def rand_fill_to_n(
    seq: Sequence[T1], fill_val: T2, n: int, rng: Rng, inplace: bool = False
) -> tuple[list[Union[T1, T2]], Rng]:
    """
    Assumes that `n >= len(c1)`
    """

    if not inplace:
        seq = [*seq]
    seq_len = len(seq)
    diff = n - seq_len

    if diff < 0:
        raise ValueError(
            "`n` has to be greater than `len(seq)`", {"n": n, "seq_len": seq_len}
        )

    if diff == 0:
        return list(seq), rng

    filler_ixs = sorted(rng.integers(seq_len, size=diff))
    shorter_chunk_ixs = mit.windowed(mit.value_chain(0, filler_ixs, n), n=2)

    return (
        list(
            it.chain.from_iterable(
                mit.intersperse(  # type: ignore
                    [fill_val], (seq[i:j] for i, j in shorter_chunk_ixs)
                )
            )
        ),
        rng,
    )


def rand_fill_shorter(
    c1: Sequence[T1], c2: Sequence[T2], fill_val: T3, rng: Rng, inplace: bool = False
) -> tuple[Sequence[Union[T1, T2, T3]], Sequence[Union[T1, T2, T3]], Rng]:
    c1_len = len(c1)
    c2_len = len(c2)
    shorter, longer, _, longer_len = (
        (c1, c2, c1_len, c2_len) if c1_len < c2_len else (c2, c1, c2_len, c1_len)
    )

    filled_shorter, rng = rand_fill_to_n(shorter, fill_val, n=longer_len, rng=rng, inplace=inplace)  # type: ignore

    return filled_shorter, longer, rng


def rand_insert_to_filled(
    seq: list[T], v: T, rng: Rng, fillval: T, inplace: bool = False
) -> tuple[list[T], Rng]:
    ix: int = rng.integers(len(seq))
    if seq[ix] == fillval:
        if not inplace:
            seq = [*seq]
        seq[ix] = v
        return seq, rng
    fv_to_right_ix = next((i for i, val in enumerate(seq[ix:]) if val == fillval), None)
    fv_to_left_ix = next(
        (i for i, val in reversed(tuple(enumerate(seq[:ix]))) if val == fillval), None
    )
    if fv_to_right_ix is not None:
        if fv_to_left_ix is not None:
            left_span = ix - fv_to_left_ix
            right_span = fv_to_right_ix - ix
            if left_span < right_span:
                copy_left = True
            else:
                copy_left = False
        else:
            copy_left = False
    elif fv_to_left_ix is not None:
        copy_left = True
    else:
        # no fillvalues
        return [*seq[:ix], v, *seq[ix + 1 :]], rng
    if not inplace:
        seq = [*seq]
    if copy_left:
        seq[fv_to_left_ix:ix] = seq[fv_to_left_ix + 1 : ix + 1]  # type: ignore
        seq[ix] = v
        return seq, rng
    seq[ix + 1 : fv_to_right_ix + 1] = seq[ix:fv_to_right_ix]  # type: ignore
    seq[ix] = v
    return seq, rng


def rand_del_from_filled(
    seq: list[T],
    rng: Rng,
    fillval: T,
    inplace: bool = False,
    shorten: bool = False,
    ommited_vals: Optional[Iterable[T]] = None,
) -> tuple[list[T], Rng]:
    """
    Removes random `seq` element that is not `fillval`. If `shorten`, deletes
    the index, else inserts `fillval`.
    """

    if ommited_vals is None:
        ommited_vals = ()
    if not inplace:
        seq = [*seq]
    del_ixs = [i for i, x in enumerate(seq) if x != fillval and x not in ommited_vals]
    if not del_ixs:
        if inplace:
            return seq, rng
        return [*seq], rng
    del_ix = del_ixs[rng.integers(len(del_ixs))]
    if shorten:
        del seq[del_ix]
    else:
        seq[del_ix] = fillval
    return seq, rng
