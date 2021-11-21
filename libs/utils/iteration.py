from collections.abc import Iterator, Sequence, Iterable, Hashable
from collections import deque
from typing import TypeVar, Union
import itertools as it

import numpy as np
import more_itertools as mit


T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
HashableT = TypeVar("HashableT", bound=Hashable)
Matrix = Union[Sequence[Sequence[T]], np.ndarray]  # type: ignore
Rng = TypeVar("Rng", bound=np.random.Generator)


def exhaust_iterator(i: Iterator) -> None:
    deque(i, maxlen=0)


def chunkify_randomly(a: list[T], n: int, rng: Rng) -> tuple[list[list[T]], Rng]:
    """
    Tries to chunkify `a` into `n` nonempty chunks. If it is impossible,
    returns smaller number of chunks.
    """

    ix_num = len(a)
    split_ixs, rng = random_chunk_range_indices(a, n, rng)

    chunks = [a[i:j] for i, j in mit.windowed(mit.value_chain(0, split_ixs, ix_num), 2)]

    return chunks, rng


def random_chunk_range_indices(a: list, n: int, rng: Rng) -> tuple[list[int], Rng]:
    """
    Returns indices for splitting `a` into `n` nonempty chunks. If it is impossible,
    returns indices for chunkifying into smaller number of chunks.
    """

    ix_num = len(a)
    split_ix_buffer: deque[int] = deque()
    ix_pool = set(range(1, ix_num - 1))

    # n - 1 split indices
    for _ in range(n - 1):
        if not ix_pool:
            break

        new_split_ix = rng.choice(list(ix_pool))
        ix_pool.difference_update([new_split_ix - 1, new_split_ix, new_split_ix + 1])
        split_ix_buffer.append(new_split_ix)

    split_ixs = sorted(split_ix_buffer)

    return split_ixs, rng


def iterator_alternating(a: Sequence[T1], b: Sequence[T2]) -> Iterator[Union[T1, T2]]:
    """
    a: [1, 2, 3, 4],
    b: [5, 6, 7, 8],
    result: [1, 6, 3, 8]
    """

    assert len(a) == len(b)

    return (source[i] for i, source in enumerate(mit.take(len(a), it.cycle((a, b)))))


def find_doubled_indices(a: Iterable[HashableT]) -> dict[HashableT, deque[int]]:
    """
    Returns mapping to indices of doubles of an element in `a`.
    """

    double_indices: dict[HashableT, deque[int]] = {}

    for i, elem in enumerate(a):
        if elem not in double_indices:
            double_indices[elem] = deque()
        else:
            double_indices[elem].append(i)

    return double_indices


def iterate_triangular_indices(
    m: Matrix, upper: bool = True, with_diagonal: bool = True
) -> Iterator[tuple[int, int]]:
    """
    Returns iterator of triangular matrix indices.

    Example
    -------
    >>> m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> result = []
    >>> for x, y in iterate_triangular_indices(m)
    >>>     result.append((x, y))
    >>> print(result)
    [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

    """

    raise DeprecationWarning(
        (
            "deprecated in favor of numpy.triu_indices, numpy_tril_indices, "
            "numpy.triu_indices_from and numpy.tril_indices_from"
        )
    )

    if isinstance(m, np.ndarray):
        height, width = m.shape
    else:
        height = len(m)
        width = len(m[0])

    offset = int(not with_diagonal)

    if upper:

        def ij_gen():  # type: ignore
            i = 0
            j_start = offset
            while i < height:
                j = j_start
                while j < width:
                    yield i, j
                    j += 1
                i += 1
                j_start += 1

    else:

        def ij_gen():
            i = j = 0
            while i < height:
                while j != i + 1 - offset:
                    yield i, j
                    j += 1
                j = 0
                i += 1

    return ((i, j) for i, j in ij_gen())


def iterate_triangular(
    m: Matrix, upper: bool = True, with_diagonal: bool = True
) -> Iterator:
    raise DeprecationWarning(
        (
            "deprecated in favor of numpy.triu_indices, numpy_tril_indices, "
            "numpy.triu_indices_from and numpy.tril_indices_from"
        )
    )

    return (m[i, j] for i, j in iterate_triangular_indices(m, upper, with_diagonal))  # type: ignore
