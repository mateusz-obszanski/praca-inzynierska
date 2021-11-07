from collections import Iterator, Sequence, deque, Iterable
from typing import TypeVar, Union
import numpy as np
import more_itertools as mit


T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
Matrix = Union[Sequence[Sequence[T]], np.ndarray]  # type: ignore



def exhaust_iterator(i: Iterator) -> None:
    deque(i, maxlen=0)


def chunkify_randomly(a: list[T], n: int) -> list[list[T]]:
    """
    Tries to chunkify `a` into `n` nonempty chunks. If it is impossible,
    returns smaller number of chunks.
    """

    ix_num = len(a)
    split_ixs = chunkify_randomly_indices(a, n)

    chunks = [
        a[i:j]
        for i, j in mit.windowed(mit.value_chain(0, split_ixs, ix_num), 2)
    ]

    return chunks


def chunkify_randomly_indices(a: list, n: int) -> list[int]:
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

        new_split_ix = np.random.choice(list(ix_pool))
        ix_pool.difference_update(
            [new_split_ix - 1, new_split_ix, new_split_ix + 1]
        )
        split_ix_buffer.append(new_split_ix)

    split_ixs = sorted(split_ix_buffer)

    return split_ixs


def iterate_zigzag(a: Sequence[T1], b: Sequence[T2]) -> Iterator[Union[T1, T2]]:
    """
    a: [1, 2, 3, 4],
    b: [5, 6, 7, 8],
    result: [1, 6, 3, 8]
    """

    source_choice = [a, b]
    return (source_choice[i % 2][i] for i in range(len(a)))


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

    return (m[i, j] for i, j in iterate_triangular_indices(m, upper, with_diagonal))  # type: ignore
