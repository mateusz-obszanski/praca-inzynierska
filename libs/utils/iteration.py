from collections import Iterator, Sequence, deque
from typing import TypeVar, Union
import numpy as np


T = TypeVar("T")
Matrix = Union[Sequence[Sequence[T]], np.ndarray]  # type: ignore


def exhaust_iterator(i: Iterator) -> None:
    deque(i, maxlen=0)


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
