from libs.utils.iteration import iterate_triangular, iterate_triangular_indices
from more_itertools import chunked
import numpy as np
import pytest


m = np.array(list(chunked(range(16), 4)))


@pytest.mark.parametrize(
    ["upper", "with_diagonal", "expected"],
    [
        (
            True,
            True,
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 2),
                (2, 3),
                (3, 3),
            ],
        ),
        (
            False,
            True,
            [
                (0, 0),
                (1, 0),
                (1, 1),
                (2, 0),
                (2, 1),
                (2, 2),
                (3, 0),
                (3, 1),
                (3, 2),
                (3, 3),
            ],
        ),
        (False, False, [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]),
        (True, False, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
    ],
)
def test_iterate_triangular_indices(upper, with_diagonal, expected):
    assert list(iterate_triangular_indices(m, upper, with_diagonal)) == expected


@pytest.mark.parametrize(
    ["upper", "with_diagonal", "expected"],
    [
        (True, True, [0, 1, 2, 3, 5, 6, 7, 10, 11, 15]),
        (True, False, [1, 2, 3, 6, 7, 11]),
        (False, True, [0, 4, 5, 8, 9, 10, 12, 13, 14, 15]),
        (False, False, [4, 8, 9, 12, 13, 14]),
    ],
)
def test_iterate_triangular(upper, with_diagonal, expected):
    assert list(iterate_triangular(m, upper, with_diagonal)) == [
        m[i, j] for i, j in iterate_triangular_indices(m, upper, with_diagonal)
    ]
