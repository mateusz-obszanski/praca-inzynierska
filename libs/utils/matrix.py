import numpy as np


Matrix = np.ndarray


def symmetricize(mx: Matrix, from_triu: bool = True) -> Matrix:
    if from_triu:
        triangular = np.triu
        offset = 1
    else:
        triangular = np.tril
        offset = -1

    return triangular(mx) + triangular(mx, offset).T


def antisymmetricize(mx: Matrix, from_triu: bool = True) -> Matrix:
    if from_triu:
        triangular = np.triu
        offset = 1
    else:
        triangular = np.tril
        offset = -1

    return triangular(mx) - triangular(mx, offset).T


def extend_cost_mx(mx: np.ndarray, copy_n: int, to_copy_ix: int) -> np.ndarray:
    """
    Extends `mx` by copying `to_copy_ix`'th row and column `copy_n` times.
    """

    n_columns = np.repeat(np.array([mx[:, to_copy_ix]]).T, copy_n, axis=1)
    n_rows = np.repeat(np.array([mx[to_copy_ix, :]]).T, copy_n, axis=1).T
    copy_to_copy_costs = np.full(shape=(copy_n, copy_n), fill_value=mx[0, 0])

    # fmt: off
    return np.block([
        [copy_to_copy_costs, n_rows],
        [n_columns,          mx    ]
    ])
    # fmt: on
