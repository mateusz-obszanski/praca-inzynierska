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
