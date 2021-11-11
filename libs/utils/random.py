from collections import Sequence
import numpy as np
from numpy import random as np_rnd


def probabilities_by_value(values: Sequence[float]) -> list[float]:
    """
    The higher the value, the bigger its share.
    """

    probabilities = np.array(values, dtype=np.float64)
    probabilities /= probabilities.sum()

    return probabilities.astype(float).astype(float).tolist()
