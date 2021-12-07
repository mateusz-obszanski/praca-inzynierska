from collections.abc import Sequence
from typing import TypeVar

import more_itertools as mit
import numpy as np


T = TypeVar("T")
Rng = TypeVar("Rng", bound=np.random.Generator)


def probabilities_by_value(values: Sequence[float]) -> list[float]:
    """
    The higher the value, the bigger its share.
    """

    probabilities = np.array(values, dtype=np.float64)
    probabilities /= probabilities.sum()

    return probabilities.astype(float).astype(float).tolist()


def randomly_chunkify(
    seq: Sequence[T], p: float, rng: Rng
) -> tuple[list[Sequence[T]], Rng]:
    """
    Chunkifies `seq` by cutting at loci generated with probability `p`.
    """

    assert 0 <= p <= 1

    seq_len = len(seq)
    # x1 <cut> x2 <nocut> ... <nocut> xn
    cuts = rng.random(size=seq_len - 1) < p
    cut_loci = np.arange(1, seq_len)[cuts]
    chunk_ranges = mit.windowed(mit.value_chain(0, cut_loci, seq_len), n=2)
    return [seq[i:j] for i, j in chunk_ranges], rng
