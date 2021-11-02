from collections import Iterable
from typing import Optional, TypeVar
import numpy as np
from ..types import Distribution


T = TypeVar("T")


CoordT = float
Coords2D = tuple[CoordT, CoordT]
Coords2DList = list[Coords2D]
Natural = int


def coords_generator(
    n: Natural,
    min_x: CoordT = 0,
    max_x: CoordT = 1,
    min_y: CoordT = 0,
    max_y: CoordT = 1,
) -> Coords2DList:
    """
    Generates random 2D coordinates.
    """

    xs = np.random.uniform(min_x, max_x, n)
    ys = np.random.uniform(min_y, max_y, n)
    return Coords2DList(zip(xs, ys))


def random_lengths(
    coords: Coords2DList,
    symmetric: bool = True,
    prohibition_p: float = 0,
    std_dev: Optional[float] = None,
) -> np.ndarray:
    """
    Returns lengths beetween indices with given coords. If `symmetric`
    and `std_dev` is `None`, returns physical lengths else generates noise.
    If not `symmetric`, `std_dev` must be provided.
    """

    forbidden = np.random.choice(  # type: ignore
        [True, False],
        size=(len(coords), len(coords)),
        p=[prohibition_p, 1 - prohibition_p],
    )

    forbidden[np.diag(len(coords) * [True])] = True

    lengths = np.array(
        [
            [np.linalg.norm(np.array(c1) - np.array(c2)) for c1 in coords]
            for c2 in coords
        ]
    )

    if std_dev:
        if not symmetric:
            raise ValueError("If `symmetric`, `std_dev` must be given")
        lengths += np.random.normal(scale=std_dev, size=(len(coords), len(coords)))

    if symmetric:
        lengths = np.triu(lengths, 1) + np.diag(lengths) + np.tril(lengths, -1)
        forbidden = np.triu(forbidden, 1) + np.diag(forbidden) + np.tril(forbidden, -1)

    lengths[forbidden] = float("inf")

    return lengths


def random_wind(
    lengths: np.ndarray,
    distribution: Distribution,
    mean: Optional[float] = None,
    std_dev: Optional[float] = None,
    max_velocity: Optional[float] = None,
) -> np.ndarray:
    """
    Generates wind along graph's edges. `max_velocity` is ignored for normal distribution.
    `std_dev` and `mean` are ignored for uniform distribution.
    """

    distribution_map = {
        Distribution.NORMAL: lambda: np.random.normal(mean, std_dev, lengths.shape),  # type: ignore
        Distribution.UNIFORM: lambda: np.random.uniform(-max_velocity, max_velocity, lengths.shape),  # type: ignore
    }

    wind_generator = distribution_map.get(distribution, NotImplemented)
    return wind_generator()  # type: ignore
