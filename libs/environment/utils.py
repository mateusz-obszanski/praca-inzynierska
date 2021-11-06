from typing import Optional, Union
import itertools as it
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from ..types import Distribution
from ..utils import symmetricize, antisymmetricize


CoordT = float
CoordsDF = pd.DataFrame
"""fields: x, y"""
DistanceMx = np.ndarray
Natural = int
WindMx = np.ndarray
"""antisymmetric matrix of vehicle's wind speed boost"""
SpeedMx = np.ndarray


def coords_random(
    n: Natural,
    min_x: CoordT = 0,
    max_x: CoordT = 1,
    min_y: CoordT = 0,
    max_y: CoordT = 1,
) -> CoordsDF:
    """
    Generates random 2D coordinates.
    """

    r_gen = np.random.uniform

    x = r_gen(min_x, max_x, size=n)
    y = r_gen(min_y, max_y, size=n)

    coords = np.stack([x ,y], axis=1)

    return CoordsDF(coords, columns=["x", "y"])


def coords_distances(
    coords: CoordsDF,
    symmetric: bool = True,
    prohibition_p: float = 0,
    std_dev: Optional[float] = None,
    forbidden_val: float = -1,
    symmetricize_from_triu: bool = True
) -> DistanceMx:
    """
    Returns lengths beetween indices with given coords. If `symmetric`
    and `std_dev` is `None`, returns physical lengths else generates noise.
    If not `symmetric`, `std_dev` must be provided.
    """

    forbidden = np.random.choice(  # type: ignore
        [True, False],
        size=2 * [coords.shape[0]],
        p=[prohibition_p, 1 - prohibition_p],
    )

    forbidden[np.diag(coords.shape[0] * [True])] = True

    coords_np = coords.to_numpy()
    lengths = distance_matrix(coords_np, coords_np)

    if std_dev:
        if not symmetric:
            raise ValueError("If `symmetric`, `std_dev` must be given")
        noise = np.abs(np.random.normal(scale=std_dev, size=2 * [coords.shape[0]]))
        lengths += np.triu(noise, 1) + np.tril(noise, -1)

        lengths = symmetricize(lengths, symmetricize_from_triu)
        forbidden = symmetricize(forbidden, symmetricize_from_triu)

        lengths[forbidden] = forbidden_val

        return lengths

    if symmetric:
        lengths = symmetricize(lengths)
        forbidden = symmetricize(forbidden)

    lengths[forbidden] = forbidden_val

    return lengths


def wind_random(
    lengths: DistanceMx,
    distribution: Union[str, Distribution] = Distribution.UNIFORM,
    mean: float = 0,
    std_dev: float = 1,
    max_velocity: float = 1,
) -> WindMx:
    """
    Generates wind along graph's edges. `max_velocity` is ignored for normal distribution.
    `std_dev` and `mean` are ignored for uniform distribution.
    """

    if isinstance(distribution, str):
        try:
            distribution = Distribution[distribution]
        except KeyError:
            raise NotImplementedError("distribution", distribution)

    distribution_map = {
        Distribution.NORMAL: lambda: np.random.normal(mean, std_dev, lengths.shape),  # type: ignore
        Distribution.UNIFORM: lambda: np.random.uniform(-max_velocity, max_velocity, lengths.shape),  # type: ignore
    }

    try:
        wind_generator = distribution_map[distribution]
    except KeyError:
        raise NotImplementedError("distribution", distribution.name)

    wind_mx = antisymmetricize(wind_generator())  # type: ignore
    np.fill_diagonal(wind_mx, 0)
    return wind_mx


def effective_speed(speed: float, wind) -> SpeedMx:
    """
    Wind should be antisymmetric.
    Negative values are instead set to 0.
    """
    effective_speed = speed + wind
    np.fill_diagonal(effective_speed, 0)
    effective_speed[effective_speed < 0] = 0
    return effective_speed


def coords_grid(n: int) -> CoordsDF:
    coord_range = list(range(n))
    x = list(it.chain.from_iterable(it.repeat(i, n) for i in range(n)))
    y = list(it.chain.from_iterable(it.repeat(coord_range, n)))
    return CoordsDF({"x": x, "y": y})
