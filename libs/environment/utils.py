from typing import Optional, Union
import itertools as it
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix as sci_distance_mx
from copy import deepcopy
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
TimeMx = np.ndarray


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

    coords = np.stack([x, y], axis=1)

    return CoordsDF(coords, columns=["x", "y"])


def coords_distances(
    coords: CoordsDF,
    symmetric: bool = True,
    std_dev: Optional[float] = None,
    symmetricize_from_triu: bool = True,
) -> DistanceMx:
    """
    Returns lengths beetween indices with given coords.
    If `symmetric` and `std_dev` is `None`, returns physical lengths else generates noise.
    If not `symmetric`, `std_dev` must be provided.
    """

    coords_np = coords.to_numpy()
    lengths = sci_distance_mx(coords_np, coords_np)

    if std_dev:
        noise = np.abs(np.random.normal(scale=std_dev, size=2 * [coords_np.shape[0]]))
        lengths += noise

    if symmetric:
        lengths = symmetricize(lengths)

    np.fill_diagonal(lengths, 0)

    return lengths


def disable_edges(
    distance_mx: DistanceMx,
    prohibition_p: float,
    symmetrically: bool = True,
    disabled_val: float = -1,
    symmetricize_from_triu: bool = True,
    inplace: bool = False,
) -> DistanceMx:
    distance_mx = distance_mx if inplace else deepcopy(distance_mx)

    vx_n = distance_mx.shape[0]
    forbidden = np.random.choice(
        [True, False],
        size=(vx_n, vx_n),
        p=[prohibition_p, 1 - prohibition_p],
    )

    forbidden[np.diag(vx_n * [True])] = True
    if symmetrically:
        forbidden = symmetricize(forbidden, symmetricize_from_triu)

    distance_mx[forbidden] = disabled_val

    return distance_mx


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


def travel_times(distance_mx: DistanceMx, effective_speed: SpeedMx) -> TimeMx:
    any_speed = (distance_mx > 0) & (effective_speed > 0)
    travel_t = np.zeros_like(distance_mx)
    travel_t[any_speed] = distance_mx[any_speed] / effective_speed[any_speed]
    return travel_t
