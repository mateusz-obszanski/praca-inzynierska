from typing import Optional, TypeVar, Union
import itertools as it
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix as sci_distance_mx
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
Rng = TypeVar("Rng", bound=np.random.Generator)


def coords_random(
    n: Natural,
    rng: Rng,
    min_x: CoordT = 0,
    max_x: CoordT = 1,
    min_y: CoordT = 0,
    max_y: CoordT = 1,
) -> tuple[np.ndarray, Rng]:
    """
    Generates random 2D coordinates.
    """

    uniform = rng.uniform

    x = uniform(min_x, max_x, size=n)
    y = uniform(min_y, max_y, size=n)

    return np.dstack((x, y)), rng


def point_distances(
    coords: np.ndarray,
    rng: Rng,
    symmetric: bool = True,
    std_dev: Optional[float] = None,
    symmetricize_from_triu: bool = True,
) -> tuple[DistanceMx, Rng]:
    """
    Returns lengths beetween indices with given coords.
    If `symmetric` and `std_dev` is `None`, returns physical lengths else
    generates noise. If not `symmetric`, `std_dev` must be provided.
    """

    lengths = sci_distance_mx(coords, coords)

    if std_dev:
        noise = np.abs(rng.normal(scale=std_dev, size=2 * [coords.shape[0]]))
        lengths += noise

    if symmetric:
        lengths = symmetricize(lengths, symmetricize_from_triu)

    np.fill_diagonal(lengths, 0)

    return lengths, rng


def insert_at_random_indices(
    val: float,
    mx: DistanceMx,
    rng: Rng,
    insertion_p: float,
    symmetrically: bool = True,
    symmetricize_from_triu: bool = True,
    inplace: bool = False,
) -> tuple[DistanceMx, Rng]:

    vx_n = mx.shape[0]

    forbidden = rng.choice(
        [True, False],
        size=(vx_n, vx_n),
        p=[insertion_p, 1 - insertion_p],
    )

    forbidden[np.diag(vx_n * [True])] = True

    if symmetrically:
        forbidden = symmetricize(forbidden, symmetricize_from_triu)

    if not inplace:
        mx_disabled = np.empty_like(mx)
        allowed = ~forbidden
        mx_disabled[allowed] = mx[allowed]
    else:
        mx_disabled = mx

    mx[forbidden] = val

    return mx, rng


def wind_random(
    side_size: int,
    rng: Rng,
    distribution: Union[str, Distribution] = Distribution.UNIFORM,
    mean: float = 0,
    std_dev: float = 1,
    max_velocity: float = 1,
    antisymmetricize_from_triu: bool = True,
) -> tuple[WindMx, Rng]:
    """
    Generates wind along graph's edges. `max_velocity` is ignored for normal distribution.
    `std_dev` and `mean` are ignored for uniform distribution.
    """

    if isinstance(distribution, str):
        try:
            distribution = Distribution[distribution]
        except KeyError:
            raise NotImplementedError("distribution", distribution)

    shape = (side_size, side_size)

    distribution_map = {
        Distribution.NORMAL: lambda: rng.normal(mean, std_dev, shape),  # type: ignore
        Distribution.UNIFORM: lambda: rng.uniform(-max_velocity, max_velocity, shape),  # type: ignore
    }

    try:
        wind_generator = distribution_map[distribution]
    except KeyError:
        raise NotImplementedError("distribution", distribution.name)

    wind_mx = antisymmetricize(wind_generator(), antisymmetricize_from_triu)  # type: ignore
    np.fill_diagonal(wind_mx, 0)
    return wind_mx, rng


def effective_speed(speed: float, wind: np.ndarray) -> SpeedMx:
    """
    Wind should be antisymmetric.
    Negative values are instead set to 0.
    """

    effective_speed = speed + wind
    np.fill_diagonal(effective_speed, 0)
    effective_speed[effective_speed < 0] = 0

    return effective_speed


def coords_grid(n: int) -> np.ndarray:
    coord_range = list(range(n))

    x = list(it.chain.from_iterable(it.repeat(i, n) for i in range(n)))
    y = list(it.chain.from_iterable(it.repeat(coord_range, n)))

    return np.dstack((x, y))


def travel_times(
    distance_mx: DistanceMx, effective_speed: SpeedMx, disabled_val: float = -1
) -> TimeMx:
    any_speed = (distance_mx > 0) & (effective_speed > 0)
    travel_t = np.full_like(distance_mx, fill_value=disabled_val)
    travel_t[any_speed] = distance_mx[any_speed] / effective_speed[any_speed]

    return travel_t
