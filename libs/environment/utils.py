import math
from collections.abc import Sequence, Iterable
from typing import Generator, Optional, TypeVar, Union
import itertools as it

import more_itertools as mit
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix as sci_distance_mx

from libs.types import Distribution
from libs.utils.matrix import symmetricize, antisymmetricize
from libs.utils.iteration import find_all_occurence_indices


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

    return np.stack((x, y), axis=1), rng


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

    to_insert = rng.random(size=(vx_n, vx_n)) < insertion_p

    to_insert[np.diag(vx_n * [True])] = True

    if symmetrically:
        to_insert = symmetricize(to_insert, symmetricize_from_triu)

    if not inplace:
        mx_inserted = np.empty_like(mx)
        allowed = ~to_insert
        mx_inserted[allowed] = mx[allowed]
    else:
        mx_inserted = mx

    mx_inserted[to_insert] = val

    return mx_inserted, rng


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


def coords_grid(n: int, spacing: float = 1) -> np.ndarray:
    coord_range = np.arange(0, n * spacing, spacing).tolist()

    x = list(it.chain.from_iterable(it.repeat(cx, n) for cx in coord_range))
    y = list(it.chain.from_iterable(it.repeat(coord_range, n)))

    return np.stack((x, y), axis=1)


def travel_times(
    distance_mx: DistanceMx, effective_speed: SpeedMx, disabled_val: float = -1
) -> TimeMx:
    any_speed = (
        (distance_mx > 0)
        & (np.isfinite(distance_mx))
        & (distance_mx != disabled_val)
        & (effective_speed > 0)
        & (np.isfinite(effective_speed))
        & (effective_speed != disabled_val)
    )
    travel_t = np.full_like(distance_mx, fill_value=disabled_val)
    travel_t[any_speed] = distance_mx[any_speed] / effective_speed[any_speed]

    return travel_t


def find_invalid_transitions(
    solution: Iterable[int], cost_mx: np.ndarray, forbidden_val: float = -1
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Returns list of tuples ((a, b), (index of a in solution, index of b in solution)).
    """

    # FIXME remove
    print(f"find invalid transitions {solution = }")

    return [
        ((a, b), (i, j))
        for (i, a), (j, b) in mit.windowed(enumerate(solution), n=2)  # type: ignore
        if not check_transition(a, b, cost_mx, forbidden_val)  # type: ignore
    ]


def find_invalid_transitions_vrpp(
    solution: Iterable[int], cost_mx: np.ndarray, forbidden_val: float, fillval: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Returns list of tuples ((a, b), (index of a in solution, index of b in solution)).
    Ignores transitions, where either source or destination equals fillval.
    """
    return [
        ((a, b), (i, j))
        for (i, a), (j, b) in mit.windowed(enumerate(solution), n=2)  # type: ignore
        if a != fillval != b and not check_transition(a, b, cost_mx, forbidden_val)  # type: ignore
    ]


def check_transition(
    a: int, b: int, dist_mx: np.ndarray, forbidden_val: float = -1
) -> bool:
    # FIXME remove
    try:
        cost = dist_mx[a, b]
    except IndexError as e:
        print(f"{a = }({type(a)}), {b = }({type(b)})")
        raise e
    return math.isfinite(cost) and cost != forbidden_val


def check_transitions(
    seq: Sequence[int],
    dist_mx: np.ndarray,
    forbid_val: float = -1,
) -> bool:
    return all(check_transitions_iter(seq, dist_mx, forbid_val))


def check_transitions_iter(
    seq: Sequence[int],
    dist_mx: np.ndarray,
    forbid_val: float = -1,
) -> Generator[bool, None, None]:
    return (
        check_transition(i, j, dist_mx, forbid_val) for i, j in mit.windowed(seq, n=2)  # type: ignore
    )


def check_doubles(seq: Sequence[int], initial_ix: int = 0):
    return all(
        (len(occs) == 1) if vx != initial_ix else True
        for vx, occs in find_all_occurence_indices(seq).items()
    )
