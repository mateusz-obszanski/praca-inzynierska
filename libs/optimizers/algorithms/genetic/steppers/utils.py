from dataclasses import dataclass
from typing import Any, Optional, TypeVar, Union
from collections.abc import Iterator, Iterable

import numpy as np

from libs.environment.cost_calculators import CostT
from libs.optimizers.algorithms.genetic.operators.mutations import Mutator


@dataclass(frozen=True)
class NextGenData:
    costs: list[CostT]
    no_of_fix_failures: int
    mutation_p: dict[Mutator, float]
    crossover_inv_p: Optional[float] = None


Rng = TypeVar("Rng", bound=np.random.Generator)
Chromosome = Union[np.ndarray, tuple[np.ndarray, np.ndarray]]
Population = list[Chromosome]
ExpStepper = Iterator[tuple[Population, NextGenData, Rng]]


def apply_mutators(
    c: np.ndarray,
    mutators: Iterable[Mutator],
    mut_ps: dict[Mutator, float],
    mut_kwargs: Iterable[dict[str, Any]],
    rng: Rng,
) -> tuple[np.ndarray, Rng]:
    for m, kwds in zip(mutators, mut_kwargs):
        c, rng = m(c, mut_ps[m], rng, **kwds)
    return c, rng
