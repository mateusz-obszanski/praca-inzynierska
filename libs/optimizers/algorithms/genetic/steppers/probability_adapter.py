from collections.abc import Iterable
from collections import deque
from typing import Any, Union
from statistics import mean
from math import isfinite

from libs.optimizers.algorithms.genetic.operators.mutations import Mutator


MutProbs = dict[Mutator, float]
CrossKwargs = dict[str, Any]


class ProbAdapter:
    def __init__(
        self,
        mut_ps: MutProbs,
        crossover_kwargs: CrossKwargs,
        adaptation_period: int,
        step: float,
    ) -> None:
        """
        `adaptation_period` - no. of iterations after which adaptation occurs.
        """
        self.queue: deque[Union[Mutator, str, None]] = deque(
            (
                *mut_ps.keys(),
                *(("crossover",) if "inversion_p" in crossover_kwargs else ()),
            )
        )
        to_fill = adaptation_period - len(self.queue)
        if to_fill > 0:
            self.queue.extend(None for _ in range(to_fill))
        self.mut_ps = mut_ps
        self.crossover_kwargs = crossover_kwargs
        self.adaptation_period = adaptation_period
        self.step = step

    def adapt(
        self,
        parent_costs: Iterable[float],
        offspring_costs: Iterable[float],
    ) -> tuple[MutProbs, CrossKwargs]:
        """
        Analyzes improvement or lack thereof of mean parents' and offspring's costs
        in each iteration and handles tweaking the probabilities for mutation
        occurence and crossover inversion.
        """

        finite_parent_costs = tuple(c for c in parent_costs if isfinite(c))
        finite_offspring_costs = tuple(c for c in offspring_costs if isfinite(c))
        if not finite_parent_costs:
            if not finite_offspring_costs:
                return self.mut_ps, self.crossover_kwargs
            # parents were invalid, but current probabilities made offspring viable
            # forcibly set ratio to make the next probability in the queue bigger
            ratio = 1
        elif not finite_offspring_costs:
            # offspring is invalid with current settings
            ratio = -1
        else:
            ratio = mean(finite_parent_costs) / mean(finite_offspring_costs)
        if ratio > 1:
            return self._modify_current(add=True)
        elif ratio < 1:
            return self._modify_current(add=False)
        else:
            return self.mut_ps, self.crossover_kwargs

    def _modify_current(self, add: bool) -> tuple[MutProbs, CrossKwargs]:
        """
        If `add`, adds `self.step` to the next probability in the queue, else subtracts.
        """

        next_to_mod = self.queue[0]
        self.queue.rotate()
        to_add = self.step * (1 if add else -1)
        if next_to_mod == "crossover":
            self.crossover_kwargs["inversion_p"] += to_add
        elif next_to_mod is not None:
            self.mut_ps[next_to_mod] += to_add  # type: ignore
        return self.mut_ps, self.crossover_kwargs
