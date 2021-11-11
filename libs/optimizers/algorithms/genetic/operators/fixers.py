from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TypeVar
from dataclasses import dataclass
from enum import Enum, auto
import itertools as it
import more_itertools as mit
import math

from libs.environment.utils import DistanceMx
from ..chromosomes import Chromosome, ChromosomeHomogenousVector
from .....utils.iteration import double_indices
from .....environment import Environment, EnvironmentTSPSimple


class FixStatus(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    NO_FIX_NEEDED = auto()


@dataclass
class FixResult:
    fix_status: FixStatus
    no_of_errors: int = 0

    def __post_init__(self) -> None:
        if self.fix_status in (FixStatus.SUCCESS, FixStatus.NO_FIX_NEEDED):
            if self.no_of_errors:
                raise ValueError(
                    (
                        "cannot set `fix_status` to `FixStatus.SUCCESS` nor"
                        "`FixStatus.NO_FIX_NEEDED` and give nonzero `no_of_errors`"
                    )
                )

            self.no_of_errors = 0


class ChromosomeFixer(ABC):
    ChromosomeT = TypeVar("ChromosomeT", bound=Chromosome)

    @abstractmethod
    def fix(
        self, chromosome: ChromosomeT, environment: Environment, inplace: bool = False
    ) -> tuple[ChromosomeT, FixStatus]:
        """
        Attempts to fix the chromosome. If it is not possible, sets success flag to False.
        """

        return (
            (chromosome, FixResult(FixStatus.SUCCESS))
            if inplace
            else (deepcopy(chromosome), FixResult(FixStatus.SUCCESS))
        )


class ChromosomeFixerTSPSimple(ChromosomeFixer):
    def fix(
        self,
        chromosome: ChromosomeHomogenousVector,
        environment: EnvironmentTSPSimple,
        inplace: bool = False,
    ) -> tuple[ChromosomeHomogenousVector, FixResult]:
        """
        Fixes only gene doubles.

        Fixes chromosome heuristically. Replaces every gene double with
        not present gene most suited based on the previous gene in sequence.
        """

        # TODO now fixes only doubles, doesn't fix nondouble impossible transitions
        # this can stay like this with assumption that previous, nonmutated
        # solution is ok and only mutations and crossovers introduce errors
        # (another function checking initial solutions?)

        chromosome, _ = super().fix(chromosome, environment, inplace)

        seq = chromosome.sequence
        doubles_mapping = double_indices(seq)
        cost_mx = environment.cost
        vx_num = cost_mx.shape[0]
        doubles = set(doubles_mapping.keys())

        transitions_possible = self.__valid_transitions(cost_mx, vx_num)

        if not doubles and all(transitions_possible):
            return chromosome, FixResult(FixStatus.SUCCESS)

        all_vxs = set(range(vx_num))

        leftout_vxs = all_vxs - doubles
        dbl_ixs = it.chain.from_iterable(doubles_mapping.values())

        for dbl_ix in dbl_ixs:
            prev_vx = seq[dbl_ix - 1]

            try:
                # if two vertices have the same cost, the smaller one (they are ints) is chosen
                _, best_replacement = min(
                    (cost, vx)
                    for vx in leftout_vxs
                    if (cost := cost_mx[prev_vx, vx]) > 0 and not math.isinf(cost)
                )
            except ValueError:
                # replacement is impossible (in place of current double)
                # it will be attempted later in place of the original gene

                continue

            previously_double = seq[dbl_ix]
            seq[dbl_ix] = best_replacement
            leftout_vxs.remove(best_replacement)
            doubles.remove(previously_double)

        if not leftout_vxs:
            no_of_errors = sum(
                not valid_transition
                for valid_transition in self.__valid_transitions(cost_mx, vx_num)
            )

            if no_of_errors == 0:
                return chromosome, FixResult(FixStatus.SUCCESS)

            return chromosome, FixResult(FixStatus.FAILURE, no_of_errors)

        # there were impossible replacements - attempt to replace in place
        # of the original genes

        all_vxs = list(all_vxs)

        for left_double in doubles:
            repl_ix = next(ix for ix, vx in enumerate(all_vxs) if vx == left_double)
            if repl_ix == 0:
                # the first element is the starting point - can not change that
                return chromosome, FixResult(FixStatus.FAILURE)

            prev_vx = repl_ix - 1

            try:
                _, best_replacement = min(
                    (cost, vx)
                    for vx in leftout_vxs
                    if (cost := cost_mx[prev_vx, vx]) > 0 and not math.isinf(cost)
                )
            except ValueError:
                # min got empty iterator - no possible replacements
                # doubles mapping has not been mutated so it is used
                no_of_errors = len(doubles_mapping) + sum(
                    not valid_transition
                    for valid_transition in self.__valid_transitions(cost_mx, vx_num)
                )
                return chromosome, FixResult(FixStatus.FAILURE, no_of_errors)

            seq[repl_ix] = best_replacement
            leftout_vxs.remove(best_replacement)

        return chromosome, FixResult(FixStatus.SUCCESS)

    def __valid_transitions(self, cost_mx: DistanceMx, vx_num: int):
        return (
            cost > 0 and not math.isinf(cost)
            for cost in (cost_mx[i, j] for i, j in mit.windowed(range(vx_num), n=2))
        )
