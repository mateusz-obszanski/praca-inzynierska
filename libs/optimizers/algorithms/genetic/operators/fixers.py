from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TypeVar, Optional
import itertools as it
import math
from ..chromosomes import Chromosome, ChromosomeHomogenousVector
from .....utils.iteration import double_indices
from .....environment import Environment, EnvironmentTSPSimple


SuccessFlag = bool
SUCCESS = True
FAILURE = False


class ChromosomeFixer(ABC):
    ChromosomeT = TypeVar("ChromosomeT", bound=Chromosome)

    @abstractmethod
    def fix(
        self, chromosome: ChromosomeT, environment: Environment, inplace: bool = False
    ) -> tuple[ChromosomeT, SuccessFlag]:
        """
        Attempts to fix the chromosome. If it is not possible, sets success flag to False.
        """
        return (chromosome, SUCCESS) if inplace else (deepcopy(chromosome), SUCCESS)


class ChromosomeFixerTSPSimple(ChromosomeFixer):
    def fix(
        self,
        chromosome: ChromosomeHomogenousVector,
        environment: EnvironmentTSPSimple,
        inplace: bool = False,
    ) -> tuple[ChromosomeHomogenousVector, SuccessFlag]:
        """
        Fixes chromosome heuristically. Replaces every gene double with
        not present gene most suited based on the previous gene in sequence.
        """

        chromosome, _ = super().fix(chromosome, environment, inplace)

        seq = chromosome.vertex_sequence
        doubles_mapping = double_indices(seq)
        cost_mx = environment.cost
        vx_num = cost_mx.shape[0]
        all_vxs = set(range(vx_num))
        doubles = set(doubles_mapping.keys())
        not_present_vxs = all_vxs - doubles
        dbl_ixs = it.chain.from_iterable(doubles_mapping.values())

        for dbl_ix in dbl_ixs:
            prev_vx = seq[dbl_ix - 1]

            try:
                # if two vertices have the same cost, the smaller one (they are ints) is chosen
                _, best_replacement = min(
                    (cost, vx)
                    for vx in not_present_vxs
                    if (cost := cost_mx[prev_vx, vx]) > 0 and not math.isinf(cost)
                )
            except ValueError:
                # replacement is impossible (in place of current double)
                # it will be attempted later in place of the original gene

                continue

            previously_double = seq[dbl_ix]
            seq[dbl_ix] = best_replacement
            not_present_vxs.remove(best_replacement)
            doubles.remove(previously_double)

        if not not_present_vxs:
            return chromosome, SUCCESS

        # there were impossible replacements - attempt to replace in place
        # of the original genes

        all_vxs = list(all_vxs)

        for left_double in doubles:
            repl_ix = next(ix for ix, vx in enumerate(all_vxs) if vx == left_double)
            if repl_ix == 0:
                # the first element is the starting point - can not change that
                return chromosome, FAILURE

            prev_vx = repl_ix - 1

            try:
                _, best_replacement = min(
                    (cost, vx)
                    for vx in not_present_vxs
                    if (cost := cost_mx[prev_vx, vx]) > 0 and not math.isinf(cost)
                )
            except ValueError:
                return chromosome, FAILURE

            seq[repl_ix] = best_replacement
            not_present_vxs.remove(best_replacement)

        return chromosome, SUCCESS
