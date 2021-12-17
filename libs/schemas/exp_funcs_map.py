from enum import Enum, auto
from typing import Callable

from libs.optimizers.algorithms.genetic.operators.mutations import (
    mutate_del,
    mutate_insert,
    mutate_insert_irp,
    mutate_swap,
)
from libs.optimizers.algorithms.genetic.operators.crossovers import (
    crossover_k_loci_ndarray,
    crossover_k_loci_poisson_ndarray,
    crossover_k_loci_poisson_with_inversion_ndarray,
    crossover_k_loci_with_inversion,
    crossover_ndarray,
)
from libs.optimizers.algorithms.genetic.operators.fixers import (
    fix_irp,
    fix_vrpp,
    fix_tsp,
)
from libs.environment.cost_calculators import (
    cost_calc_irp,
    cost_calc_vrpp,
    cost_calc_tsp,
    cost_calc_vrp,
)


class ExperimentType(Enum):
    TSP = auto()
    VRP = auto()
    SDVRP = auto()
    IRP = auto()


__TSP = ExperimentType.TSP
__VRP = ExperimentType.VRP
__SDVRP = ExperimentType.SDVRP
__IRP = ExperimentType.IRP

__mutators = "mutators"
__crossovers = "crossovers"
__cost_calcs = "cost_calcs"
__fixers = "fixers"


__TSP_CROSSOVERS = [
    crossover_ndarray,
    crossover_k_loci_ndarray,
    crossover_k_loci_poisson_ndarray,
    crossover_k_loci_with_inversion,
    crossover_k_loci_poisson_with_inversion_ndarray,
]
__TSP_MUTATORS = [mutate_swap]


EXP_ALLOWED_FUNCS: dict[ExperimentType, dict[str, list[Callable]]] = {
    __TSP: {
        __mutators: __TSP_MUTATORS,
        __crossovers: __TSP_CROSSOVERS,
        __cost_calcs: [cost_calc_tsp],
        __fixers: [fix_tsp],
    },
    __VRP: {
        __mutators: __TSP_MUTATORS,
        __crossovers: __TSP_CROSSOVERS,
        __cost_calcs: [cost_calc_vrp],
        __fixers: [fix_tsp],
    },
    __SDVRP: {
        __mutators: [*__TSP_MUTATORS, mutate_insert, mutate_del],
        __crossovers: __TSP_CROSSOVERS,
        __cost_calcs: [cost_calc_vrpp],
        __fixers: [fix_vrpp],
    },
    __IRP: {
        __mutators: [*__TSP_MUTATORS, mutate_insert_irp, mutate_del],
        __crossovers: __TSP_CROSSOVERS,
        __cost_calcs: [cost_calc_irp],
        __fixers: [fix_irp],
    },
}

TSP_MUTATORS = {"swap": mutate_swap}

del __TSP, __VRP, __SDVRP, __IRP
del __mutators, __crossovers, __cost_calcs, __fixers
del __TSP_CROSSOVERS, TSP_MUTATORS
