import sys
sys.path.append(".")
sys.path.append("..")

from enum import Enum, auto
from time import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy
import math
import pickle

import numpy as np
import pandas as pd
import more_itertools as mit
import matplotlib.pyplot as plt
import seaborn as sns
from rich.progress import Progress

from libs.environment import EnvironmentTSPSimple
from libs.environment.cost.tsp import CostT, TSPCostCalculatorSimple
from libs.optimizers.algorithms.genetic.chromosomes import (
    Chromosome,
    ChromosomeHomogenousVector,
)
from libs.optimizers.algorithms.genetic.population import Population
from libs.solution.initial_solution_generators.heuristic import (
    SolutionCreatorTSPSimpleHeuristicNN,
)
from libs.solution.initial_solution_generators.random import (
    SolutionCreatorTSPSimpleRandom,
)
from libs.optimizers.algorithms.genetic.population.parent_selectors import (
    ParentSelectorElitistRandomized,
)
from libs.optimizers.algorithms.genetic.population.population_selectors import (
    PopulationSelectorProbabilistic,
)
from libs.optimizers.algorithms.genetic.operators.mutations import (
    MutatorHomogenousVectorShuffle,
    MutatorHomogenousVectorSwap,
)
from libs.optimizers.algorithms.genetic.operators.crossovers import (
    CrossoverHomogenousVectorKPointPoisson,
)
from libs.optimizers.algorithms.genetic.operators.fixers import (
    ChromosomeFixerTSPSimple,
)
from libs.optimizers.algorithms.genetic.population.generators import (
    PopulationGenerator,
)
from libs.solution.representation import SolutionRepresentationTSP
from libs.environment.utils import (
    travel_times,
    coords_distances,
    disable_edges,
    effective_speed,
    wind_random,
    coords_random,
)


class GeneticAlgorithmFinishReason(Enum):
    MAX_ITERATIONS = auto()
    MAX_ITERATIONS_NO_UPDATE = auto()
    TIMEOUT = auto()


@dataclass
class GeneticAlgorithmExecutionData:
    best_chromosome: Chromosome
    best_cost: CostT
    initial_population: Population
    initial_best_cost: CostT
    end_population: Population
    iteration_best_costs: list[CostT]
    execution_time: float
    finish_reason: GeneticAlgorithmFinishReason
    last_iteration_n: int


def genetic_algorithm_tsp_simple_test(
    environment,
    population_size: int,
    max_iterations: int,
    max_iterations_no_update: int,
    timeout: float,
    swap_p: float,
    swap_lam: float,
    shuffle_p: float,
    shuffle_lam: float,
    crossover_lam: float,
    invalidity_weight: float,
    error_weight: float,
    cost_weight: float,
) -> GeneticAlgorithmExecutionData:
    solution_creator_heuristic = SolutionCreatorTSPSimpleHeuristicNN()
    solution_creator_random = SolutionCreatorTSPSimpleRandom()
    heuristic_solution = solution_creator_random.create(environment, initial_vx=0)
    random_solutions = [
        solution_creator_random.create(environment, initial_vx=0)
        for _ in range(population_size - 1)
    ]

    initial_population: Population = [
        ChromosomeHomogenousVector(sol.representation)
        for sol in mit.value_chain(heuristic_solution, random_solutions)
    ]

    cost_calculator = TSPCostCalculatorSimple()
    parent_selector = ParentSelectorElitistRandomized()
    population_selector = PopulationSelectorProbabilistic()
    mutators = [
        MutatorHomogenousVectorSwap(probability=swap_p, lam=swap_lam),
        MutatorHomogenousVectorShuffle(probability=shuffle_p, lam=shuffle_lam),
    ]
    crossover = CrossoverHomogenousVectorKPointPoisson(lam=crossover_lam)
    fixer = ChromosomeFixerTSPSimple()
    generation_generator = PopulationGenerator()

    current_population = deepcopy(initial_population)

    initial_best_chromosome = initial_population[0]

    initial_best_cost, _ = cost_calculator.calculate_total(
        SolutionRepresentationTSP(initial_best_chromosome.sequence), environment
    )

    t0 = time()

    for chromosome in initial_population[1:]:
        current_cost, _ = cost_calculator.calculate_total(
            SolutionRepresentationTSP(chromosome.sequence), environment
        )

        if current_cost < initial_best_cost:
            initial_best_chromosome = chromosome
            initial_best_cost = current_cost

    iteration_since_last_update = 0
    NaN = float("NaN")
    iteration_best_costs: list[CostT] = [NaN for _ in range(max_iterations)]
    best_chromosome = deepcopy(initial_best_chromosome)
    best_cost = initial_best_cost

    prev_update_t = time()

    with Progress() as progress:

        task_time = progress.add_task("[blue]Elapsed time...", total=timeout)
        task_iter_no_update = progress.add_task("[cyan]Iterations sice update...", total=max_iterations_no_update)
        task_iter = progress.add_task("[yellow]Iterations...", total=max_iterations)
        i = 0

        while i < max_iterations:
            delta_t = time() - t0

            if (reason_timeout := (delta_t > timeout)) or (
                reason_iterations := (
                    iteration_since_last_update > max_iterations_no_update
                )
            ):
                reason = (
                    GeneticAlgorithmFinishReason.TIMEOUT
                    if reason_timeout
                    else GeneticAlgorithmFinishReason.MAX_ITERATIONS_NO_UPDATE
                )

                return GeneticAlgorithmExecutionData(
                    best_chromosome,
                    best_cost,
                    initial_population,
                    initial_best_cost,
                    end_population=current_population,
                    iteration_best_costs=iteration_best_costs,
                    execution_time=delta_t,
                    finish_reason=reason,
                    last_iteration_n=i,
                )

            current_population, population_generation_data = generation_generator.generate(
                current_population,
                environment,
                cost_calculator,
                parent_selector,
                mutators,
                crossover,
                fixer,
                population_selector,
                invalidity_weight,
                error_weight,
                cost_weight,
            )

            costs = population_generation_data["new_costs"]

            current_best_chromosome = current_population[0]
            current_best_cost = costs[0]

            for chromosome, cost in zip(current_population[1:], costs[1:]):
                if cost < current_best_cost:
                    current_best_chromosome = chromosome
                    current_best_cost = cost

            iteration_best_costs[i] = current_best_cost

            if current_best_cost < best_cost:
                best_chromosome = current_best_chromosome
                best_cost = current_best_cost
                iteration_since_last_update = 0
            else:
                iteration_since_last_update += 1

            if int(delta_t - prev_update_t):
                progress.update(task_time, completed=delta_t)
                progress.update(task_iter_no_update, completed=iteration_since_last_update)
                progress.update(task_iter, completed=i)

                prev_update_t = time()

            i += 1

        progress.update(task_time, completed=time() - t0)
        progress.update(task_iter, completed=i)
        progress.update(task_iter_no_update, completed=iteration_since_last_update)

        return GeneticAlgorithmExecutionData(
            best_chromosome,
            best_cost,
            initial_population,
            initial_best_cost,
            end_population=current_population,
            iteration_best_costs=iteration_best_costs,
            execution_time=time() - t0,
            finish_reason=GeneticAlgorithmFinishReason.MAX_ITERATIONS,
            last_iteration_n=max_iterations - 1,
        )


def experiment_setup():
    coords = coords_random(5, max_x=10, max_y=10)
    distances = coords_distances(coords, std_dev=0.1)
    permitted_distances = disable_edges(distances, prohibition_p=0.1)
    wind = wind_random(permitted_distances, max_velocity=1)
    speed = 2.5
    eff_speed = effective_speed(speed, wind)
    travel_t = travel_times(distances, eff_speed)
    environment = EnvironmentTSPSimple(travel_t)

    solution_creator_heuristic = SolutionCreatorTSPSimpleHeuristicNN()
    heuristic_solution = solution_creator_heuristic.create(environment, initial_vx=0)
    cost_calculator = TSPCostCalculatorSimple()
    heuristic_cost = cost_calculator.calculate_total(heuristic_solution, environment)

    max_cost = max(filter(lambda x: x > 0 and math.isfinite(x), np.nditer(environment.cost)))  # type: ignore
    mean_cost = np.mean(
        [x for x in np.nditer(environment.cost) if x > 0 and math.isfinite(x)]
    )

    exp_args = {
        "environment": environment,
        "population_size": 20,
        "max_iterations": 100,
        "max_iterations_no_update": 20,
        "timeout": 10,
        "swap_p": 0.1,
        "swap_lam": 1,
        "shuffle_p": 0.05,
        "shuffle_lam": 1,
        "crossover_lam": 1,
        "invalidity_weight": 0.2 * mean_cost,
        "error_weight": 0.05 * mean_cost,
        "cost_weight": 1
    }

    other_info = {"heuristic_cost": heuristic_cost}

    return exp_args, other_info


def main():
    print("initializing environment")

    exp_args, other_info = experiment_setup()

    print("calling the algorithm")

    result = genetic_algorithm_tsp_simple_test(**exp_args)

    print(f"{other_info['heuristic_cost'] = }")
    print(
        (
            f"{result.finish_reason     = }\n"
            f"{result.execution_time    = }\n"
            f"{result.last_iteration_n  = }\n"
            f"{result.initial_best_cost = }\n"
            f"{result.best_cost         = }"
        )
    )

    now_str = datetime.now().strftime(r"%Y-%M-%d_%H-%m-%S")
    save_path = Path(f"experiment0_{now_str}.pickle")

    print(f"saving results as {save_path}")

    with save_path.open("wb") as f:
        pickle.dump(result, f)

    print("results saved")

    print("plotting")

    data = pd.DataFrame(
        data=result.iteration_best_costs,
        columns=["best cost"],
        # index=[("iteration", i) for i in range(len(result.iteration_best_costs))]
    )
    plt.figure()
    sns.set_theme()
    ax = sns.lineplot(data=data)
    _ = ax.set_title("Experiment details")
    plt.savefig(f"experiment0_{now_str}_plot.jpg")

    print("exiting")


if __name__ == "__main__":
    main()
