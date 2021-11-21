from typing import Callable, Generator

from ..libs.solution import SolutionTSP
from ..libs.optimizers.algorithms.genetic.population import PopulationTSP
from ..libs.environment.cost_calculators import CostGen, CostT, calculate_total_cost


BestSolution = SolutionTSP
BestCost = CostT
StepperTSP = Generator[tuple[BestSolution, BestCost], None, None]


def genetic_tsp_gen(
    population: PopulationTSP,
    cost_calculator: Callable[[SolutionTSP], CostGen],
    crossover: Callable[[SolutionTSP, SolutionTSP], tuple[SolutionTSP, SolutionTSP]],
    chromosome_fixer: Callable[[SolutionTSP], SolutionTSP],
    new_generation_creator: Callable[[PopulationTSP], PopulationTSP],
) -> StepperTSP:
    
