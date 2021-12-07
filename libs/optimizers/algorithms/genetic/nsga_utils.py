"""
Based on: https://github.com/baopng/NSGA-II

Authors:
- Pham Ngo Gia Bao, Ho Chi Minh University of Technology
- Tram Loi Quan, Ho Chi Minh University of Technology
- A/Prof. Quan Thanh Tho, Ho Chi Minh University of Technology (advisor)
- A/Prof. Akhil Garg, Shantou University (advisor)

Original licence:

Copyright (c) 2018 Authors (noted in README.md)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from typing import Callable, Generator, Optional, TypeVar
from collections.abc import Iterable
from copy import deepcopy

import numpy as np


Rng = TypeVar("Rng", bound=np.random.Generator)


class NSGAPopulation(list):
    def __init__(self, __iterable: Optional[Iterable] = None):
        if __iterable is None:
            super().__init__()
        else:
            super().__init__(__iterable)
        self.fronts: list[list[Individual]] = []


class Individual(object):
    def __init__(self):
        self.rank: Optional[int] = None
        self.crowding_distance: Optional[float] = None
        self.domination_count: Optional[int] = None
        self.dominated_solutions: Optional[list["Individual"]] = None
        self.features = None
        self.objective_vals: Optional[list[float]] = None

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.features == other.features
        return False

    def calculate_objectives(
        self, obj_funcs: Iterable[Callable], unpack_args: bool = False
    ):
        if unpack_args:
            self.objective_vals = [f(*self.features) for f in obj_funcs]  # type: ignore
        else:
            self.objective_vals = [f(self.features) for f in obj_funcs]

    def dominates(self, other: "Individual"):
        assert self.objective_vals is not None and other.objective_vals is not None
        and_condition = True
        or_condition = False
        for first, second in zip(self.objective_vals, other.objective_vals):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
        return and_condition and or_condition


def fast_nondominated_sort(
    population: NSGAPopulation, inplace: bool = False
) -> NSGAPopulation:
    if not inplace:
        population = NSGAPopulation(population)
    population.fronts = [[]]
    assert len(population) >= 0
    for individual in population:
        individual.domination_count = 0
        individual.dominated_solutions = []
        for other_individual in population:
            if individual.dominates(other_individual):
                individual.dominated_solutions.append(other_individual)
            elif other_individual.dominates(individual):
                individual.domination_count += 1
        if individual.domination_count == 0:
            individual.rank = 0
            population.fronts[0].append(individual)
    i = 0
    while len(population.fronts[i]) > 0:
        temp = []
        for individual in population.fronts[i]:
            for other_individual in individual.dominated_solutions:  # type: ignore
                other_individual.domination_count -= 1  # type: ignore
                if other_individual.domination_count == 0:
                    other_individual.rank = i + 1
                    temp.append(other_individual)
        i = i + 1
        population.fronts.append(temp)
    return population


def calculate_crowding_distance(
    front: list[Individual], inplace: bool = False
) -> list[Individual]:
    if not inplace:
        front = deepcopy(front)

    if len(front) <= 0:
        return front

    solutions_num = len(front)
    for individual in front:
        individual.crowding_distance = 0

    for m in range(len(front[0].objective_vals)):  # type: ignore
        front.sort(key=lambda individual: individual.objective_vals[m])  # type: ignore
        front[0].crowding_distance = 10 ** 9
        front[solutions_num - 1].crowding_distance = 10 ** 9
        m_values = [individual.objective_vals[m] for individual in front]  # type: ignore
        scale = max(m_values) - min(m_values)
        if scale == 0:
            scale = 1
        for i in range(1, solutions_num - 1):
            front[i].crowding_distance += (  # type: ignore
                front[i + 1].objective_vals[m] - front[i - 1].objective_vals[m]  # type: ignore
            ) / scale

    return front


def crowding_operator(individual: Individual, other_individual: Individual) -> bool:
    if (individual.rank < other_individual.rank) or (  # type: ignore
        (individual.rank == other_individual.rank)
        and (individual.crowding_distance > other_individual.crowding_distance)  # type: ignore
    ):
        return True
    else:
        return False


def tournament(
    population: list[Individual], num_of_tour_particips: int, tour_p: float, rng: Rng
) -> tuple[Individual, Rng]:
    participants_iter = iter(rng.choice(population, size=num_of_tour_particips))
    best = next(participants_iter)
    for participant in participants_iter:
        # avoid crowding, choose with probability
        if crowding_operator(participant, best) and rng.random() > tour_p:
            best = participant

    return best, rng


def create_children(
    population: NSGAPopulation,
    num_of_tour_particips: int,
    tour_p: float,
    rng: Rng,
) -> tuple[list[Individual], Rng]:
    # TODO mutators, objective calculators, crossovers
    # TODO fixing
    children = []
    while len(children) < len(population):
        parent1, rng = tournament(population, num_of_tour_particips, tour_p, rng)
        parent2 = parent1
        while parent1 == parent2:
            parent2 = tournament(population, num_of_tour_particips, tour_p, rng)
        # TODO crossover -> mutation -> fixing, if fail, mutate parents -> fixing, if fail -> return original parents
        child1, child2 = __crossover(parent1, parent2)
        __mutate(child1)
        __mutate(child2)
        # TODO vv those are Individuals
        child1.calculate_objectives()
        child2.calculate_objectives()
        children.append(child1)
        children.append(child2)

    return children, rng


def evolver(
    initial_pop: NSGAPopulation, population_num: int, inplace: bool = False
) -> Generator[list[Individual], None, None]:
    """
    Care about returned_population.fronts[0] - this is THE solution
    """
    # TODO yield additional data about fixing etc
    population = fast_nondominated_sort(initial_pop, inplace=inplace)
    for i, front in enumerate(population.fronts):
        population.fronts[i] = calculate_crowding_distance(front, inplace=inplace)
    children = create_children(initial_pop)
    returned_population = None
    while True:
        population.extend(children)
        population = fast_nondominated_sort(population, inplace=inplace)
        new_population = NSGAPopulation()
        front_num = 0
        while len(new_population) + len(population.fronts[front_num]) <= population_num:
            calculate_crowding_distance(population.fronts[front_num])
            new_population.extend(population.fronts[front_num])
            front_num += 1
        calculate_crowding_distance(population.fronts[front_num])
        population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)  # type: ignore
        new_population.extend(
            population.fronts[front_num][0 : population_num - len(new_population)]
        )
        returned_population = population
        yield returned_population
        population = new_population
        population = fast_nondominated_sort(population, inplace=inplace)
        for i, front in enumerate(population.fronts):
            population.fronts[i] = calculate_crowding_distance(front)
        children = create_children(population)
