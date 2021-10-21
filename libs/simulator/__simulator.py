from dataclasses import dataclass, field
from functools import lru_cache


@dataclass
class City:
    population: float


@dataclass
class Direction:
    from_point: City
    to_point: City


@dataclass
class Wind:
    direction: Direction
    speed: float  # [m/s]


@dataclass
class AirRoute:
    length: float
    wind: Wind


@dataclass
class Simulator:
    def simulate(self):
        ...


class SimulationVisualizer:
    def visualize(self):
        ...
