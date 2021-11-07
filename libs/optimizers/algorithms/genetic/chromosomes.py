from abc import ABC
from dataclasses import dataclass


class Chromosome(ABC):
    """
    Abstract base class.
    """


@dataclass
class ChromosomeTSP(Chromosome):
    vertex_sequence: list[int]
