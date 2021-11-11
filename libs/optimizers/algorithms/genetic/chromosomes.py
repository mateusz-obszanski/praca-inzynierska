from abc import ABC
from dataclasses import dataclass


class Chromosome(ABC):
    """
    Abstract base class.
    """

    def __init__(self, sequence) -> None:
        super().__init__()
        self.sequence = sequence


@dataclass
class ChromosomeHomogenousVector(Chromosome):
    sequence: list[int]
