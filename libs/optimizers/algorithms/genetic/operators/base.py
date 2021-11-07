from abc import ABC, abstractmethod
from typing import Union

from ..chromosomes import Chromosome


class GeneticOperator(ABC):
    """
    Abstract base class
    """

    @abstractmethod
    def execute(self) -> Union[Chromosome, tuple[Chromosome, ...]]:
        ...
