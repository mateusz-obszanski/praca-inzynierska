from abc import ABC, abstractmethod

from .base import GeneticOperator
from ..chromosomes import Chromosome
from .....types import PositiveInt


class Crossover(GeneticOperator, ABC):
    """
    Abstract base class.
    """

    @abstractmethod
    def execute(
        self, chromosome1: Chromosome, chromosome2: Chromosome
    ) -> tuple[Chromosome, ...]:
        ...

        # Problem - crossover jest losowy, mutacja pewnych
        # fragmentów chromosomu jest nieopłacalna. Wiadomo
        # o tym dopiero po wielu iteracjach
        # Rozwiązanie - adaptacja. Zapamiętuj, gdzie były
        # zmiany i czy dana zmiana poprawiła wynik.
        # Od ustalonej iteracji losuj punkty crossoveru
        # i mutacji z updatowanym rozkładem prawdopodobieństwa
        # opłacalności...


class CrossoverKPoint(Crossover, ABC):
    """
    Abstract base class.
    """

    @abstractmethod
    def execute(
        self, chromosome1: Chromosome, chromosome2: Chromosome, k: PositiveInt = 1
    ) -> tuple[Chromosome, ...]:
        ...
