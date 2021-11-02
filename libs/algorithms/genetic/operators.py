"""
Mutation and crossover operators for genetic algorithm.
"""


from abc import ABC, abstractmethod
from .chromosomes import Chromosome


class Mutation(ABC):
    @staticmethod
    @abstractmethod
    def __call__(chromosome: Chromosome, *args, **kwargs) -> Chromosome:
        ...


class Crossover(ABC):
    @staticmethod
    @abstractmethod
    def __call__(
        chromosome: Chromosome, k: int = 1, *args, **kwargs
    ) -> tuple[Chromosome, ...]:
        """
        Performs k-point crossover between chromosomes.
        """
        # Problem - crossover jest losowy, mutacja pewnych
        # fragmentów chromosomu jest nieopłacalna. Wiadomo
        # o tym dopiero po wielu iteracjach
        # Rozwiązanie - adaptacja. Zapamiętuj, gdzie były
        # zmiany i czy dana zmiana poprawiła wynik.
        # Od ustalonej iteracji losuj punkty crossoveru
        # i mutacji z updatowanym rozkładem prawdopodobieństwa
        # opłacalności...
