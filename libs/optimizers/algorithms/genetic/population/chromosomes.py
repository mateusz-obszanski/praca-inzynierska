from typing import Generic, TypeVar

from .....solution import SolutionTSP


Gene = TypeVar("Gene")
Chromosome = Generic[Gene]
ChromosomeTSP = SolutionTSP
