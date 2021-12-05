from typing import Generic, TypeVar

from libs.solution import SolutionTSP


Gene = TypeVar("Gene")
Chromosome = Generic[Gene]
ChromosomeTSP = SolutionTSP
