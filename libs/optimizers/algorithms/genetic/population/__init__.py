"""
Chromosome population-oriented functionality.
"""


from typing import Sequence, TypeVar

from .chromosomes import ChromosomeTSP


Chromosome = TypeVar("Chromosome")
Population = Sequence[Chromosome]
PopulationTSP = list[ChromosomeTSP]
