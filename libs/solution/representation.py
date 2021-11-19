from abc import ABC
from dataclasses import dataclass
from typing import Any


class SolutionRepresentation(ABC):
    """
    Abstract base class.
    Contains `representation` field.
    The first entry in `representation` indicates initial point.
    """

    def __init__(self, representation) -> None:
        super().__init__()
        self.representation = representation


@dataclass
class SolutionRepresentationTSP(SolutionRepresentation):
    representation: list[int]
