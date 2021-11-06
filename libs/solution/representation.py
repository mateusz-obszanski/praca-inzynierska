from abc import ABC
from dataclasses import dataclass
from typing import Protocol, Any


class SolutionRepresentation(Protocol):
    """
    Contains `representation` field.
    The first entry in `representation` indicates initial point.
    """

    representation: Any


@dataclass
class SolutionRepresentationTSP:
    representation: list[int]
