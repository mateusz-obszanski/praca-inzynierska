from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import Iterable
from typing import Any


Actor = Any
Background = Any


@dataclass
class Environment(ABC):
    background: Background
    actors: Iterable[Actor]
