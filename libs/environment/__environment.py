from abc import ABC
from typing import Any
from dataclasses import dataclass
import numpy as np


Actor = Any
Background = Any
TimeMx = np.ndarray
DistanceMx = np.ndarray


class Environment(ABC):
    """
    Abstract base class.
    Groups information about environment.
    """

    cost: Any


class EnvironmentWithActor(Environment, ABC):
    """
    Abstract base class.
    """

    actor: Any


class EnvironmentStatic(Environment, ABC):
    """
    Abstract base class.
    Contains info about environment that does not change in time.
    """


class EnvironmentDynamic(Environment, ABC):
    """
    Abstract base class.
    Contains info about environment changing in time.
    """


class EnvironmentTSP(EnvironmentStatic, ABC):
    """
    Abstract base class.
    """


@dataclass
class EnvironmentTSPSimple(EnvironmentTSP):
    """
    Cost is a distance between graph vertices.
    """

    cost: DistanceMx
