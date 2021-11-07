from collections import Iterable
from typing import Any, Protocol
import numpy as np


Actor = Any
Background = Any
TimeMx = np.ndarray
DistanceMx = np.ndarray


class Environment(Protocol):
    """
    Groups information about environment.
    """

    cost: Any


class EnvironmentWithActor(Environment):
    actor: Any


class EnvironmentStatic(Environment):
    """
    Contains info about environment that does not change in time.
    """


class EnvironmentDynamic(Environment):
    """
    Contains info about environment changing in time.
    """


class EnvironmentTSP(EnvironmentStatic):
    ...


class EnvironmentTSPSimple(EnvironmentTSP):
    """
    Cost is a distance between graph vertices.
    """

    cost: DistanceMx
