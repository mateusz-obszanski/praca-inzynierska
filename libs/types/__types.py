from enum import Enum, auto
from dataclasses import dataclass


Time = float


VehicleDecisions = type(NotImplemented)
"""
Endcodes heterogenous decision sequence for each vehicle simulable.
"""

# TODO implement functions for advancing vehicle and vertex actions in action selector
# TODO implement functionality that could be modified in the future (i.e. calc functions) as abstract classes for dependency injection
class VehicleActionSelector:
    """
    Selects next action from sequence of heterogenous actions encoded in VehicleDecisions
    based on vehicle simulable state (i.e. where it is, energy left etc.), available vertex
    connections' states and global environments state.
    """


class Distribution(Enum):
    UNIFORM = auto()
    NORMAL = auto()


@dataclass
class VehicleState:
    base_speed: float


@dataclass
class ConnectionState:
    length: float
    wind_speed: float


@dataclass
class VertexState:
    goods: float


@dataclass
class EnvironmentState:
    time: Time


@dataclass
class VehicleSimulable:
    decisions: Decisions
    current_action_selector: ActionSelector

