from abc import ABC, abstractmethod
from ..types import VehicleState, EdgeState, EnvironmentState, Time
from copy import copy


class StateTransitionExecutor(ABC):
    """
    Abstract base class.
    """

    @abstractmethod
    def state_transition(self, environment: EnvironmentState) -> EnvironmentState:
        ...


class STESimple(StateTransitionExecutor):
    def __effective_speed(self, vehicle: VehicleState, connection: EdgeState) -> float:
        return vehicle.base_speed + connection.wind_speed

    def __traversal_time(self, vehicle: VehicleState, connection: EdgeState) -> Time:
        return connection.length / self.__effective_speed(vehicle, connection)

    def state_transition(self, environment: EnvironmentState) -> EnvironmentState:
        new_environment = copy(EnvironmentState)
        new_environment.time = environment.time + self.traversal_time(
            environment.vehicle, connection
        )
        return new_environment
