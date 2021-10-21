from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any, Protocol, Union, runtime_checkable
from math import pi


CoordT = Union[int, float]


class Coords(Iterator):
    def __init__(self, x: CoordT, *args: CoordT, immutable: bool = False) -> None:
        super().__init__()
        self.immutable = immutable
        self._coords = [x, *args]

    def __iter__(self) -> Iterator[CoordT]:
        return iter(self._coords)

    @property
    def x(self) -> CoordT:
        return self._coords[0]

    @x.setter
    def x(self, v: CoordT) -> None:
        if self.immutable:
            self._coords[0] = v
            return

        raise MutatingImmutableCoordsError("x")

    @property
    def y(self) -> CoordT:
        return self._coords[1]

    @y.setter
    def y(self, v: CoordT) -> None:
        if self.immutable:
            self._coords[1] = v
            return

        raise MutatingImmutableCoordsError("x")

    @property
    def z(self) -> CoordT:
        return self._coords[2]

    @z.setter
    def z(self, v: CoordT) -> None:
        if self.immutable:
            self._coords[2] = v
            return

        raise MutatingImmutableCoordsError("x")


class CoordsError(Exception):
    """
    Abstract coords error.
    """


class MutatingImmutableCoordsError(CoordsError):
    ...


@dataclass
class PolarCoords2D:
    def __init__(self, angle: CoordT, radius: CoordT) -> None:
        """
        angle in radians 0 <= angle <= 2*pi
        """
        assert 0 <= angle <= 2 * pi


@runtime_checkable
class Simulable(Protocol):
    def simulate(self, *args, **kwargs) -> Any:
        ...
