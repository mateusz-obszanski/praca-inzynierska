from queue import PriorityQueue
from dataclasses import dataclass
from . import Time


@dataclass
class Event:
    start_t: Time
    end_t: Time


class EventQueue:
    def __init__(self) -> None:
        self.__queue: PriorityQueue[Event] = PriorityQueue()
