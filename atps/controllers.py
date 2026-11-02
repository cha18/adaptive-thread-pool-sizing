from abc import ABC, abstractmethod
import math

from atps.stats import Snapshot


class Controller(ABC):
    @abstractmethod
    def decide(self, snapshot: Snapshot, num_workers: int, queue_length: int) -> int:
        """Return the pool size wanted for the next interval. Pure: no threads, no clock."""



class FixedController(Controller):
    def __init__(self, n):
        self.n = n

    def decide(self, snapshot, num_workers, queue_length):
        return self.n


class RuleController(Controller):
    def __init__(self, headroom=0.25, use_queue=True, drain_time=1.0, shrink_ticks=3):
        self.headroom = headroom #h
        self.use_queue = use_queue
        self.drain_time = drain_time #D
        self.shrink_ticks = shrink_ticks
        self._low_ticks = 0

    def decide(self, snapshot, num_workers, queue_length):
        # target = ceil(λW(1+h) + Q·W/D), min 1; hold if W is None;
        # grow at once; shrink only after shrink_ticks low ticks in a row.
        if snapshot.mean_service is None: return num_workers
        qwd = queue_length*snapshot.mean_service/self.drain_time
        if not self.use_queue: qwd = 0
        target = max(1, math.ceil(snapshot.arrival_rate*snapshot.mean_service*(1+self.headroom)+ qwd))
        if target >= num_workers: #if target commands growth, immediately grow
            self._low_ticks = 0
            return target
        else: #if target commands shrink, increment the shrink command count (self._low_ticks), and shrink only if the count reached self.shrink_ticks
            self._low_ticks += 1
            if self._low_ticks >= self.shrink_ticks:
                self._low_ticks = 0
                return target
            else:
                return num_workers
