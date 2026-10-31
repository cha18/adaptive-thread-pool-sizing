from dataclasses import dataclass
import math
import time
import threading



@dataclass(frozen=True)
class Snapshot:
    arrival_rate: float
    completed: int
    mean_service: float | None
    p95_latency: float | None

@dataclass(frozen=True)
class TaskTiming:
    submitted: float
    started: float
    finished: float

    @property
    def wait(self):
        return self.started - self.submitted

    @property
    def service(self):
        return self.finished - self.started

    @property
    def latency(self):
        return self.finished - self.submitted




class Stats:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._submit_timestamps = []
        self._finished_timings = []
        self._size_changes = []

    @property
    def timings(self):
        with self._lock:
            return list(self._finished_timings)

    @property
    def sizes(self):
        with self._lock:
            return list(self._size_changes)

    def now(self):
        return self._clock()

    def record_submit(self):
        with self._lock:
            timestamp = self.now()
            self._submit_timestamps.append(timestamp)
            return timestamp

    def record_done(self, timing):
        with self._lock:
            self._finished_timings.append(timing)

    def record_size(self, n):
        with self._lock:
            self._size_changes.append((self.now(), n))

    def snapshot(self, window=1.0):
        with self._lock:
            cutoff = self.now()-window
            arrival_rate = len([x for x in self._submit_timestamps if x > cutoff])/window
            recent_tasks = [x for x in self._finished_timings if x.finished > cutoff]
            completed = len(recent_tasks)
            mean_service = sum([t.service for t in recent_tasks])/completed if completed>0 else None
            p95_latency = percentile([t.latency for t in recent_tasks], 95) if completed>0 else None
            return Snapshot(arrival_rate, completed, mean_service, p95_latency)

def percentile(values: list, p):
    if not values:
        raise ValueError("no values found")
    sorted_values = sorted(values)
    return sorted_values[math.ceil(p/100 * len(values))-1]
