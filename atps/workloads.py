from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Task:
    at: float       # seconds after the run starts
    service: float  # seconds the task sleeps for (I/O-bound work)


# Each scenario is a list of phases: (duration [seconds], arrival rate [tasks/s]).
# Mean service time = 0.1s everywhere, therefore according to Little's Law a rate of lambda needs about lambda × 0.1 busy threads.
SCENARIOS = {
    "steady": [(8, 40)],                                  # ~4 threads all the way through
    "spike":  [(3, 10), (2, 150), (3, 10)],               # 1 → 15 → 1 threads
    "ramp":   [(1, r) for r in (5, 15, 25, 35, 45, 55, 65, 80)],  # slowly climbing load
    "on_off": [(2, 60), (2, 0), (2, 60), (2, 0)],         # bursts with idle gaps: tests shrinking
}
MEAN_SERVICE = 0.1


def generate(phases, mean_service=MEAN_SERVICE, seed=0) -> list[Task]:
    """Poisson arrivals, one phase after another; exponential service times. Same seed → same list."""
    rng = random.Random(seed)
    tasks = []
    start = 0
    for duration, rate in phases:
        end = start + duration
        if rate>0:
            t = start + rng.expovariate(rate)
            while t < end:
                tasks.append(Task(t, rng.expovariate(1/mean_service)))
                t += rng.expovariate(rate)
        start = end
    return tasks
