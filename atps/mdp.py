"""What the DQN sees (state), what it can do (actions), and what it is paid for (reward)."""
import math

from atps.stats import Snapshot

ACTIONS = ("halve", "minus_one", "hold", "plus_one", "double")
MAX_WORKERS = 64
HEADROOM = 0.25       # same as RuleController, so both aim at the same target
THREAD_COST = 0.02    # w: one thread above target costs as much as 20 ms of p95


def littles_law_target(snapshot: Snapshot) -> int:
    """Threads needed by Little's Law plus headroom: ceil(λW(1+h)), at least 1. 1 when W is unknown."""
    if snapshot.mean_service is None:
        return 1
    return max(1, math.ceil(snapshot.arrival_rate * snapshot.mean_service * (1 + HEADROOM)))


def apply_action(action: int, num_workers: int) -> int:
    """Index into ACTIONS → new pool size, kept within 1..MAX_WORKERS. Halve rounds down."""
    match action:
        case 0: new_size = num_workers // 2
        case 1: new_size = num_workers - 1
        case 2: new_size = num_workers
        case 3: new_size = num_workers + 1
        case 4: new_size = num_workers * 2
        case _: raise ValueError(f"unknown action {action}")
    return max(1, min(MAX_WORKERS, new_size))


def state(snapshot: Snapshot, num_workers: int, queue_length: int) -> list[float]:
    """A fixed-length list of floats, each roughly in 0..a few. Must cope with None fields."""
    return [
        num_workers / MAX_WORKERS,             # 0..1
        min(queue_length, 1000) / 100,         # 0..10; capped so one huge backlog can't swamp the rest
        float(snapshot.p95_latency or 0.0),    # seconds, ~0.1..6; None (nothing finished yet) → 0
        snapshot.arrival_rate * (snapshot.mean_service or 0.0) / MAX_WORKERS,  # λW: threads needed, 0..~1
        _alive(snapshot, num_workers) / MAX_WORKERS,  # includes retiring threads still finishing queued work
    ]


def _alive(snapshot: Snapshot, num_workers: int) -> int:
    """Threads actually running. num_workers stops counting a thread the moment its pill is queued,
    but during a burst that thread keeps working until it reaches the pill, so it still costs."""
    return num_workers if snapshot.live_threads is None else snapshot.live_threads


def reward(snapshot: Snapshot, num_workers: int, queue_length: int) -> float:
    """Higher is better: penalise p95 latency and threads above what Little's Law says is needed."""
    latency_cost = snapshot.p95_latency or 0.0   # None: nothing finished in the window, so no latency observed
    excess_threads = max(0, _alive(snapshot, num_workers) - littles_law_target(snapshot))
    return -latency_cost - THREAD_COST * excess_threads
