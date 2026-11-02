# ATPS: Adaptive Thread Pool Sizing

A Python thread pool that resizes itself while it runs. 

A controller watches live stats (arrival rate, service time, queue length) and picks the pool size for the next interval, aiming for a low tail latency without paying for idle threads.

Choosing a pool size is a trade-off. Too few threads causes tasks to queue, so p95 latency climbs. Too many causes idle threads that you pay for. The right size changes as the load changes, so a fixed `max_workers` is either too small at peak or wasteful the rest of the time.




## How it works

```
workload ──submit──▶ pool (shared queue + worker threads) ──▶ stats
                        ▲                                      │
                        └──── resize ◀── controller ◀──────────┘
                              (control loop, every 100 ms)
```

- **Pool** (`atps/pool.py`): Is a `concurrent.futures.Executor` with one shared queue. It starts up new threads to grow. It shrinks by queuing one "poison pill" per retiring thread, and the worker that picks it up exits after finishing its current task (no task is ever cut off)
- **Stats** (`atps/stats.py`): Records the submitted, started and finished times for every task, giving a windowed snapshot of arrival rate, mean service time and p95 latency.
- **Controllers** (`atps/controllers.py`): An interface `decide(snapshot, num_workers, queue_length) -> int`. Every controller is a pure function of the stats, so they're tested without threads.
  - `FixedController(n)`: The baseline
  - `RuleController`: Little's Law plus headroom plus a backlog term:

    **target = ⌈ λ·W·(1 + h) + Q·W / D ⌉**

    λ·W is the average number of busy threads (Little's Law). The headroom h=0.25 keeps workers no more than 80% busy. Q·W/D adds enough threads to clear the current queue within D seconds. 
  The pool grows immediately but is only allowed to shrink after 3 low readings in a row, because too few threads hurts users at once unlike too many which only costs a little.
- **Control loop** (`atps/loop.py`): A background thread that runs one tick per interval. It uses `Event.wait` rather than `sleep`, so shutdown stops it at once.

Tasks are I/O-bound (`time.sleep`), which releases the GIL. CPU-bound work wouldn't benefit from more threads in standard CPython, so it's out of scope.

## Evaluation

Each scenario is a seeded Poisson arrival process with exponential service times (mean 100 ms). 
Every controller gets **exactly the same task list** for a given seed. 
Tasks are submitted **open-loop**, at their scheduled times whether or not the pool keeps up, so an undersized pool can't slow the arrivals down and hide its overload.

| Scenario | Load |
|---|---|
| steady | 40 tasks/s for 8 s |
| spike | 10/s, then 150/s for 2 s, then 10/s |
| ramp | 5 → 80 tasks/s over 8 s |
| on_off | 60/s for 2 s, idle for 2 s, repeated |

The rule controller is able to see the queue length- the same input the planned DQN will get- so any DQN success comes from the policy and not from extra information. `rule-noqueue` is an ablation which omits the backlog term.

### Results

Mean p95 latency in seconds over 3 seeds (Apple M4, Python 3.14). [brackets show the average number of threads].

| Scenario | fixed-4 | fixed-16 | rule | rule-noqueue |
|---|---|---|---|---|
| steady | 0.70 (4) | 0.32 (16) | 0.38 (5.6) | 0.42 (5.4) |
| spike | 5.79 (4) | 0.34 (16) | 0.43 (7.1) | 0.54 (6.7) |
| ramp | 2.50 (4) | 0.31 (16) | 0.34 (5.7) | 0.36 (5.6) |
| on_off | 1.38 (4) | 0.33 (16) | 0.52 (6.0) | 0.63 (5.4) |

What the numbers show:
- The rule controller's p95 is within 0.03–0.19 s of fixed-16 while using **35–45% of the threads**.
- Fixed-4 collapses under the spike (5.8 s p95). Once the queue builds, it can't catch up.
- The backlog term (Q·W/D) cuts p95 by 20–26% on the bursty scenarios, for about 0.5 extra threads.
- **Where the rule is weakest:** `on_off`. It shrinks during idle gaps and is slow to grow back when a burst starts, partly because there's no service-time data until the first tasks finish, so it holds. This is the gap the DQN controller will target.

Comparisons between controllers are the focus here, absolute numbers are machine-dependent.
Full per-run data: [`results/eval.csv`](results/eval.csv). 

## Run it

Requires Python 3.12+. The pool and evaluation use only the standard library, and the tests require `pytest`.

```bash
python -m pytest -q                       # 51 tests
python -m atps.evaluate --seeds 3         # writes results/eval.csv 
```

<!--```python
from atps.pool import AdaptiveThreadPoolExecutor
from atps.controllers import RuleController

with AdaptiveThreadPoolExecutor(initial_workers=2, controller=RuleController()) as pool:
    future = pool.submit(fetch, url)
```-->

## Limitations

- If a controller raises an exception, the control loop thread stops and the pool stays at its last size.
- At the very start of a burst there's no service-time data yet, so the rule holds instead of growing.
- Python's GIL means this only helps I/O-bound work.
- Results come from simulated I/O (`sleep`), not real network calls.


May be developed to include free-threaded Python 3.14t for CPU-bound work readings

`experiment/` holds my original December 2025 prototype, kept for history.
