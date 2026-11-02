import argparse
import csv
import time
from pathlib import Path

from atps.controllers import FixedController, RuleController
from atps.pool import AdaptiveThreadPoolExecutor
from atps.stats import percentile
from atps.workloads import SCENARIOS, generate

MODEL_PATH = "models/dqn.pt"


def _load_dqn():
    from atps.dqn import DQNController  # imported here so the other controllers don't need torch
    return DQNController.load(MODEL_PATH)


# name → (controller factory, initial workers)
CONTROLLERS = {
    "fixed-4":      (lambda: FixedController(4), 4),
    "fixed-16":     (lambda: FixedController(16), 16),
    "rule":         (lambda: RuleController(use_queue=True), 2),
    "rule-noqueue": (lambda: RuleController(use_queue=False), 2),
    "dqn":          (_load_dqn, 2),
}

FIELDS = ["scenario", "controller", "seed", "tasks", "p50", "p95", "p99",
          "mean_wait", "makespan", "avg_threads", "max_threads", "thread_seconds"]


def average_threads(sizes, start, end):
    """Time-weighted mean pool size over [start, end]. sizes is stats.sizes: [(time, size), ...]."""
    total = 0.0
    for i, (t, n) in enumerate(sizes):
        seg_start = max(t, start)
        seg_end = min(sizes[i + 1][0] if i + 1 < len(sizes) else end, end)
        if seg_end > seg_start:
            total += n * (seg_end - seg_start)
    return total / (end - start)


def run_one(tasks, controller, initial_workers, interval=0.1):
    """Submit tasks at their scheduled times (open loop), wait for all, return metrics."""
    pool = AdaptiveThreadPoolExecutor(initial_workers, controller=controller, interval=interval)
    try:
        t0 = pool.stats.now()
        futures = []
        for task in tasks:
            delay = t0 + task.at - pool.stats.now()
            if delay > 0:
                time.sleep(delay)
            futures.append(pool.submit(time.sleep, task.service))
        for f in futures:
            f.result()
        timings = pool.stats.timings
        sizes = pool.stats.sizes  # read before shutdown, which logs a final 0
    finally:
        pool.shutdown()

    end = max(t.finished for t in timings)
    latencies = [t.latency for t in timings]
    avg = average_threads(sizes, t0, end)
    return {
        "tasks": len(timings),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "mean_wait": sum(t.wait for t in timings) / len(timings),
        "makespan": end - t0,
        "avg_threads": avg,
        "max_threads": max(n for t, n in sizes if t <= end),
        "thread_seconds": avg * (end - t0),
    }


def main():
    parser = argparse.ArgumentParser(description="Run every controller on every scenario and write a CSV.")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    default = [c for c in CONTROLLERS if c != "dqn" or Path(MODEL_PATH).exists()]
    parser.add_argument("--controllers", nargs="*", default=default)
    parser.add_argument("--out", default="results/eval.csv")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for seed in range(args.seeds):
            for scenario in args.scenarios:
                tasks = generate(SCENARIOS[scenario], seed=seed)
                for name in args.controllers:
                    make, initial = CONTROLLERS[name]
                    row = run_one(tasks, make(), initial)
                    row.update(scenario=scenario, controller=name, seed=seed)
                    writer.writerow(row)
                    f.flush()
                    print(f"{scenario:8} {name:13} seed={seed}  p95={row['p95']:.3f}s  "
                          f"avg_threads={row['avg_threads']:.1f}  makespan={row['makespan']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
