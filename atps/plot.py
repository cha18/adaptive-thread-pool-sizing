"""Pool size over time on one scenario: what each controller actually did."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atps.evaluate import CONTROLLERS, run_one
from atps.workloads import MEAN_SERVICE, SCENARIOS, generate

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"rule": "#2a78d6", "dqn": "#eb6834"}  # categorical slots 1 and 2


def needed_threads(phases):
    """Little's Law per phase: λW threads busy. Returns step points (t, n)."""
    points, t = [], 0.0
    for duration, rate in phases:
        points.append((t, rate * MEAN_SERVICE))
        t += duration
    points.append((t, points[-1][1]))
    return points


def step(points, end):
    xs, ys = zip(*points)
    return list(xs) + [end], list(ys) + [ys[-1]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="on_off")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controllers", nargs="*", default=["rule", "dqn"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    phases = SCENARIOS[args.scenario]
    tasks = generate(phases, seed=args.seed)
    rows = {}
    for name in args.controllers:
        make, initial = CONTROLLERS[name]
        rows[name] = (run_one(tasks, make(), initial), initial)
    # a run ends when its last task finishes; drawing past that would invent data
    end = min(sum(d for d, _ in phases), max(row["makespan"] for row, _ in rows.values()))

    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    needed = [(t, n) for t, n in needed_threads(phases) if t < end]
    xs, ys = step(needed, end)
    ax.step(xs, ys, where="post", color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)),
            label="threads needed (λW)")

    ends = []
    for name, (row, initial) in rows.items():
        run_end = min(end, row["makespan"])
        log = [(t, n) for t, n in row["live_log"] if 0 <= t <= run_end]  # threads actually alive
        log = [(0.0, initial)] + log
        xs, ys = step(log, run_end)
        ax.step(xs, ys, where="post", color=SERIES.get(name), linewidth=2,
                label=f"{name}  (p95 {row['p95']:.2f}s, avg {row['avg_threads']:.1f} threads)")
        ends.append((ys[-1], name, run_end))

    # end labels: nudge apart when the lines finish at nearly the same size
    ends.sort()
    for i, (y, name, x) in enumerate(ends):
        crowded = any(abs(y - other) < 1.5 for j, (other, _, _) in enumerate(ends) if j != i)
        dy = (-6 if i == 0 else 6) if crowded else 0
        ax.annotate(name, (x, y), xytext=(6, dy), textcoords="offset points",
                    va="center", color=INK, fontsize=9)

    ax.set_xlim(0, end)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("time (s)", color=MUTED)
    ax.set_ylabel("threads alive", color=MUTED)
    ax.set_title(f"Threads alive over time: {args.scenario}, seed {args.seed}", color=INK, loc="left", fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    out = args.out or f"results/pool_size_{args.scenario}.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
