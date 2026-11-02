import pytest

from atps.controllers import FixedController, RuleController
from atps.evaluate import average_threads, run_one
from atps.workloads import Task


def test_average_threads_is_time_weighted():
    # 2 threads for 1s, then 6 threads for 3s -> (2 + 18) / 4 = 5
    sizes = [(0.0, 2), (1.0, 6), (9.0, 0)]
    assert average_threads(sizes, start=0.0, end=4.0) == pytest.approx(5.0)


def test_average_threads_clips_to_the_window():
    # the size was set before the window started; only [10, 12] counts
    sizes = [(5.0, 3), (11.0, 1)]
    assert average_threads(sizes, start=10.0, end=12.0) == pytest.approx(2.0)


def test_run_one_reports_every_task():
    tasks = [Task(at=i * 0.01, service=0.02) for i in range(30)]
    row = run_one(tasks, FixedController(4), initial_workers=4)
    assert row["tasks"] == 30
    assert row["avg_threads"] == pytest.approx(4.0)
    assert 0 < row["p50"] <= row["p95"] <= row["p99"]


def test_an_undersized_pool_has_worse_p95_than_a_big_one():
    tasks = [Task(at=i * 0.01, service=0.1) for i in range(50)]   # lambda=100, W=0.1 → ~10 threads needed
    small = run_one(tasks, FixedController(2), initial_workers=2)
    big = run_one(tasks, FixedController(16), initial_workers=16)
    assert small["p95"] > 3 * big["p95"]


def test_the_rule_controller_grows_under_load():
    tasks = [Task(at=i * 0.01, service=0.1) for i in range(100)]
    row = run_one(tasks, RuleController(), initial_workers=1, interval=0.05)
    assert row["max_threads"] >= 5
