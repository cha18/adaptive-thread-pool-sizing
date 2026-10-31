from decimal import IEEE_CONTEXT_MAX_BITS
import time
import pytest
from atps.pool import AdaptiveThreadPoolExecutor
from atps.stats import Stats, TaskTiming, percentile
from atps.stats import TaskTiming, percentile



class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds




def test_percentile_matches_the_nearest_rank_example():
    values = [15,20,35,40,50]
    assert [percentile(values, p) for p in (5,30,40,50,100)] == [15,20,20,35, 50]

def test_percentile_sorts_its_input():
    assert percentile([50,15,40,20,35], 50) == 35

def test_p95_of_one_to_hundred_is_95():
    assert percentile([i for i in range(101)], 95) == 95

def test_percentile_of_nothing_raises():
    with pytest.raises(ValueError):
        percentile([], 95)


def test_task_timing_splits_latency_into_wait_and_service():
    t = TaskTiming(submitted=1.0, started=1.5, finished=2.25)
    assert t.latency == 2.25-1.0
    assert t.wait == 1.5-1.0
    assert t.service == 2.25-1.5

def test_arrival_rate_only_counts_the_window():
    clock = FakeClock()
    stats = Stats(clock=clock)
    for _ in range(5):
        stats.record_submit()
    clock.advance(2.0)
    for _ in range(3):
        stats.record_submit()
    assert stats.snapshot(window=1.0).arrival_rate == 3.0

def test_empty_window_reports_none_not_zero():
    clock = FakeClock()
    stats = Stats(clock=clock)
    snapshot = stats.snapshot(window=1.0)
    assert snapshot.arrival_rate == 0.0
    assert snapshot.completed == 0
    assert snapshot.mean_service == None
    assert snapshot.p95_latency == None

def test_snapshot_uses_tasks_finished_in_the_window():
    clock = FakeClock()
    stats = Stats(clock=clock)
    stats.record_done(TaskTiming(0.0, 0.0, 0.5))    # latency 0.5, service 0.5 (completed long time ago)
    clock.advance(10.0)
    stats.record_done(TaskTiming(9.0, 9.5, 9.6))    # latency 0.6, service 0.1
    stats.record_done(TaskTiming(9.2, 9.6, 9.9))    # latency 0.7, service 0.3
    snap = stats.snapshot(window=1.0)
    assert snap.completed == 2
    assert snap.mean_service == pytest.approx(0.2)
    assert snap.p95_latency == pytest.approx(0.7)


def test_pool_records_a_timing_for_every_task():
    with AdaptiveThreadPoolExecutor(initial_workers=2) as pool:
        for f in [pool.submit(time.sleep, 0.1) for _ in range(4)]:
            f.result(timeout=2)
        timings = pool.stats.timings
    assert len(timings) == 4
    assert all(t.service >= 0.09 for t in timings)


def test_wait_grows_when_the_pool_is_too_small():
    with AdaptiveThreadPoolExecutor(initial_workers=1) as pool:
        for f in [pool.submit(time.sleep, 0.1) for _ in range(3)]:
            f.result(timeout=2)
        waits = sorted(t.wait for t in pool.stats.timings)
    assert waits[-1] >= 0.18             # the third task queued behind two others


class SlowStats(Stats):
    # making recording slow on purpose: a pool that records after setting the result lets the caller see the result first, and gets caught.
    def record_done(self, timing):
        time.sleep(0.05)
        super().record_done(timing)


def test_stats_are_recorded_before_the_result_is_visible():
    # If a controller sees a result, the stats must already include that task.
    with AdaptiveThreadPoolExecutor(initial_workers=1, stats=SlowStats()) as pool:
        pool.submit(lambda: None).result(timeout=1)
        assert len(pool.stats.timings) == 1


def test_queue_length_counts_tasks_not_pills():
    with AdaptiveThreadPoolExecutor(initial_workers=2) as pool:
        blockers = [pool.submit(time.sleep, 0.3) for _ in range(2)]
        time.sleep(0.05)                 # both workers are now busy
        queued = [pool.submit(time.sleep, 0) for _ in range(3)]
        pool.resize(1)                   # adds a pill to the queue
        assert pool.queue_length == 3
        for f in blockers + queued:
            f.result(timeout=2)

def test_record_size_logs_time_and_size():
    clock = FakeClock()
    stats = Stats(clock=clock)
    stats.record_size(4)
    clock.advance(1.5)
    stats.record_size(8)
    assert stats.sizes == [(0.0, 4), (1.5, 8)]

def test_pool_logs_every_resize():
    clock = FakeClock()
    stats = Stats(clock=clock)
    with AdaptiveThreadPoolExecutor(initial_workers=2, stats=stats) as pool:
        pool.resize(4)
        clock.advance(1.5)
        pool.resize(8)
        pool.resize(3)
        pool.resize(3) #this should not be logged since resize() is meant to return if old and new times are equal
        assert [n for _,n in pool.stats.sizes] == [2,4,8,3]
