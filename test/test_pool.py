import threading
import time
import pytest
from atps.pool import AdaptiveThreadPoolExecutor





def wait_until(condition, timeout=2.0):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_submit_returns_the_result():
    with AdaptiveThreadPoolExecutor(initial_workers=2) as pool:
        future = pool.submit(pow, 2, 10)
        assert future.result(timeout=1) == 1024

def test_starts_with_initial_workers():
    with AdaptiveThreadPoolExecutor(initial_workers=3) as pool:
        assert pool.num_workers == 3

def test_map_comes_free_from_executor():
    with AdaptiveThreadPoolExecutor(initial_workers=4) as pool:
        assert list(pool.map(lambda x: x*x, range(5)), ) == [0, 1, 4, 9, 16]

def test_exception_goes_to_the_future_and_worker_survives():
    with AdaptiveThreadPoolExecutor(initial_workers=1) as pool:
        bad = pool.submit(lambda: 1/0)
        with pytest.raises(ZeroDivisionError):
            bad.result(timeout=1)
        assert pool.submit(lambda: "still alive").result(timeout=1) == "still alive"

def test_grow_runs_more_tasks_at_once():
    with AdaptiveThreadPoolExecutor(initial_workers=1) as pool:
        pool.resize(10)
        start = time.perf_counter()
        for f in [pool.submit(time.sleep, 0.2) for _ in range(10)]:
            f.result(timeout=2)
        assert time.perf_counter() - start < 0.5

def test_shrink_retires_idle_workers():
    with AdaptiveThreadPoolExecutor(initial_workers=10) as pool:
        pool.resize(4)
        assert pool.num_workers == 4
        assert wait_until(lambda: threading.active_count() <= 1 + 4) #the 1 is referring to the main thread

def test_shrink_never_interrupts_a_running_task():
    with AdaptiveThreadPoolExecutor(initial_workers=2) as pool:
        running = [pool.submit(time.sleep, 0.3) for _ in range(2)]
        time.sleep(0.05)
        pool.resize(1)
        for f in running:
            assert f.result(timeout=2) is None #if interrupted it would raise CancelledError

def test_resize_is_clamped_to_one_and_max():
    with AdaptiveThreadPoolExecutor(initial_workers=2, max_workers=8) as pool:
        pool.resize(100)
        assert pool.num_workers == 8
        pool.resize(0)
        assert pool.num_workers == 1

def test_grow_while_pills_are_still_queued():
    #checks whether the pool grows to the correct size when growing while pills are queued behind tasks
    with AdaptiveThreadPoolExecutor(initial_workers=2) as pool:
        blockers = [pool.submit(time.sleep, 0.3) for _ in range(4)]
        pool.resize(1)
        pool.resize(3)
        assert pool.num_workers == 3
        for f in blockers:
            f.result(timeout=3)
        assert wait_until(lambda: threading.active_count() == 1+3) #the 1 is referring to the main thread

def test_submit_after_shutdown_raises():
    pool = AdaptiveThreadPoolExecutor(initial_workers=2)
    pool.shutdown()
    with pytest.raises(RuntimeError):
        pool.submit(time.sleep, 0.3)
    assert wait_until(lambda: threading.active_count() == 1) #only main thread alive

def test_shutdown_waits_for_queued_work():
    pool = AdaptiveThreadPoolExecutor(initial_workers=2)
    running = [pool.submit(time.sleep, 0.3) for _ in range(6)]
    pool.shutdown()
    for future in running: assert future.done()


def test_a_retiring_thread_counts_as_alive_until_it_reaches_its_pill():
    with AdaptiveThreadPoolExecutor(initial_workers=4) as pool:
        futures = [pool.submit(time.sleep, 0.2) for _ in range(8)]   # 4 running, 4 queued
        pool.resize(2)                                              # 2 pills go behind the queued tasks
        assert pool.num_workers == 2
        assert pool.stats.live[-1][1] == 4                          # nobody has exited yet
        for f in futures:
            f.result()
        deadline = time.monotonic() + 2
        while pool.stats.live[-1][1] != 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pool.stats.live[-1][1] == 2
