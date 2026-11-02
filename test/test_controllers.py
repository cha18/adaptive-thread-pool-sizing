import threading
import time

from atps.controllers import FixedController, RuleController
from atps.loop import ControlLoop
from atps.pool import AdaptiveThreadPoolExecutor
from atps.stats import Snapshot


def snap(arrival_rate, mean_service):
    # controllers only read these two fields
    return Snapshot(arrival_rate=arrival_rate, completed=10,
                    mean_service=mean_service, p95_latency=None)


def wait_until(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False



def test_fixed_always_returns_its_n():
    c = FixedController(8)
    assert c.decide(snap(30, 0.2), num_workers=3, queue_length=100) == 8


# rule controller formnula

def test_rule_is_littles_law_plus_headroom():
    c = RuleController(headroom=0.25, use_queue=False)
    # λW = 30 × 0.2 = 6, plus 25% = 7.5, rounded up = 8
    assert c.decide(snap(30, 0.2), num_workers=8, queue_length=0) == 8


def test_rule_holds_when_there_is_no_service_data():
    c = RuleController()
    assert c.decide(snap(30, None), num_workers=5, queue_length=40) == 5


def test_rule_adds_threads_to_drain_the_queue():
    c = RuleController(headroom=0.25, use_queue=True, drain_time=1.0)
    # λW(1+h) = 10 × 0.1 × 1.25 = 1.25; backlog 20 × 0.1 / 1.0 = 2.0; total 3.25 → 4
    assert c.decide(snap(10, 0.1), num_workers=2, queue_length=20) == 4


def test_rule_without_queue_ignores_the_backlog():
    c = RuleController(headroom=0.25, use_queue=False)
    assert c.decide(snap(10, 0.1), num_workers=2, queue_length=20) == 2


# rule controller growth/shrink rate

def test_rule_grows_immediately():
    c = RuleController(headroom=0.25, use_queue=False, shrink_ticks=3)
    assert c.decide(snap(30, 0.2), num_workers=2, queue_length=0) == 8


def test_rule_shrinks_only_after_enough_low_ticks():
    c = RuleController(headroom=0.25, use_queue=False, shrink_ticks=3)
    low = snap(10, 0.1)                  # wants 2
    assert c.decide(low, num_workers=8, queue_length=0) == 8   # _low_tick 1: hold
    assert c.decide(low, num_workers=8, queue_length=0) == 8   # _low_tick 2: hold
    assert c.decide(low, num_workers=8, queue_length=0) == 2   # _low_tick 3: shrink


def test_rule_never_asks_for_fewer_than_one():
    c = RuleController(headroom=0.25, use_queue=False, shrink_ticks=1)
    assert c.decide(snap(0, 0.1), num_workers=4, queue_length=0) == 1


# Control loop

def test_loop_resizes_the_pool_to_the_controllers_target():
    with AdaptiveThreadPoolExecutor(initial_workers=2) as pool:
        loop = ControlLoop(pool, FixedController(6), interval=0.02)
        loop.start()
        try:
            assert wait_until(lambda: pool.num_workers == 6)
        finally:
            loop.stop()


def test_loop_stops_promptly_even_with_a_long_interval():
    with AdaptiveThreadPoolExecutor(initial_workers=1) as pool:
        loop = ControlLoop(pool, FixedController(1), interval=10.0)
        loop.start()
        start = time.perf_counter()
        loop.stop()
        assert time.perf_counter() - start < 0.5


def test_pool_with_a_rule_controller_grows_under_load():
    controller = RuleController(headroom=0.25, shrink_ticks=5)
    with AdaptiveThreadPoolExecutor(initial_workers=1, controller=controller,
                                    interval=0.05) as pool:
        futures = [pool.submit(time.sleep, 0.2) for _ in range(20)]
        assert wait_until(lambda: pool.num_workers >= 4, timeout=3)
        for f in futures:
            f.result(timeout=5)


def test_rule_cooldown_resets_when_load_returns():
    c = RuleController(headroom=0.25, use_queue=False, shrink_ticks=3)
    low, high = snap(10, 0.1), snap(30, 0.2)   # want 2 and 8
    assert c.decide(low, num_workers=8, queue_length=0) == 8    # low tick 1: hold
    assert c.decide(low, num_workers=8, queue_length=0) == 8    # low tick 2: hold
    assert c.decide(high, num_workers=8, queue_length=0) == 8   # not low: count resets
    assert c.decide(low, num_workers=8, queue_length=0) == 8    # low tick 1
    assert c.decide(low, num_workers=8, queue_length=0) == 8    # low tick 2


def test_shutdown_stops_the_control_loop():
    pool = AdaptiveThreadPoolExecutor(initial_workers=2, controller=FixedController(2), interval=0.02)
    pool.shutdown()
    assert threading.active_count() == 1
