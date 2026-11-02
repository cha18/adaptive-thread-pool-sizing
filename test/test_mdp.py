import math

from atps.mdp import ACTIONS, MAX_WORKERS, apply_action, reward, state
from atps.stats import Snapshot

HALVE, MINUS, HOLD, PLUS, DOUBLE = range(5)


def snap(arrival_rate=30.0, mean_service=0.1, p95=0.2):
    return Snapshot(arrival_rate=arrival_rate, completed=10, mean_service=mean_service, p95_latency=p95)


# apply_action

def test_actions_are_in_the_agreed_order():
    assert ACTIONS == ("halve", "minus_one", "hold", "plus_one", "double")


def test_each_action_changes_the_size():
    assert apply_action(HALVE, 8) == 4
    assert apply_action(MINUS, 8) == 7
    assert apply_action(HOLD, 8) == 8
    assert apply_action(PLUS, 8) == 9
    assert apply_action(DOUBLE, 8) == 16


def test_halve_rounds_down():
    assert apply_action(HALVE, 5) == 2


def test_never_below_one():
    assert apply_action(HALVE, 1) == 1
    assert apply_action(MINUS, 1) == 1


def test_never_above_max():
    assert apply_action(PLUS, MAX_WORKERS) == MAX_WORKERS
    assert apply_action(DOUBLE, 40) == MAX_WORKERS


# state

def test_state_is_a_list_of_floats():
    s = state(snap(), num_workers=4, queue_length=3)
    assert isinstance(s, list) and s
    assert all(isinstance(x, float) for x in s)


def test_state_has_the_same_length_when_fields_are_none():
    full = state(snap(), num_workers=4, queue_length=3)
    empty = state(snap(arrival_rate=0.0, mean_service=None, p95=None), num_workers=4, queue_length=0)
    assert len(full) == len(empty)
    assert all(math.isfinite(x) for x in empty)


def test_state_is_normalised_even_under_heavy_load():
    # the worst the scenarios produce: a big spike, a long queue, a full pool
    s = state(snap(arrival_rate=200.0, mean_service=0.3, p95=6.0), num_workers=MAX_WORKERS, queue_length=500)
    assert all(math.isfinite(x) and abs(x) <= 10 for x in s)


def test_state_sees_the_queue():
    assert state(snap(), 4, queue_length=0) != state(snap(), 4, queue_length=50)


# reward

def test_higher_p95_is_worse():
    assert reward(snap(p95=0.5), 4, 0) < reward(snap(p95=0.2), 4, 0)


def test_threads_above_the_littles_law_target_cost_something():
    # λW = 30*0.1 = 3 busy threads. 40 threads is wasteful
    assert reward(snap(), 40, 0) < reward(snap(), 8, 0)


def test_reward_is_finite_when_there_is_no_data():
    r = reward(snap(arrival_rate=0.0, mean_service=None, p95=None), num_workers=1, queue_length=0)
    assert math.isfinite(r)
