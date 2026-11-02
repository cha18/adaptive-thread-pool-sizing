from atps.workloads import SCENARIOS, Task, generate


def test_same_seed_gives_the_same_workload():
    assert generate([(5, 20)], seed=7) == generate([(5, 20)], seed=7)


def test_different_seeds_give_different_workloads():
    assert generate([(5, 20)], seed=1) != generate([(5, 20)], seed=2)


def test_tasks_are_in_arrival_order_and_inside_the_phases():
    tasks = generate([(2, 30), (3, 10)], seed=0)
    times = [t.at for t in tasks]
    assert times == sorted(times)
    assert all(0 <= t < 5 for t in times)


def test_arrival_count_is_close_to_rate_times_duration():
    # 20 s at 50/s -> expect 1000; Poisson std is ~32, so +-150 never fails by chance
    tasks = generate([(20, 50)], seed=0)
    assert 850 < len(tasks) < 1150


def test_each_phase_uses_its_own_rate():
    tasks = generate([(10, 5), (10, 50)], seed=0)
    first = sum(1 for t in tasks if t.at < 10)
    second = len(tasks) - first
    assert second > 5 * first


def test_a_zero_rate_phase_has_no_arrivals():
    tasks = generate([(2, 40), (3, 0), (2, 40)], seed=0)
    assert not any(2 <= t.at < 5 for t in tasks)
    assert any(t.at >= 5 for t in tasks)


def test_service_times_are_positive_with_the_right_mean():
    tasks = generate([(20, 50)], seed=0, mean_service=0.1)
    assert all(t.service > 0 for t in tasks)
    mean = sum(t.service for t in tasks) / len(tasks)
    assert 0.08 < mean < 0.12


def test_every_scenario_generates():
    for phases in SCENARIOS.values():
        tasks = generate(phases, seed=0)
        assert tasks and all(isinstance(t, Task) for t in tasks)
