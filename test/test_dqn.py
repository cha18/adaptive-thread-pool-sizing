import pytest

torch = pytest.importorskip("torch")

from atps.dqn import DQNAgent, DQNController
from atps.mdp import ACTIONS, MAX_WORKERS
from atps.stats import Snapshot


def snap(arrival_rate=30.0, mean_service=0.1, p95=0.2):
    return Snapshot(arrival_rate=arrival_rate, completed=10, mean_service=mean_service, p95_latency=p95)


def test_agent_picks_a_valid_action():
    agent = DQNAgent()
    c = DQNController(agent)
    for n in (1, 8, MAX_WORKERS):
        assert 1 <= c.decide(snap(), num_workers=n, queue_length=5) <= MAX_WORKERS


def test_training_stores_transitions_and_learns():
    agent = DQNAgent(batch_size=8)
    c = DQNController(agent, train=True)
    n = 4
    for i in range(20):
        n = c.decide(snap(p95=0.1 + i * 0.01), num_workers=n, queue_length=i)
    assert len(agent.memory) == 19      # first tick has no previous action to reward
    assert c.losses                     # optimize ran once the memory held a batch


def test_inference_does_not_explore_or_store():
    agent = DQNAgent()
    c = DQNController(agent, train=False)
    picks = {c.decide(snap(), num_workers=8, queue_length=2) for _ in range(20)}
    assert len(picks) == 1              # greedy: same input, same answer
    assert len(agent.memory) == 0


def test_save_and_load_give_the_same_policy(tmp_path):
    agent = DQNAgent(seed=3)
    path = tmp_path / "dqn.pt"
    agent.save(path)
    loaded = DQNController.load(path)
    original = DQNController(agent)
    for q in (0, 10, 100):
        assert original.decide(snap(), 8, q) == loaded.decide(snap(), 8, q)
