"""DQN controller, following the PyTorch DQN tutorial: replay memory, a policy and a target network,
epsilon-greedy exploration, Huber loss, soft target updates."""
from collections import deque, namedtuple
import math
import random

import torch
from torch import nn, optim

from atps.controllers import Controller
from atps.mdp import ACTIONS, apply_action, reward, state
from atps.stats import Snapshot

torch.set_num_threads(1)  # keep torch from competing with the pool's threads for cores

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state"])


class QNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions, hidden=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.layers(x)  # one Q-value per action


class ReplayMemory:
    def __init__(self, capacity=20_000):
        self._memory = deque(maxlen=capacity)

    def push(self, *transition):
        self._memory.append(Transition(*transition))

    def sample(self, batch_size):
        return random.sample(self._memory, batch_size)

    def __len__(self):
        return len(self._memory)


class DQNAgent:
    def __init__(self, gamma=0.9, lr=1e-3, batch_size=64, tau=0.005,
                 eps_start=0.9, eps_end=0.05, eps_decay=3000, seed=0):
        random.seed(seed)
        torch.manual_seed(seed)
        n_inputs = len(state(Snapshot(0.0, 0, None, None), 1, 0))
        self.policy = QNetwork(n_inputs, len(ACTIONS))
        self.target = QNetwork(n_inputs, len(ACTIONS))
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory()
        self.gamma, self.batch_size, self.tau = gamma, batch_size, tau
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.steps = 0

    @property
    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.steps / self.eps_decay)

    def act(self, s, explore):
        if explore:
            eps = self.epsilon
            self.steps += 1
            if random.random() < eps:
                return random.randrange(len(ACTIONS)) # the probability of this running is eps
        with torch.no_grad():
            return int(self.policy(torch.tensor([s], dtype=torch.float32)).argmax(dim=1).item())

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = Transition(*zip(*self.memory.sample(self.batch_size)))
        states = torch.tensor(batch.state, dtype=torch.float32)
        actions = torch.tensor(batch.action).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32)
        next_states = torch.tensor(batch.next_state, dtype=torch.float32)

        q = self.policy(states).gather(1, actions).squeeze(1)       # Q(s, a) for the action taken
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1).values      # best Q from s', by the slow network
        expected = rewards + self.gamma * next_q                     # no terminal states: the pool runs on

        loss = nn.functional.smooth_l1_loss(q, expected)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.optimizer.step()

        # soft update: target ← τ·policy + (1−τ)·target
        target_sd, policy_sd = self.target.state_dict(), self.policy.state_dict()
        for k in policy_sd:
            target_sd[k] = policy_sd[k] * self.tau + target_sd[k] * (1 - self.tau)
        self.target.load_state_dict(target_sd)
        return loss.item()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.target.load_state_dict(self.policy.state_dict())


class DQNController(Controller):
    """One per episode. With train=True it explores, stores transitions and learns on every tick."""

    def __init__(self, agent, train=False):
        self.agent = agent
        self.train = train
        self._prev = None  # (state, action) from the last tick
        self.losses = []

    @classmethod
    def load(cls, path):
        agent = DQNAgent()
        agent.load(path)
        return cls(agent, train=False)

    def decide(self, snapshot, num_workers, queue_length):
        s = state(snapshot, num_workers, queue_length)
        if self.train and self._prev is not None:
            prev_s, prev_a = self._prev
            # the reward for last tick's action is what we observe now
            self.agent.memory.push(prev_s, prev_a, reward(snapshot, num_workers, queue_length), s)
            loss = self.agent.optimize()
            if loss is not None:
                self.losses.append(loss)
        a = self.agent.act(s, explore=self.train)
        self._prev = (s, a)
        return apply_action(a, num_workers)
