import argparse
from pathlib import Path

from atps.dqn import DQNAgent, DQNController
from atps.evaluate import run_one
from atps.workloads import SCENARIOS, generate

TRAIN_SEED_OFFSET = 1000  # evaluation uses seeds 0, 1, 2; training never sees them


def main():
    parser = argparse.ArgumentParser(description="Train the DQN controller against the real pool.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", default="models/dqn.pt")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    agent = DQNAgent()
    names = list(SCENARIOS)
    for ep in range(args.episodes):
        scenario = names[ep % len(names)]
        tasks = generate(SCENARIOS[scenario], seed=TRAIN_SEED_OFFSET + ep)
        controller = DQNController(agent, train=True)
        row = run_one(tasks, controller, initial_workers=2)
        loss = sum(controller.losses) / len(controller.losses) if controller.losses else float("nan")
        print(f"ep {ep:3}  {scenario:7} p95={row['p95']:.3f}s  avg_threads={row['avg_threads']:5.1f}  "
              f"eps={agent.epsilon:.2f}  loss={loss:.4f}", flush=True)
        agent.save(out)  # save every episode, so a stopped run still leaves a model


if __name__ == "__main__":
    main()
