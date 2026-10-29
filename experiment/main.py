import matplotlib.pyplot as plt
import numpy as np
from dql_agent import Agent
from scheduler import TaskSchedulerEnv

def train_model(episodes=50):
    env = TaskSchedulerEnv()
    agent = Agent(env.state_dim, env.action_dim)
    
    print(f"Training Agent for {episodes} episodes...")
    
    rewards_history = []
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        
        # Randomize scenario for training robustness
        scenario = np.random.choice(['cpu', 'io', 'mixed'])
        
        for time_step in range(100): # 100 steps per episode
            action = agent.select_action(state, is_training=True)
            next_state, reward = env.step(action, scenario)
            
            agent.store_transition(state, action, reward, next_state)
            agent.learn()
            
            state = next_state
            total_reward += reward
            
        agent.update_target_network()
        rewards_history.append(total_reward)
        print(f"Episode {e+1}/{episodes} - Reward: {total_reward:.2f} - Epsilon: {agent.steps_done}")

    return agent

def evaluate_performance(agent=None):
    """
    If agent is None, runs Static Baseline (N=10).
    If agent is provided, runs DQL.
    """
    env = TaskSchedulerEnv()
    state = env.reset()
    
    # Tracking metrics
    time_points = []
    throughput_vals = []
    n_vals = []
    queue_vals = []
    
    scenario_sequence = ['io']*30 + ['cpu']*30 + ['mixed']*30 + ['spike']*10
    
    if agent is None:
        print("Running Baseline (Static N=10)...")
        env.current_n = 10 # Static Baseline
    else:
        print("Running DQL Model...")

    for i, scenario in enumerate(scenario_sequence):
        if agent:
            action = agent.select_action(state, is_training=False)
        else:
            action = 1 # 1 = Hold (Do nothing, keep static)
            env.current_n = 10 # Force static
            
        next_state, reward = env.step(action, scenario)
        state = next_state
        
        # Extract metrics from state
        # State: [Queue, CPU, Wait, Service, Throughput, N]
        throughput_vals.append(state[4])
        queue_vals.append(state[0])
        n_vals.append(state[5])
        time_points.append(i)

    return time_points, throughput_vals, n_vals, queue_vals

# --- Main Execution ---

# 1. Train the Agent
trained_agent = train_model(episodes=30)

# 2. Run Scenarios
t_dql, tp_dql, n_dql, q_dql = evaluate_performance(trained_agent)
t_base, tp_base, n_base, q_base = evaluate_performance(agent=None)

# 3. Plot Results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot 1: Throughput
ax1.plot(t_base, tp_base, label='Static N=10', color='gray', linestyle='--')
ax1.plot(t_dql, tp_dql, label='DQL Agent', color='blue', linewidth=2)
ax1.set_ylabel('Throughput (tasks/s)')
ax1.set_title('Performance Comparison: Static vs DQL')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Queue Size (Latency Indicator)
ax2.plot(t_base, q_base, label='Static N=10', color='gray', linestyle='--')
ax2.plot(t_dql, q_dql, label='DQL Agent', color='red', linewidth=2)
ax2.set_ylabel('Queue Size')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Thread Pool Size (N)
ax3.plot(t_base, n_base, label='Static N=10', color='gray', linestyle='--')
ax3.plot(t_dql, n_dql, label='DQL N', color='green', linewidth=2)
ax3.set_ylabel('Pool Size (N)')
ax3.set_xlabel('Time Steps (Scenario: IO -> CPU -> Mixed -> Spike)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dql_optimization_results.png')
plt.show()