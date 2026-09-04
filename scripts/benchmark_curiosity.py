"""
Comparative Benchmark: Standard DQN vs DQN with Evolved Curiosity Engine.
Demonstrates empirical exploration advantage on a Sparse-Reward Noisy-TV Navigation Task:
  - 15-step sparse chain (goal at index 14, reward 0 everywhere except goal = +10.0)
  - Stochastic white noise dimensions (Noisy-TV distractor)
  - Evaluates goal reach rate, max frontier expansion, and unique state coverage.
"""

from __future__ import annotations

import os
import random
import sys
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PROJECT_ROOT, Settings, setup_logger

from src.core.agent import DQNAgent
from src.schema.dqn import CuriosityConfig, DQNConfig, LayerConfig

logger = setup_logger(Settings.LOG_DIR / "core.log", name="alphadqn.benchmark.curiosity")


class SparseNoisyChainEnv:
    """
    15-step chain navigation environment with sparse goal reward and Noisy-TV distractor.
    State representation: [normalized_pos, dist_to_goal, noise_1, noise_2, noise_3, noise_4] (dim=6).
    Actions: 0=Left, 1=Right, 2=No-Op.
    """

    def __init__(self, chain_length: int = 15, max_steps: int = 60, seed: int = 42) -> None:
        self.chain_length = chain_length
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.pos = 0
        self.current_step = 0

    def _get_obs(self) -> list[float]:
        norm_pos = float(self.pos) / float(self.chain_length - 1)
        dist_goal = float(self.chain_length - 1 - self.pos) / float(self.chain_length - 1)
        # Uncorrelated stochastic white noise dimensions (Noisy-TV)
        noise = self.rng.standard_normal(4).tolist()
        return [norm_pos, dist_goal] + noise

    def reset(self) -> list[float]:
        self.pos = 0
        self.current_step = 0
        return self._get_obs()

    def step(self, action: int) -> tuple[list[float], float, bool]:
        self.current_step += 1
        if action == 1:  # Right
            self.pos = min(self.chain_length - 1, self.pos + 1)
        elif action == 0:  # Left
            self.pos = max(0, self.pos - 1)

        # Sparse reward: 10.0 ONLY upon reaching the final goal cell
        done = False
        reward = 0.0
        if self.pos == self.chain_length - 1:
            reward = 10.0
            done = True
        elif self.current_step >= self.max_steps:
            done = True

        return self._get_obs(), reward, done


def run_evaluation(curiosity_enabled: bool, num_episodes: int = 25, seed: int = 42) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    config = DQNConfig(
        state_size=6,
        action_size=3,
        layers=[
            LayerConfig(units=64, activation="relu"),
            LayerConfig(units=64, activation="relu"),
        ],
        batch_size=16,
        train_interval=2,
        memory_capacity=5000,
        epsilon=0.9,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        learning_rate=0.001,
        curiosity=CuriosityConfig(
            enabled=curiosity_enabled,
            feature_dim=16,
            intrinsic_reward_scale=0.6,
            frontier_boost=1.75,
            outward_boost_scale=0.75,
            global_boost_max=2.0,
            episodic_decay_power=1.4,
        ) if curiosity_enabled else None,
        auto_save_interval=0,
    )

    agent = DQNAgent(config=config)
    agent.init()

    env = SparseNoisyChainEnv(chain_length=15, max_steps=60, seed=seed)

    goals_reached = 0
    max_position_reached = 0
    total_extrinsic_reward = 0.0

    for ep in range(num_episodes):
        obs = env.reset()
        agent.reset_episode()
        ep_reward = 0.0

        for step in range(env.max_steps):
            action = agent.act(obs)
            next_obs, reward, done = env.step(action)
            ep_reward += reward

            agent.remember(obs, action, reward, next_obs, done)
            agent.train_batch()

            max_position_reached = max(max_position_reached, env.pos)
            if env.pos == env.chain_length - 1:
                goals_reached += 1
                break

            obs = next_obs
            if done:
                break

        total_extrinsic_reward += ep_reward

    curiosity_metrics = agent.get_curiosity_metrics() or {}

    return {
        "curiosity_enabled": curiosity_enabled,
        "num_episodes": num_episodes,
        "goals_reached": goals_reached,
        "goal_reach_rate": round(goals_reached / num_episodes, 4),
        "max_position_reached": max_position_reached,
        "total_extrinsic_reward": round(total_extrinsic_reward, 2),
        "final_epsilon": round(float(agent.epsilon), 4),
        "unique_states_discovered": curiosity_metrics.get("unique_states_global", 0),
    }


def main():
    print("=" * 70)
    print("🔬 COMPARATIVE BENCHMARK: STANDARD DQN vs. DQN + EVOLVED CURIOSITY")
    print("   Environment: 15-Step Sparse Chain with Stochastic Noisy-TV Distractor")
    print("=" * 70)

    print("\n[1/2] Running Standard DQN (No Curiosity)...")
    res_vanilla = run_evaluation(curiosity_enabled=False, num_episodes=25, seed=42)

    print("\n[2/2] Running DQN + Evolved Curiosity Engine...")
    res_curious = run_evaluation(curiosity_enabled=True, num_episodes=25, seed=42)

    print("\n" + "=" * 70)
    print(f"{'Metric':<30} | {'Standard DQN':<16} | {'DQN + Curiosity':<16}")
    print("-" * 70)
    vanilla_frontier = f"{res_vanilla['max_position_reached']}/14"
    curious_frontier = f"{res_curious['max_position_reached']}/14"
    vanilla_rate = f"{res_vanilla['goal_reach_rate'] * 100:.1f}%"
    curious_rate = f"{res_curious['goal_reach_rate'] * 100:.1f}%"

    print(f"{'Goals Discovered (out of 25)':<30} | {res_vanilla['goals_reached']:<16} | {res_curious['goals_reached']:<16}")
    print(f"{'Goal Reach Rate':<30} | {vanilla_rate:>16} | {curious_rate:>16}")
    print(f"{'Deepest Frontier Reached':<30} | {vanilla_frontier:<16} | {curious_frontier:<16}")
    print(f"{'Total Extrinsic Reward':<30} | {res_vanilla['total_extrinsic_reward']:<16} | {res_curious['total_extrinsic_reward']:<16}")
    print(f"{'Unique States Discovered':<30} | {'N/A':<16} | {res_curious['unique_states_discovered']:<16}")
    print("=" * 70)


    if res_curious["goals_reached"] > res_vanilla["goals_reached"]:
        print("🎉 SUCCESS: Evolved Curiosity Engine achieved decisive exploration advantage!")
    else:
        print("⚠️ Warning: Both models performed similarly.")


if __name__ == "__main__":
    main()
