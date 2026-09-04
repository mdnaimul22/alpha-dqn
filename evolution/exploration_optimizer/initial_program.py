"""
Initial Evolvable Exploration Policy Module for OpenEvolve.

Defines the action selection and exploration policy for Deep Q-Networks.
OpenEvolve will mutate, refine, and optimize this module to discover novel
exploration strategies (e.g. directed epistemic exploration, adaptive Boltzmann
temperature, noisy action perturbation, or state-dependent Thompson sampling)
that maximize deep diffusion speed, state visitation entropy, and action coherence.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


class ExplorationPolicyModule:
    """
    Evolvable Exploration Policy for Discrete Action Reinforcement Learning.
    Baseline implementation: Epsilon-Greedy blended with Temperature-Softmax and Action Persistence.
    """

    def __init__(self, state_dim: int, action_dim: int, seed: int = 42):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)

        # Action visitation counters & state statistics
        self.action_counts = np.zeros(self.action_dim, dtype=np.int64)
        self.total_steps = 0
        self.last_action = 0
        self.temperature = 1.0

    def select_action(
        self,
        q_values: np.ndarray,
        state: np.ndarray,
        step: int = 0,
        epsilon: float = 0.1,
    ) -> int:
        """
        Select discrete action using exploration policy.

        Args:
            q_values: 1D array of predicted Q-values from neural network [action_dim]
            state: 1D state vector observation [state_dim]
            step: Global or episodic environment step counter
            epsilon: Reference exploration rate from training schedule

        Returns:
            Chosen discrete action integer in range [0, action_dim - 1]
        """
        q = np.asarray(q_values, dtype=np.float64)

        # 1. Check for confident exploitation: when Q-spread is high, exploit
        q_range = float(np.max(q) - np.min(q))
        if q_range > 3.0 and self.rng.random() > epsilon * 0.5:
            action = int(np.argmax(q))
            self.last_action = action
            return action

        # 2. Temperature-scaled Softmax Exploration
        if self.rng.random() < epsilon:
            # Exploration mode: Boltzmann softmax over normalized Q-values
            scaled_q = (q - np.max(q)) / max(0.1, self.temperature)
            exp_q = np.exp(np.clip(scaled_q, -20.0, 20.0))
            probs = exp_q / (np.sum(exp_q) + 1e-12)

            # Blend with action frequency bonus (encourages under-visited actions)
            total = max(1, np.sum(self.action_counts))
            freq = self.action_counts / float(total)
            bonus = 1.0 / (np.sqrt(freq + 1e-5))
            bonus_probs = bonus / np.sum(bonus)

            blended_probs = 0.7 * probs + 0.3 * bonus_probs
            blended_probs = blended_probs / np.sum(blended_probs)

            action = int(self.rng.choice(self.action_dim, p=blended_probs))
        else:
            # Exploitation mode: Greedy argmax
            action = int(np.argmax(q))

        self.last_action = action
        return action

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> dict:
        """
        Update internal exploration statistics from transition.
        """
        self.total_steps += 1
        if 0 <= action < self.action_dim:
            self.action_counts[action] += 1

        # Smooth temperature decay
        self.temperature = max(0.05, self.temperature * 0.9995)

        return {
            "total_steps": self.total_steps,
            "temperature": self.temperature,
        }

    def reset_episode(self) -> None:
        """Reset episode-specific exploration state."""
        self.last_action = 0

    def reset(self) -> None:
        """Full reset of exploration policy counters."""
        self.action_counts.fill(0)
        self.total_steps = 0
        self.last_action = 0
        self.temperature = 1.0


def create_exploration_module(state_dim: int, action_dim: int, seed: int = 42) -> ExplorationPolicyModule:
    """Factory function required by OpenEvolve evaluator."""
    return ExplorationPolicyModule(state_dim=state_dim, action_dim=action_dim, seed=seed)


if __name__ == "__main__":
    mod = create_exploration_module(state_dim=4, action_dim=3, seed=42)
    s = np.array([0.1, 0.2, 0.3, 0.4])
    q = np.array([1.2, 2.5, 0.8])
    a = mod.select_action(q, s, step=0, epsilon=0.2)
    print(f"Selected action: {a}")
    mod.update(s, a, 1.0, s, False)
