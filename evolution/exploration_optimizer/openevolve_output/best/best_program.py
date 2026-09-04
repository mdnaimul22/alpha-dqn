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
    Evolvable Directed Exploration Policy with Sticky Action Persistence and State-Action UCB.
    Combines temporal momentum (for rapid deep diffusion) with count-based epistemic UCB (for high state entropy).
    """

    def __init__(self, state_dim: int, action_dim: int, seed: int = 42):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)

        self.action_counts = np.zeros(self.action_dim, dtype=np.float64)
        self.state_counts: dict[tuple, float] = {}
        self.sa_counts: dict[tuple, np.ndarray] = {}
        self.total_steps = 0
        self.last_action = 0
        self.prev_action = 0
        self.sticky_counter = 0
        self.last_raw_state: np.ndarray | None = None

    def _get_state_key(self, state: np.ndarray) -> tuple:
        """Discretize continuous state observation into fine grid bins for UCB tracking."""
        return tuple(np.round(np.asarray(state, dtype=np.float64) * 3.5).astype(int))

    def select_action(
        self,
        q_values: np.ndarray,
        state: np.ndarray,
        step: int = 0,
        epsilon: float = 0.1,
    ) -> int:
        q = np.asarray(q_values, dtype=np.float64)
        sorted_q = np.sort(q)
        q_margin = float(sorted_q[-1] - sorted_q[-2]) if len(sorted_q) > 1 else 0.0

        # 1. Confident Exploitation: Select optimal action when Q-margin is high
        if q_margin > 0.58 and self.rng.random() > (epsilon * 0.05):
            action = int(np.argmax(q))
            self.prev_action = self.last_action
            self.last_action = action
            self.sticky_counter = 0
            self.last_raw_state = np.asarray(state, dtype=np.float64).copy()
            return action

        # 2. Continuous Collision Sensing: Detect stuck states via continuous state difference
        is_wall_bump = False
        if self.last_raw_state is not None:
            is_wall_bump = bool(np.allclose(state, self.last_raw_state, atol=1e-5))

        if is_wall_bump:
            self.sticky_counter = 0

        # 3. Deterministic Sticky Action Persistence: Maintain momentum to boost deep diffusion speed
        if self.sticky_counter > 0:
            self.sticky_counter -= 1
            self.last_raw_state = np.asarray(state, dtype=np.float64).copy()
            return self.last_action

        # 4. Directed Epistemic UCB + Anti-Dithering Momentum Exploration
        s_key = self._get_state_key(state)
        sa_cnt = self.sa_counts.get(s_key, np.zeros(self.action_dim, dtype=np.float64))
        s_cnt = self.state_counts.get(s_key, 0.0)

        sa_ucb = 0.82 / np.sqrt(sa_cnt + 1.0)
        s_novelty = 0.52 / np.sqrt(s_cnt + 1.0)
        global_bal = 0.22 / np.sqrt(self.action_counts + 1.0)

        inertia = np.zeros(self.action_dim, dtype=np.float64)
        if is_wall_bump:
            # Heavily penalize repeating an action that caused a wall collision
            inertia[self.last_action] = -1.8
        else:
            inertia[self.last_action] = 0.58
            if self.prev_action != self.last_action:
                inertia[self.prev_action] -= 0.35

        scores = q + (epsilon + 0.05) * (sa_ucb + s_novelty + global_bal) + inertia

        if self.rng.random() < (epsilon * 0.2):
            logits = (scores - np.max(scores)) / 0.35
            exp_q = np.exp(np.clip(logits, -20.0, 20.0))
            probs = exp_q / np.sum(exp_q)
            action = int(self.rng.choice(self.action_dim, p=probs))
        else:
            action = int(np.argmax(scores))

        self.prev_action = self.last_action
        self.last_action = action
        self.sticky_counter = int(self.rng.integers(3, 6))
        self.last_raw_state = np.asarray(state, dtype=np.float64).copy()
        return action

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> dict:
        self.total_steps += 1
        if 0 <= action < self.action_dim:
            self.action_counts[action] += 1.0
            s_key = self._get_state_key(state)
            self.state_counts[s_key] = self.state_counts.get(s_key, 0.0) + 1.0
            if s_key not in self.sa_counts:
                self.sa_counts[s_key] = np.zeros(self.action_dim, dtype=np.float64)
            self.sa_counts[s_key][action] += 1.0
        self.last_raw_state = np.asarray(state, dtype=np.float64).copy()

        return {"total_steps": self.total_steps}

    def reset_episode(self) -> None:
        self.last_action = 0
        self.prev_action = 0
        self.sticky_counter = 0
        self.last_raw_state = None

    def reset(self) -> None:
        self.action_counts.fill(0)
        self.state_counts.clear()
        self.sa_counts.clear()
        self.total_steps = 0
        self.last_action = 0
        self.prev_action = 0
        self.sticky_counter = 0
        self.last_raw_state = None


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
