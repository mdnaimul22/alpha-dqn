"""
Initial Evolvable Curiosity Reward Module for OpenEvolve.

This program defines the curiosity and intrinsic exploration reward mechanism.
OpenEvolve will mutate, refine, and optimize this module to discover novel
exploration bonus formulations that maximize state-space coverage, achieve rapid
habituation decay on familiar states, and maintain immunity against stochastic noise (Noisy-TV).
"""

import numpy as np


class CuriosityRewardModule:
    """
    Intrinsic Curiosity Reward Module.
    
    Computes an intrinsic reward bonus r_i(s, a, s') to guide reinforcement
    learning agents (e.g., DQN) in sparse, deceptive, or hard-exploration environments.
    """

    def __init__(self, state_dim: int, action_dim: int, seed: int = 42):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)

        # Baseline: Random Network Distillation (RND) linear projection baseline
        self.feature_dim = 16
        
        # Fixed random target projection (represents static target embedding)
        self.target_matrix = self.rng.standard_normal((self.state_dim, self.feature_dim)) / np.sqrt(self.state_dim)
        
        # Trainable online predictor matrix
        self.online_matrix = np.zeros((self.state_dim, self.feature_dim), dtype=np.float64)
        
        # State frequency tracker for habituation (discretized pseudo-count)
        self.visit_counts = {}
        self.learning_rate = 0.05

    def _discretize_state(self, state: np.ndarray) -> tuple:
        """Discretize continuous state vectors into spatial grid keys for count tracking."""
        rounded = np.round(np.asarray(state, dtype=np.float64), 1)
        return tuple(rounded.tolist())

    def compute_intrinsic_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray
    ) -> float:
        """
        Compute intrinsic exploration reward bonus for transition (state, action, next_state).
        
        Args:
            state: Current state observation (1D array of shape [state_dim])
            action: Action taken (int)
            next_state: Resulting state observation (1D array of shape [state_dim])
            
        Returns:
            Non-negative float bonus >= 0.0 (normalized typically between 0.0 and 1.0).
        """
        s_next = np.asarray(next_state, dtype=np.float64)

        # 1. Feature projections
        target_features = s_next @ self.target_matrix
        predicted_features = s_next @ self.online_matrix

        # 2. Prediction error (RND novelty proxy)
        prediction_error = float(np.mean((target_features - predicted_features) ** 2))

        # 3. Habituation / Boredom decay factor based on visit count
        s_key = self._discretize_state(s_next)
        count = self.visit_counts.get(s_key, 0) + 1
        habituation_factor = 1.0 / np.sqrt(count)

        # 4. Synthesized curiosity bonus
        intrinsic_reward = prediction_error * habituation_factor

        # Guardrails: non-negative, finite, bounded
        if not np.isfinite(intrinsic_reward) or intrinsic_reward < 0.0:
            return 0.0
        return float(min(intrinsic_reward, 1.0))

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray
    ) -> dict:
        """
        Update online predictor / internal memory from a batch of experienced transitions.
        
        Args:
            states: Batch of current states [B, state_dim]
            actions: Batch of actions [B]
            next_states: Batch of next states [B, state_dim]
            
        Returns:
            Dictionary of metrics (e.g., training loss).
        """
        batch_size = len(next_states)
        if batch_size == 0:
            return {"loss": 0.0}

        s_next = np.asarray(next_states, dtype=np.float64)

        # Update visit counts for habituation
        for s in s_next:
            s_key = self._discretize_state(s)
            self.visit_counts[s_key] = self.visit_counts.get(s_key, 0) + 1

        # Online gradient descent update to match target projection
        target_feat = s_next @ self.target_matrix
        pred_feat = s_next @ self.online_matrix
        diff = pred_feat - target_feat

        # Gradient: dLoss/dW = (1/B) * (s_next.T @ diff)
        grad = (s_next.T @ diff) / float(batch_size)
        self.online_matrix -= self.learning_rate * grad

        loss = float(np.mean(diff ** 2))
        return {"loss": loss}

    def reset(self) -> None:
        """Reset episode-specific temporary buffers or caches if applicable."""
        pass


def create_curiosity_module(state_dim: int, action_dim: int) -> CuriosityRewardModule:
    """
    Factory function required by OpenEvolve evaluator.
    Instantiates and returns a CuriosityRewardModule instance.
    """
    return CuriosityRewardModule(state_dim=state_dim, action_dim=action_dim)


if __name__ == "__main__":
    # Quick sanity check
    mod = create_curiosity_module(state_dim=4, action_dim=2)
    s = np.array([0.1, 0.2, 0.3, 0.4])
    ns = np.array([0.2, 0.3, 0.4, 0.5])
    r = mod.compute_intrinsic_reward(s, 0, ns)
    print(f"Initial intrinsic reward: {r:.6f}")
    
    # Train step
    metrics = mod.update(np.array([s]), np.array([0]), np.array([ns]))
    print(f"Update loss: {metrics['loss']:.6f}")
    
    # Next reward should decay
    r_after = mod.compute_intrinsic_reward(s, 0, ns)
    print(f"Intrinsic reward after 1 update: {r_after:.6f}")
