"""
Intrinsic Curiosity Reward Module.

Evolved and optimized via OpenEvolve to discover novel exploration bonuses.
Combines:
  1. QR-Orthogonal RND Feature Prediction (linear feature embedding with maximal information capacity)
  2. Inverse Dynamics Model (IDM) Action Controllability & Locality Rejection (Noisy-TV immunity)
  3. Trajectory Frontier Expansion (+75% surge upon breaking outward distance records)
  4. Directional Outward Progress Multiplier (anti-backtracking momentum)
  5. Multi-Scale 3-Tier Global Novelty Cushion with Superlinear Power Habituation Decay
  6. Exponential Moving Average Adaptive Error Normalization
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.config import setup_logger, Settings
from src.schema.dqn import CuriosityConfig

logger = setup_logger(Settings.LOG_DIR / "core.log", name="alphadqn.core.curiosity")


class CuriosityEngine:
    """
    Production-grade Intrinsic Curiosity & Exploration Reward Engine.
    Discovered through autonomous multi-island evolutionary optimization.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[CuriosityConfig] = None,
        seed: int = 42,
    ) -> None:
        self.state_dim: int = int(state_dim)
        self.action_dim: int = int(action_dim)
        self.config: CuriosityConfig = config or CuriosityConfig()
        self.rng = np.random.default_rng(seed)

        self.feature_dim: int = self.config.feature_dim

        # 1. Fixed orthogonal target matrix for linear feature projection (QR factorized)
        q, _ = np.linalg.qr(
            self.rng.standard_normal((max(self.state_dim, self.feature_dim), self.feature_dim))
        )
        self.target_matrix: np.ndarray = q[: self.state_dim, : self.feature_dim].astype(np.float64)

        # 2. Trainable online RND predictor matrix
        self.online_matrix: np.ndarray = np.zeros((self.state_dim, self.feature_dim), dtype=np.float64)

        # 3. Trainable linear inverse dynamics matrix: delta_state -> action logits
        self.idm_matrix: np.ndarray = np.zeros((self.state_dim, self.action_dim), dtype=np.float64)

        # 4. Hyperparameters
        self.learning_rate: float = self.config.learning_rate
        self.running_error_mean: float = self.config.running_error_init

        # 5. Multi-scale habituation trackers
        self.visit_counts_ep: Dict[Tuple[float, ...], int] = {}
        self.visit_counts_glob: Dict[Tuple[float, ...], int] = {}

        # 6. Running state distribution statistics and trajectory frontier tracker
        self.state_mean: np.ndarray = np.zeros(self.state_dim, dtype=np.float64)
        self.step_count: int = 0
        self.start_state: Optional[np.ndarray] = None
        self.max_dist_ep: float = 0.0

        logger.info(
            f"Initialized CuriosityEngine (state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"feature_dim={self.feature_dim}, frontier_boost={self.config.frontier_boost}x)"
        )

    def _discretize_state(self, state: np.ndarray) -> Tuple[float, ...]:
        """Discretize continuous state vectors into spatial grid keys."""
        s = np.asarray(state, dtype=np.float64).ravel()
        return tuple(np.round(s, 2).tolist())

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax implementation."""
        if x.ndim == 1:
            exps = np.exp(x - np.max(x))
            return exps / (np.sum(exps) + 1e-12)
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / (np.sum(exps, axis=-1, keepdims=True) + 1e-12)

    def compute_intrinsic_reward(
        self,
        state: Union[List[float], np.ndarray],
        action: int,
        next_state: Union[List[float], np.ndarray],
    ) -> float:
        """
        Compute intrinsic exploration reward bonus for transition (state, action, next_state).
        
        Returns:
            Normalized non-negative float bonus between 0.0 and 1.0.
        """
        s = np.asarray(state, dtype=np.float64).ravel()
        s_next = np.asarray(next_state, dtype=np.float64).ravel()

        if self.start_state is None:
            self.start_state = s.copy()

        # 1. Orthogonal linear RND feature prediction error with adaptive scale
        target_feat = s_next @ self.target_matrix
        pred_feat = s_next @ self.online_matrix
        raw_pred_err = float(np.mean((target_feat - pred_feat) ** 2))
        scaled_novelty = raw_pred_err / (self.running_error_mean + 1e-6)

        # 2. Inverse Dynamics Model Controllability & Transition Rejection for Noisy-TV
        ds = s_next - s
        ds_norm_sq = float(np.sum(ds ** 2))

        logits = ds @ self.idm_matrix
        probs = self._softmax(logits)
        chance_prob = 1.0 / float(max(1, self.action_dim))
        p_action = float(probs[action]) if (0 <= action < self.action_dim) else chance_prob

        floor = self.config.controllability_floor
        controllability = floor + (1.0 - floor) * max(0.0, (p_action - chance_prob) / (1.0 - chance_prob + 1e-6))
        motion_factor = 0.15 if ds_norm_sq < 1e-5 else 1.0
        locality_rejection = float(np.exp(-0.3 * ds_norm_sq)) * motion_factor

        noise_rejection = controllability * locality_rejection

        # 3. Trajectory Frontier Expansion & Outward Progress Multiplier
        dist_mean = float(np.linalg.norm(s_next - self.state_mean))
        dist_start = float(np.linalg.norm(s_next - self.start_state))
        dist_prev = float(np.linalg.norm(s - self.start_state))

        frontier_boost = 1.0
        if dist_start > self.max_dist_ep + 1e-5:
            frontier_boost = self.config.frontier_boost
            self.max_dist_ep = dist_start

        outward_progress = max(0.0, dist_start - dist_prev)
        outward_boost = 1.0 + self.config.outward_boost_scale * np.tanh(2.5 * outward_progress)

        dist_factor = (1.0 + 0.35 * dist_start + 0.4 * np.tanh(dist_mean)) * frontier_boost * outward_boost

        # 4. Multi-scale Habituation Decay with Graduated First-Visit Episodic & Global Boost
        s_key = self._discretize_state(s)
        ns_key = self._discretize_state(s_next)

        c_ep_ns = self.visit_counts_ep.get(ns_key, 0) + 1
        c_glob_ns = self.visit_counts_glob.get(ns_key, 0) + 1
        self.visit_counts_ep[ns_key] = c_ep_ns

        global_boost = self.config.global_boost_max if c_glob_ns == 1 else (1.3 if c_glob_ns <= 3 else 1.0)
        episodic_boost = 1.75 if c_ep_ns == 1 else 1.0
        episodic_decay = episodic_boost / (c_ep_ns ** self.config.episodic_decay_power)

        habituation_factor = episodic_decay * global_boost

        intrinsic_reward = 0.48 * scaled_novelty * dist_factor * noise_rejection * habituation_factor

        if not np.isfinite(intrinsic_reward) or intrinsic_reward < 0.0:
            return 0.0
        return float(min(intrinsic_reward, 1.0))

    def update(
        self,
        states: Union[List[List[float]], np.ndarray],
        actions: Union[List[int], np.ndarray],
        next_states: Union[List[List[float]], np.ndarray],
    ) -> Dict[str, float]:
        """
        Update online predictor matrix, IDM matrix, error statistics, and state distributions.
        
        Args:
            states: Batch of states [B, state_dim]
            actions: Batch of actions [B]
            next_states: Batch of next states [B, state_dim]
            
        Returns:
            Dictionary containing loss and metric summaries.
        """
        batch_size = len(next_states)
        if batch_size == 0:
            return {"loss": 0.0, "rnd_loss": 0.0}

        s_arr = np.asarray(states, dtype=np.float64)
        ns_arr = np.asarray(next_states, dtype=np.float64)
        a_arr = np.asarray(actions, dtype=np.int32)

        if ns_arr.ndim == 1:
            ns_arr = ns_arr.reshape(batch_size, -1)
            s_arr = s_arr.reshape(batch_size, -1)

        # 1. Update running state distribution mean & global visit counts
        for ns in ns_arr:
            self.step_count += 1
            self.state_mean += (ns.ravel() - self.state_mean) / float(self.step_count)
            ns_key = self._discretize_state(ns)
            self.visit_counts_glob[ns_key] = self.visit_counts_glob.get(ns_key, 0) + 1

        # 2. Vectorized RND Predictor Gradient Step
        target_feat = ns_arr @ self.target_matrix
        pred_feat = ns_arr @ self.online_matrix
        diff = pred_feat - target_feat

        grad_rnd = (ns_arr.T @ diff) / float(batch_size)
        self.online_matrix -= self.learning_rate * grad_rnd

        rnd_loss = float(np.mean(diff ** 2))
        self.running_error_mean = 0.95 * self.running_error_mean + 0.05 * rnd_loss

        # 3. Vectorized Inverse Dynamics Model (IDM) Cross-Entropy Gradient Step
        ds_arr = ns_arr - s_arr
        logits = ds_arr @ self.idm_matrix
        probs = self._softmax(logits)

        # Construct one-hot action ground truth
        a_onehot = np.zeros((batch_size, self.action_dim), dtype=np.float64)
        valid_mask = (a_arr >= 0) & (a_arr < self.action_dim)
        if np.any(valid_mask):
            a_onehot[np.arange(batch_size)[valid_mask], a_arr[valid_mask]] = 1.0

        grad_idm = (ds_arr.T @ (probs - a_onehot)) / float(batch_size)
        self.idm_matrix -= self.learning_rate * grad_idm

        return {
            "loss": rnd_loss,
            "rnd_loss": rnd_loss,
            "running_error_mean": self.running_error_mean,
        }

    def reset_episode(self) -> None:
        """Reset episode-specific visit count buffer and trajectory frontier tracker."""
        self.visit_counts_ep.clear()
        self.start_state = None
        self.max_dist_ep = 0.0

    def reset(self) -> None:
        """Alias for reset_episode()."""
        self.reset_episode()

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational runtime metrics of the curiosity module."""
        return {
            "unique_states_global": len(self.visit_counts_glob),
            "unique_states_episode": len(self.visit_counts_ep),
            "running_error_mean": float(self.running_error_mean),
            "max_distance_episode": float(self.max_dist_ep),
            "total_transition_steps": int(self.step_count),
        }


# OpenEvolve and cross-module compatibility aliases
CuriosityRewardModule = CuriosityEngine


def create_curiosity_module(state_dim: int, action_dim: int, seed: int = 42) -> CuriosityEngine:
    """Factory function matching OpenEvolve evaluator interface."""
    return CuriosityEngine(state_dim=state_dim, action_dim=action_dim, seed=seed)

