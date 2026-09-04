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
    Combines QR-Orthogonal RND Feature Prediction, Action Inverse Dynamics Controllability
    (Noisy-TV Rejection), Episodic Frontier & Outward Progress Expansion, and Multi-Scale Habituation.
    """

    def __init__(self, state_dim: int, action_dim: int, seed: int = 42):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)

        self.feature_dim = 32

        # Fixed orthogonal target matrix for linear feature projection
        q, _ = np.linalg.qr(
            self.rng.standard_normal((max(self.state_dim, self.feature_dim), self.feature_dim))
        )
        self.target_matrix = q[: self.state_dim, : self.feature_dim]

        # Trainable online RND predictor matrix
        self.online_matrix = np.zeros((self.state_dim, self.feature_dim), dtype=np.float64)

        # Trainable linear inverse dynamics matrix: delta_state -> action logits
        self.idm_matrix = np.zeros((self.state_dim, self.action_dim), dtype=np.float64)

        # Model hyperparameters
        self.learning_rate = 0.1
        self.running_error_mean = 0.02

        # Multi-scale habituation trackers
        self.visit_counts_ep = {}
        self.visit_counts_glob = {}

        # Running state distribution statistics and trajectory frontier tracker
        self.state_mean = np.zeros(self.state_dim, dtype=np.float64)
        self.step_count = 0
        self.start_state = None
        self.max_dist_ep = 0.0

    def _discretize_state(self, state: np.ndarray) -> tuple:
        """Discretize continuous state vectors into spatial grid keys."""
        s = np.asarray(state, dtype=np.float64).ravel()
        return tuple(np.round(s, 2).tolist())

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax implementation."""
        if x.ndim == 1:
            exps = np.exp(x - np.max(x))
            return exps / (np.sum(exps) + 1e-12)
        else:
            exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exps / (np.sum(exps, axis=-1, keepdims=True) + 1e-12)

    def compute_intrinsic_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray
    ) -> float:
        """Compute intrinsic bonus for transition (state, action, next_state)."""
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

        controllability = 0.25 + 0.75 * max(0.0, (p_action - chance_prob) / (1.0 - chance_prob + 1e-6))
        motion_factor = 0.15 if ds_norm_sq < 1e-5 else 1.0
        locality_rejection = float(np.exp(-0.3 * ds_norm_sq)) * motion_factor

        noise_rejection = controllability * locality_rejection

        # 3. Trajectory Frontier Expansion & Outward Progress Multiplier
        dist_mean = float(np.linalg.norm(s_next - self.state_mean))
        dist_start = float(np.linalg.norm(s_next - self.start_state))
        dist_prev = float(np.linalg.norm(s - self.start_state))

        frontier_boost = 1.0
        if dist_start > self.max_dist_ep + 1e-5:
            frontier_boost = 1.75
            self.max_dist_ep = dist_start

        outward_progress = max(0.0, dist_start - dist_prev)
        outward_boost = 1.0 + 0.75 * np.tanh(2.5 * outward_progress)

        dist_factor = (1.0 + 0.35 * dist_start + 0.4 * np.tanh(dist_mean)) * frontier_boost * outward_boost

        # 4. Multi-scale Habituation Decay with Graduated First-Visit Episodic & Global Boost
        s_key = self._discretize_state(s)
        ns_key = self._discretize_state(s_next)

        c_ep_ns = self.visit_counts_ep.get(ns_key, 0) + 1
        c_glob_ns = self.visit_counts_glob.get(ns_key, 0) + 1
        self.visit_counts_ep[ns_key] = c_ep_ns

        global_boost = 2.0 if c_glob_ns == 1 else (1.3 if c_glob_ns <= 3 else 1.0)
        episodic_boost = 1.75 if c_ep_ns == 1 else 1.0
        episodic_decay = episodic_boost / (c_ep_ns ** 1.4)

        habituation_factor = episodic_decay * global_boost

        intrinsic_reward = 0.48 * scaled_novelty * dist_factor * noise_rejection * habituation_factor

        if not np.isfinite(intrinsic_reward) or intrinsic_reward < 0.0:
            return 0.0
        return float(min(intrinsic_reward, 1.0))

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray
    ) -> dict:
        """Update online predictor matrix, IDM matrix, error stats, and state statistics."""
        batch_size = len(next_states)
        if batch_size == 0:
            return {"loss": 0.0}

        s_arr = np.asarray(states, dtype=np.float64)
        ns_arr = np.asarray(next_states, dtype=np.float64)
        a_arr = np.asarray(actions, dtype=np.int32)
        if ns_arr.ndim == 1:
            ns_arr = ns_arr.reshape(batch_size, -1)
            s_arr = s_arr.reshape(batch_size, -1)

        # Update running state mean & global visit counts
        for ns in ns_arr:
            self.step_count += 1
            self.state_mean += (ns.ravel() - self.state_mean) / float(self.step_count)
            ns_key = self._discretize_state(ns)
            self.visit_counts_glob[ns_key] = self.visit_counts_glob.get(ns_key, 0) + 1

        # RND Predictor Update
        target_feat = ns_arr @ self.target_matrix
        pred_feat = ns_arr @ self.online_matrix
        diff = pred_feat - target_feat

        grad_rnd = (ns_arr.T @ diff) / float(batch_size)
        self.online_matrix -= self.learning_rate * grad_rnd

        rnd_loss = float(np.mean(diff ** 2))
        self.running_error_mean = 0.95 * self.running_error_mean + 0.05 * rnd_loss

        # Inverse Dynamics Model (IDM) Update
        ds_arr = ns_arr - s_arr
        logits = ds_arr @ self.idm_matrix
        probs = self._softmax(logits)

        # One-hot target actions
        a_onehot = np.zeros((batch_size, self.action_dim), dtype=np.float64)
        valid_mask = (a_arr >= 0) & (a_arr < self.action_dim)
        a_onehot[np.arange(batch_size)[valid_mask], a_arr[valid_mask]] = 1.0

        grad_idm = (ds_arr.T @ (probs - a_onehot)) / float(batch_size)
        self.idm_matrix -= self.learning_rate * grad_idm

        return {"loss": rnd_loss}

    def reset(self) -> None:
        """Reset episode-specific visit count buffer and trajectory frontier tracker."""
        self.visit_counts_ep.clear()
        self.start_state = None
        self.max_dist_ep = 0.0


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
