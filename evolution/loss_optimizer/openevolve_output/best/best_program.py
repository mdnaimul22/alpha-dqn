"""
Initial Evolvable Adaptive Loss Module for OpenEvolve.

Defines the loss formulation for training the Deep Q-Network (Double-DQN).
OpenEvolve will mutate, refine, and optimize this module to discover novel
loss formulations that maximize Bellman convergence speed, maintain robustness
against outlier TD spikes, dynamically adapt curvature, and encourage optimistic value estimation.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import torch
import torch.nn as nn


class AdaptiveLossModule:
    """
    Evolvable Bellman Loss Module for Deep Q-Networks.
    Baseline implementation: Smooth L1 (Huber Loss) with running error variance tracking.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        
        # Adaptive curvature & asymmetric parameters
        self.beta = 1.0
        self.tau = 0.598  # Asymmetric optimism factor targeting optimal gradient ratio ~1.487
        self.running_td_mean = 0.0
        self.running_td_var = 1.0
        self.step_count = 0

    def compute_loss(
        self,
        current_q: torch.Tensor,
        target_q: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scalar differentiable loss between predicted Q-values and Bellman targets.

        Args:
            current_q: Predicted Q-values for chosen actions [batch_size]
            target_q: Computed Double-DQN Bellman targets [batch_size]
            weights: Optional importance-sampling weights [batch_size]

        Returns:
            Scalar PyTorch loss tensor suitable for backward()
        """
        td_errors = current_q - target_q
        beta = max(1e-4, float(self.beta))

        # Smooth Pseudo-Huber formulation (C-infinity smooth gradient transition)
        huber_loss = torch.sqrt(td_errors ** 2 + beta ** 2) - beta

        # Asymmetric expectile weighting (optimistic penalty for underestimation)
        asym_weight = torch.where(td_errors < 0, self.tau, 1.0 - self.tau)
        per_sample_loss = 2.0 * asym_weight * huber_loss

        if weights is not None:
            per_sample_loss = per_sample_loss * weights

        return torch.mean(per_sample_loss)

    def update_stats(self, td_errors: np.ndarray) -> dict:
        """
        Update running statistics using robust estimators (MAD & variance).

        Args:
            td_errors: 1D NumPy array of TD error residuals (current_q - target_q)
        """
        if len(td_errors) == 0:
            return {"beta": self.beta}

        arr = np.asarray(td_errors, dtype=np.float64)
        batch_mean = float(np.mean(arr))
        batch_var = float(np.var(arr))

        # Robust scale estimation via Median Absolute Deviation (MAD)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        robust_std = 1.4826 * mad
        scale = robust_std if robust_std > 1e-4 else np.sqrt(max(1e-6, batch_var))

        self.step_count += 1
        alpha = 0.43
        self.running_td_mean = (1.0 - alpha) * self.running_td_mean + alpha * batch_mean
        self.running_td_var = (1.0 - alpha) * self.running_td_var + alpha * batch_var

        # Adaptively scale beta to match error scale (0.82x robust scale for balanced convergence & stability)
        target_beta = float(np.clip(0.82 * scale, 0.007, 20.0))
        if self.step_count == 1:
            self.beta = target_beta
        else:
            self.beta = (1.0 - alpha) * self.beta + alpha * target_beta

        return {
            "beta": self.beta,
            "running_td_mean": self.running_td_mean,
            "running_td_var": self.running_td_var,
        }

    def reset(self) -> None:
        """Reset running statistics to initial values."""
        self.beta = 1.0
        self.tau = 0.598
        self.running_td_mean = 0.0
        self.running_td_var = 1.0
        self.step_count = 0


def create_loss_module(seed: int = 42) -> AdaptiveLossModule:
    """Factory function required by OpenEvolve evaluator."""
    return AdaptiveLossModule(seed=seed)


if __name__ == "__main__":
    mod = create_loss_module(seed=42)
    pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = torch.tensor([1.2, 1.8, 4.5])
    loss = mod.compute_loss(pred, target)
    loss.backward()
    print(f"Sample Loss: {loss.item():.6f}")
    print(f"Gradients: {pred.grad}")
    stats = mod.update_stats((pred - target).detach().numpy())
    print(f"Updated Stats: {stats}")
