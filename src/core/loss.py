"""
Adaptive Pseudo-Huber Expectile Loss Module.
Evolved and discovered via OpenEvolve (Fitness: 0.9844, Asymmetric Optimism: 0.9975).

Combines:
  1. Pseudo-Huber Loss (Charbonnier formulation): C-infinity smooth gradient transition everywhere without kinks.
  2. Asymmetric Expectile Weighting (tau=0.598): 1.487x higher penalty for underestimation errors.
  3. Robust MAD (Median Absolute Deviation) Scale Estimation: Outlier-resistant scale tracking.
  4. Dynamic Exponential Moving Average (EMA) Beta-Adaptation: Error-scale curvature normalization.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn

from src.config import Settings, setup_logger
from src.schema.dqn import AdaptiveLossConfig

logger = setup_logger(Settings.LOG_DIR / "core.log", name="alphadqn.core.loss")


class AdaptiveLoss(nn.Module):
    """
    Production-grade Adaptive Loss for Deep Q-Networks.
    Discovered through autonomous multi-island evolutionary optimization.
    """

    def __init__(self, config: Optional[AdaptiveLossConfig] = None, seed: int = 42):
        super().__init__()
        self.config: AdaptiveLossConfig = config or AdaptiveLossConfig()
        self.rng = np.random.default_rng(seed)

        # Dynamic curvature and asymmetric parameters
        self.beta: float = float(self.config.initial_beta)
        self.tau: float = float(self.config.tau)
        self.running_td_mean: float = 0.0
        self.running_td_var: float = 1.0
        self.step_count: int = 0

        logger.info(
            f"Initialized AdaptiveLoss (beta={self.beta:.4f}, tau={self.tau:.3f}, alpha={self.config.ema_alpha})"
        )

    def forward(
        self,
        current_q: torch.Tensor,
        target_q: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Alias for compute_loss() matching standard PyTorch nn.Module interface."""
        return self.compute_loss(current_q, target_q, weights=weights)

    def compute_loss(
        self,
        current_q: torch.Tensor,
        target_q: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scalar differentiable Pseudo-Huber expectile loss.

        Args:
            current_q: Predicted Q-values for chosen actions [batch_size]
            target_q: Double-DQN Bellman targets [batch_size]
            weights: Optional importance-sampling weights [batch_size]

        Returns:
            Differentiable scalar PyTorch loss tensor
        """
        td_errors = current_q - target_q
        beta = max(1e-4, float(self.beta))

        # 1. Smooth Pseudo-Huber formulation (C-infinity smooth gradient transition)
        huber_loss = torch.sqrt(td_errors ** 2 + beta ** 2) - beta

        # 2. Asymmetric expectile weighting (optimistic penalty for underestimation)
        # When td_errors < 0 (underestimation: current_q < target_q), weight = tau (~0.598)
        # When td_errors >= 0 (overestimation: current_q >= target_q), weight = 1 - tau (~0.402)
        asym_weight = torch.where(td_errors < 0, self.tau, 1.0 - self.tau)
        per_sample_loss = 2.0 * asym_weight * huber_loss

        if weights is not None:
            per_sample_loss = per_sample_loss * weights

        return torch.mean(per_sample_loss)

    def update_stats(self, td_errors: np.ndarray) -> Dict[str, float]:
        """
        Update running statistics using robust estimators (MAD & variance).

        Args:
            td_errors: 1D NumPy array of TD error residuals (current_q - target_q)
        """
        if len(td_errors) == 0:
            return {"beta": self.beta}

        arr = np.asarray(td_errors, dtype=np.float64).ravel()
        batch_mean = float(np.mean(arr))
        batch_var = float(np.var(arr))

        # Robust scale estimation via Median Absolute Deviation (MAD)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        robust_std = 1.4826 * mad
        scale = robust_std if robust_std > 1e-4 else np.sqrt(max(1e-6, batch_var))

        self.step_count += 1
        alpha = self.config.ema_alpha
        self.running_td_mean = (1.0 - alpha) * self.running_td_mean + alpha * batch_mean
        self.running_td_var = (1.0 - alpha) * self.running_td_var + alpha * batch_var

        # Adaptively scale beta to match error scale
        target_beta = float(
            np.clip(
                self.config.beta_scale * scale,
                self.config.min_beta,
                self.config.max_beta,
            )
        )
        if self.step_count == 1:
            self.beta = target_beta
        else:
            self.beta = (1.0 - alpha) * self.beta + alpha * target_beta

        return {
            "beta": float(self.beta),
            "running_td_mean": float(self.running_td_mean),
            "running_td_var": float(self.running_td_var),
        }

    def reset(self) -> None:
        """Reset running statistics to initial state."""
        self.beta = float(self.config.initial_beta)
        self.running_td_mean = 0.0
        self.running_td_var = 1.0
        self.step_count = 0

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational runtime metrics of the adaptive loss function."""
        return {
            "current_beta": float(self.beta),
            "tau_optimism": float(self.tau),
            "running_td_mean": float(self.running_td_mean),
            "running_td_var": float(self.running_td_var),
            "total_updates": int(self.step_count),
        }


# OpenEvolve and cross-module compatibility aliases
AdaptiveLossModule = AdaptiveLoss


def create_loss_module(seed: int = 42) -> AdaptiveLoss:
    """Factory function matching OpenEvolve evaluator interface."""
    return AdaptiveLoss(seed=seed)
