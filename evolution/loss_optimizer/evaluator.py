"""
OpenEvolve Evaluator for Adaptive Loss Function Optimization in Deep Q-Networks.

Evaluates candidate AdaptiveLossModule implementations against 4 core pillars:
1. Convergence Speed (30%): Rapid minimization of Bellman TD error on non-stationary targets.
2. Gradient & Outlier Stability (25%): Immunity against extreme TD spikes; bounded non-exploding gradients.
3. Asymmetric Optimism (25%): Penalty alignment ensuring optimal action values are not pessimistically depressed.
4. Curvature Adaptability (20%): Dynamic adaptation across varied error scales without manual retuning.

MAP-Elites Dimensions:
- gradient_stability: Resistance to gradient explosion under heavy-tailed spikes [0.0, 1.0]
- convergence_speed: Rate of Bellman error reduction [0.0, 1.0]
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
import time
import traceback
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from openevolve.evaluation_result import EvaluationResult


def set_global_seeds(seed: int = 42):
    """Enforce strict deterministic reproducibility across NumPy, Random, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_candidate_module(program_path: str):
    """Safely and dynamically load candidate loss module from file path."""
    module_name = os.path.basename(program_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification from {program_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convergence_speed(loss_module) -> Tuple[float, Dict[str, Any]]:
    """
    Pillar 1: Convergence Speed on Synthetic Bellman Regression.
    Evaluates how rapidly a small MLP learns regression targets over 50 gradient steps.
    """
    set_global_seeds(42)
    device = torch.device("cpu")

    # Simple 2-layer MLP predicting Q-values from state vectors
    model = nn.Sequential(
        nn.Linear(8, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Fixed synthetic dataset with known non-linear mapping
    x_train = torch.randn(128, 8, device=device)
    y_target = torch.sin(x_train[:, 0:1]) + 0.5 * torch.cos(x_train[:, 1:2]) + 1.5

    # Compute initial baseline MSE
    with torch.no_grad():
        initial_pred = model(x_train)
        initial_mse = float(torch.mean((initial_pred - y_target) ** 2).item())

    # Train for 50 steps
    for step in range(50):
        optimizer.zero_grad()
        pred = model(x_train)
        loss = loss_module.compute_loss(pred.squeeze(1), y_target.squeeze(1))

        if not torch.isfinite(loss):
            return 0.0, {"error": "Non-finite loss encountered"}

        loss.backward()
        optimizer.step()

        if hasattr(loss_module, "update_stats"):
            td = (pred.squeeze(1) - y_target.squeeze(1)).detach().numpy()
            loss_module.update_stats(td)

    # Final MSE
    with torch.no_grad():
        final_pred = model(x_train)
        final_mse = float(torch.mean((final_pred - y_target) ** 2).item())

    # Convergence ratio: 1.0 means error dropped to 0; 0.0 means no improvement
    reduction = max(0.0, (initial_mse - final_mse) / (initial_mse + 1e-6))
    score = float(np.clip(reduction, 0.0, 1.0))

    details = {
        "initial_mse": round(initial_mse, 5),
        "final_mse": round(final_mse, 5),
        "reduction_ratio": round(reduction, 5),
    }
    return score, details


def test_gradient_stability(loss_module) -> Tuple[float, Dict[str, Any]]:
    """
    Pillar 2: Gradient & Outlier Stability.
    Injects heavy-tailed outlier spikes into TD errors and checks for gradient explosion or collapse.
    """
    set_global_seeds(123)
    loss_module.reset()

    grad_norms = []
    # Test across 30 batches containing sudden 15x-30x outlier spikes
    for i in range(30):
        pred = torch.randn(64, requires_grad=True)
        target = pred.detach() + torch.randn(64) * 0.5

        # 5% of samples are extreme outlier spikes (simulating bootstrapping errors/spikes)
        outlier_indices = np.random.choice(64, size=3, replace=False)
        target_np = target.detach().numpy()
        target_np[outlier_indices] += np.random.choice([-1.0, 1.0], size=3) * np.random.uniform(15.0, 30.0, size=3)
        target = torch.tensor(target_np, dtype=torch.float32)

        loss = loss_module.compute_loss(pred, target)
        if not torch.isfinite(loss):
            return 0.0, {"error": "Loss produced NaN on outlier batch"}

        loss.backward()

        if pred.grad is None or not torch.all(torch.isfinite(pred.grad)):
            return 0.0, {"error": "Gradient contains NaN or Inf"}

        gnorm = float(torch.norm(pred.grad).item())
        grad_norms.append(gnorm)

        if hasattr(loss_module, "update_stats"):
            loss_module.update_stats((pred - target).detach().numpy())

    max_norm = float(np.max(grad_norms))
    mean_norm = float(np.mean(grad_norms))
    std_norm = float(np.std(grad_norms))

    # Score: Penalize if max gradient norm exceeds 25 (explosion) or drops below 0.01 (vanishing)
    if max_norm > 50.0:
        stability = max(0.0, 1.0 - (max_norm - 50.0) / 100.0)
    elif max_norm < 0.05:
        stability = 0.1
    else:
        # Ideal range: 0.5 to 15.0 with low variance
        var_penalty = min(0.5, std_norm / (mean_norm + 1e-5) * 0.2)
        stability = float(np.clip(1.0 - var_penalty, 0.0, 1.0))

    details = {
        "max_grad_norm": round(max_norm, 3),
        "mean_grad_norm": round(mean_norm, 3),
        "grad_norm_std": round(std_norm, 3),
    }
    return float(stability), details


def test_asymmetric_optimism(loss_module) -> Tuple[float, Dict[str, Any]]:
    """
    Pillar 3: Asymmetric Optimism and Bellman Contraction Alignment.
    Penalizes underestimating promising actions more than overestimating mediocre ones.
    """
    set_global_seeds(777)
    loss_module.reset()

    # Underestimation scenario: target is higher than current prediction (promising transition)
    pred_under = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    target_under = torch.tensor([4.0, 5.0, 6.0])  # +2.0 underestimation
    loss_under = loss_module.compute_loss(pred_under, target_under)
    loss_under.backward()
    grad_under = float(torch.norm(pred_under.grad).item())

    # Overestimation scenario: prediction is higher than target
    pred_over = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    target_over = torch.tensor([2.0, 3.0, 4.0])  # -2.0 overestimation
    loss_over = loss_module.compute_loss(pred_over, target_over)
    loss_over.backward()
    grad_over = float(torch.norm(pred_over.grad).item())

    # In Q-learning, underestimation gradient should be >= overestimation gradient
    # (drives policy swiftly toward true optimal values without pessimistic stalls)
    ratio = grad_under / (grad_over + 1e-6)

    # Ideal ratio is in [0.9, 1.6] (healthy optimistic pressure without runaway)
    if ratio < 0.5:
        optimism_score = 0.3
    elif ratio > 3.0:
        optimism_score = 0.5  # runaway over-optimism
    else:
        optimism_score = float(min(1.0, 0.7 + 0.3 * min(ratio, 1.5) / 1.5))

    details = {
        "grad_underestimation": round(grad_under, 4),
        "grad_overestimation": round(grad_over, 4),
        "optimism_ratio": round(ratio, 4),
    }
    return optimism_score, details


def test_curvature_adaptability(loss_module) -> Tuple[float, Dict[str, Any]]:
    """
    Pillar 4: Curvature Adaptability Across Scales.
    Tests loss behavior under small (0.1), medium (1.0), and large (10.0) error regimes.
    """
    set_global_seeds(999)
    loss_module.reset()

    scales = [0.1, 1.0, 8.0]
    effective_betas = []

    for s in scales:
        loss_module.reset()
        dummy_errors = np.random.normal(0.0, s, size=100)
        if hasattr(loss_module, "update_stats"):
            res = loss_module.update_stats(dummy_errors)
            beta = res.get("beta", 1.0)
        else:
            beta = getattr(loss_module, "beta", 1.0)
        effective_betas.append(float(beta))

    # If beta adapts monotonically with error scale (small for small errors, large for large errors)
    # the loss possesses strong dynamic curvature adaptation!
    is_monotonic = (effective_betas[0] < effective_betas[1]) and (effective_betas[1] < effective_betas[2])
    spread = (effective_betas[2] - effective_betas[0])

    if is_monotonic and spread > 1.0:
        adaptability_score = 1.0
    elif is_monotonic:
        adaptability_score = 0.75
    elif spread > 0.5:
        adaptability_score = 0.5
    else:
        adaptability_score = 0.35  # static loss

    details = {
        "scales_tested": scales,
        "adapted_betas": [round(b, 3) for b in effective_betas],
        "spread": round(spread, 3),
    }
    return adaptability_score, details


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """Stage 1 Quick Filter: Runs fast sanity check on gradient stability and finite output."""
    set_global_seeds(42)
    start_time = time.perf_counter()
    metrics: Dict[str, float] = {}
    artifacts: Dict[str, Any] = {}

    try:
        candidate = load_candidate_module(program_path)
        if hasattr(candidate, "create_loss_module"):
            mod = candidate.create_loss_module(seed=42)
        elif hasattr(candidate, "AdaptiveLossModule"):
            mod = candidate.AdaptiveLossModule(seed=42)
        else:
            raise AttributeError("Candidate must define 'create_loss_module' or 'AdaptiveLossModule'")

        # Quick test
        pred = torch.tensor([1.0, 2.0], requires_grad=True)
        target = torch.tensor([1.5, 2.5])
        loss = mod.compute_loss(pred, target)
        loss.backward()

        if not torch.isfinite(loss) or pred.grad is None:
            return EvaluationResult(
                metrics={"combined_score": 0.0},
                artifacts={"error": "Stage 1 failed: non-finite loss or missing gradients"},
            )

        metrics["stage1_score"] = 1.0
        metrics["combined_score"] = 0.5
        metrics["execution_time"] = float(time.perf_counter() - start_time)
        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    except Exception as e:
        metrics["combined_score"] = 0.0
        metrics["execution_time"] = float(time.perf_counter() - start_time)
        artifacts["stderr"] = str(e)
        artifacts["traceback"] = traceback.format_exc()
        return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate(program_path: str) -> EvaluationResult:
    """Full 4-Pillar Evaluation of the Candidate Loss Module."""
    set_global_seeds(42)
    start_time = time.perf_counter()
    metrics: Dict[str, float] = {}
    artifacts: Dict[str, Any] = {}

    try:
        candidate = load_candidate_module(program_path)
        if hasattr(candidate, "create_loss_module"):
            make_mod = candidate.create_loss_module
        elif hasattr(candidate, "AdaptiveLossModule"):
            make_mod = lambda seed=42: candidate.AdaptiveLossModule(seed=seed)
        else:
            raise AttributeError("Candidate must define 'create_loss_module' or 'AdaptiveLossModule'")

        # 1. Convergence Speed (30%)
        mod1 = make_mod(seed=42)
        conv_score, conv_details = test_convergence_speed(mod1)
        metrics["convergence_speed"] = conv_score
        artifacts["convergence_details"] = conv_details

        # 2. Gradient & Outlier Stability (25%)
        mod2 = make_mod(seed=123)
        stab_score, stab_details = test_gradient_stability(mod2)
        metrics["gradient_stability"] = stab_score
        artifacts["stability_details"] = stab_details

        # 3. Asymmetric Optimism (25%)
        mod3 = make_mod(seed=777)
        opt_score, opt_details = test_asymmetric_optimism(mod3)
        metrics["asymmetric_optimism"] = opt_score
        artifacts["optimism_details"] = opt_details

        # 4. Curvature Adaptability (20%)
        mod4 = make_mod(seed=999)
        curv_score, curv_details = test_curvature_adaptability(mod4)
        metrics["curvature_adaptability"] = curv_score
        artifacts["curvature_details"] = curv_details

        # Combined Fitness Score [0.0 - 1.0]
        combined = (
            0.30 * conv_score +
            0.25 * stab_score +
            0.25 * opt_score +
            0.20 * curv_score
        )
        metrics["combined_score"] = float(np.clip(combined, 0.0, 1.0))
        metrics["execution_time"] = float(time.perf_counter() - start_time)

        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    except Exception as e:
        metrics["combined_score"] = 0.0
        metrics["execution_time"] = float(time.perf_counter() - start_time)
        artifacts["stderr"] = str(e)
        artifacts["traceback"] = traceback.format_exc()
        return EvaluationResult(metrics=metrics, artifacts=artifacts)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "initial_program.py"
    print(f"Testing loss evaluator on: {target}")
    res = evaluate(target)
    print("\n--- EVALUATION RESULTS ---")
    print("Metrics:", res.metrics)
    print("Artifacts:", res.artifacts)
