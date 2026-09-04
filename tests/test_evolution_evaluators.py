"""
Unit and Determinism Tests for OpenEvolve Evaluators:
  1. Adaptive Loss Function Optimizer Evaluator
  2. Novel Exploration Strategy Optimizer Evaluator
"""

import pytest
from evolution.loss_optimizer.evaluator import evaluate as evaluate_loss, evaluate_stage1 as stage1_loss
from evolution.exploration_optimizer.evaluator import evaluate as evaluate_expl, evaluate_stage1 as stage1_expl


class TestAdaptiveLossEvaluator:
    """Tests for Phase 1 Adaptive Loss Evaluator."""

    def test_loss_evaluator_stage1(self):
        res = stage1_loss("evolution/loss_optimizer/initial_program.py")
        assert res.metrics["combined_score"] > 0.0
        assert "error" not in res.artifacts

    def test_loss_evaluator_full_4_pillars(self):
        res = evaluate_loss("evolution/loss_optimizer/initial_program.py")
        metrics = res.metrics

        assert "combined_score" in metrics
        assert 0.0 <= metrics["combined_score"] <= 1.0
        assert 0.0 <= metrics["convergence_speed"] <= 1.0
        assert 0.0 <= metrics["gradient_stability"] <= 1.0
        assert 0.0 <= metrics["asymmetric_optimism"] <= 1.0
        assert 0.0 <= metrics["curvature_adaptability"] <= 1.0

    def test_loss_evaluator_determinism(self):
        """Must produce 100% reproducible deterministic scores on identical seed."""
        res1 = evaluate_loss("evolution/loss_optimizer/initial_program.py")
        res2 = evaluate_loss("evolution/loss_optimizer/initial_program.py")

        assert res1.metrics["combined_score"] == pytest.approx(res2.metrics["combined_score"], abs=1e-7)
        assert res1.metrics["convergence_speed"] == pytest.approx(res2.metrics["convergence_speed"], abs=1e-7)
        assert res1.metrics["gradient_stability"] == pytest.approx(res2.metrics["gradient_stability"], abs=1e-7)


class TestExplorationEvaluator:
    """Tests for Phase 2 Novel Exploration Evaluator."""

    def test_exploration_evaluator_stage1(self):
        res = stage1_expl("evolution/exploration_optimizer/initial_program.py")
        assert res.metrics["combined_score"] > 0.0
        assert "error" not in res.artifacts

    def test_exploration_evaluator_full_4_pillars(self):
        res = evaluate_expl("evolution/exploration_optimizer/initial_program.py")
        metrics = res.metrics

        assert "combined_score" in metrics
        assert 0.0 <= metrics["combined_score"] <= 1.0
        assert 0.0 <= metrics["diffusion_speed"] <= 1.0
        assert 0.0 <= metrics["state_entropy"] <= 1.0
        assert 0.0 <= metrics["action_coherence"] <= 1.0
        assert 0.0 <= metrics["exploit_accuracy"] <= 1.0

    def test_exploration_evaluator_determinism(self):
        """Must produce 100% reproducible deterministic scores on identical seed."""
        res1 = evaluate_expl("evolution/exploration_optimizer/initial_program.py")
        res2 = evaluate_expl("evolution/exploration_optimizer/initial_program.py")

        assert res1.metrics["combined_score"] == pytest.approx(res2.metrics["combined_score"], abs=1e-7)
        assert res1.metrics["diffusion_speed"] == pytest.approx(res2.metrics["diffusion_speed"], abs=1e-7)
        assert res1.metrics["state_entropy"] == pytest.approx(res2.metrics["state_entropy"], abs=1e-7)
