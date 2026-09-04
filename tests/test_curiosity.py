"""
Unit tests for the Evolved Intrinsic Curiosity Engine and DQNAgent integration.
"""

import numpy as np
import pytest
import torch

from src.core.curiosity import CuriosityEngine
from src.core.agent import DQNAgent
from src.schema.dqn import CuriosityConfig, DQNConfig


class TestCuriosityEngine:
    """Test suite for standalone CuriosityEngine operations."""

    def test_curiosity_initialization(self):
        config = CuriosityConfig(enabled=True, feature_dim=32)
        engine = CuriosityEngine(state_dim=10, action_dim=4, config=config)

        assert engine.state_dim == 10
        assert engine.action_dim == 4
        assert engine.feature_dim == 32
        assert engine.target_matrix.shape == (10, 32)
        assert engine.online_matrix.shape == (10, 32)
        assert engine.idm_matrix.shape == (10, 4)

        metrics = engine.get_metrics()
        assert metrics["unique_states_global"] == 0
        assert metrics["total_transition_steps"] == 0

    def test_habituation_decay_on_repetition(self):
        """Repeated exposure to identical transitions must decay curiosity reward toward 0."""
        engine = CuriosityEngine(state_dim=4, action_dim=2, config=CuriosityConfig(enabled=True))
        state = np.array([0.5, -0.2, 0.8, -0.4], dtype=np.float64)
        action = 0
        next_state = np.array([0.6, -0.1, 0.7, -0.3], dtype=np.float64)

        r_initial = engine.compute_intrinsic_reward(state, action, next_state)
        assert r_initial > 0.0

        # Repeatedly update with identical transition 40 times
        for _ in range(40):
            engine.update(np.array([state]), np.array([action]), np.array([next_state]))

        r_final = engine.compute_intrinsic_reward(state, action, next_state)
        assert r_final < r_initial * 0.05, f"Expected >95% habituation decay, got r_initial={r_initial}, r_final={r_final}"

    def test_noisy_tv_immunity(self):
        """Pure white-noise transitions must receive lower bonus than structured novel states."""
        engine = CuriosityEngine(state_dim=4, action_dim=2, config=CuriosityConfig(enabled=True))
        rng = np.random.default_rng(123)

        # 1. Train on 30 structured, predictable transitions
        for step in range(30):
            s_struct = np.sin(np.array([step * 0.1, step * 0.2, step * 0.05, step * 0.3]))
            act = step % 2
            ns_struct = s_struct + 0.1 * (act + 1)
            engine.update(np.array([s_struct]), np.array([act]), np.array([ns_struct]))

        # Test novel structured state
        s_novel = np.sin(np.array([31 * 0.1, 31 * 0.2, 31 * 0.05, 31 * 0.3]))
        ns_novel = s_novel + 0.1
        r_structured = float(engine.compute_intrinsic_reward(s_novel, 0, ns_novel))

        # 2. Evaluate pure stochastic white noise transitions
        noise_rewards = []
        for _ in range(30):
            s_noise = rng.standard_normal(4)
            act = int(rng.integers(0, 2))
            ns_noise = rng.standard_normal(4)  # Completely unpredictable Gaussian noise
            engine.update(np.array([s_noise]), np.array([act]), np.array([ns_noise]))
            noise_rewards.append(float(engine.compute_intrinsic_reward(s_noise, act, ns_noise)))

        avg_noise_reward = float(np.mean(noise_rewards))
        assert r_structured > avg_noise_reward, (
            f"Expected structured novel reward ({r_structured}) > pure noise reward ({avg_noise_reward})"
        )

    def test_frontier_expansion_boost(self):
        """Disclosing states that exceed the maximum episode distance frontier triggers frontier boost."""
        config = CuriosityConfig(enabled=True, frontier_boost=1.75)
        engine = CuriosityEngine(state_dim=4, action_dim=2, config=config)

        start = np.array([0.0, 0.0, 0.0, 0.0])
        step1 = np.array([0.5, 0.0, 0.0, 0.0])
        step2 = np.array([1.5, 0.0, 0.0, 0.0])  # Breaks distance record

        r1 = engine.compute_intrinsic_reward(start, 0, step1)
        r2 = engine.compute_intrinsic_reward(step1, 0, step2)

        assert engine.max_dist_ep >= 1.5
        assert r2 > 0.0

    def test_episodic_reset(self):
        """Resetting episode clears episodic visit counts and start state."""
        engine = CuriosityEngine(state_dim=4, action_dim=2)
        s = np.array([0.1, 0.2, 0.3, 0.4])
        ns = np.array([0.5, 0.6, 0.7, 0.8])

        engine.compute_intrinsic_reward(s, 0, ns)
        assert len(engine.visit_counts_ep) > 0
        assert engine.start_state is not None

        engine.reset_episode()
        assert len(engine.visit_counts_ep) == 0
        assert engine.start_state is None
        assert engine.max_dist_ep == 0.0


class TestDQNAgentCuriosityIntegration:
    """Test suite for DQNAgent with CuriosityEngine integration."""

    def test_agent_curiosity_disabled_by_default(self):
        config = DQNConfig(state_size=4, action_size=2)
        agent = DQNAgent(config)

        assert agent.curiosity is None
        assert agent.get_curiosity_metrics() is None

    def test_agent_curiosity_enabled(self):
        curiosity_cfg = CuriosityConfig(
            enabled=True,
            intrinsic_reward_scale=0.5,
            feature_dim=16,
        )
        config = DQNConfig(
            state_size=4,
            action_size=2,
            batch_size=8,
            memory_capacity=100,
            train_interval=1,
            curiosity=curiosity_cfg,
        )
        agent = DQNAgent(config)

        assert agent.curiosity is not None
        assert agent.curiosity.feature_dim == 16

        # Feed transitions with 0 external reward
        for i in range(15):
            s = [float(i) * 0.1, 0.0, 0.0, 0.0]
            ns = [float(i + 1) * 0.1, 0.0, 0.0, 0.0]
            agent.remember(s, action=0, reward=0.0, next_state=ns, done=False)

        # Check that stored reward in memory is > 0 due to intrinsic curiosity bonus
        sample_r = agent.memory._rewards[:10]
        assert np.any(sample_r > 0.0), "Expected intrinsic curiosity bonus to be added to rewards"

        # Execute training batch
        agent.train_batch()
        metrics = agent.get_curiosity_metrics()
        assert metrics is not None
        assert metrics["total_transition_steps"] > 0
        assert metrics["unique_states_global"] > 0

    def test_curiosity_engine_matches_openevolve_champion_evaluation(self):
        """Verify that src/core/curiosity.py reproduces the OpenEvolve champion evaluation score."""
        from evolution.curiosity_optimizer.evaluator import evaluate
        res = evaluate("/home/naimul/alpha-dqn/src/core/curiosity.py")
        metrics = res.metrics
        assert metrics["combined_score"] >= 0.69, f"Expected combined >= 0.69, got {metrics['combined_score']}"
        assert metrics["goal_reach_rate"] >= 0.65, f"Expected goal_reach >= 0.65, got {metrics['goal_reach_rate']}"
        assert metrics["noisy_tv_immunity"] >= 0.90, f"Expected noise_immunity >= 0.90, got {metrics['noisy_tv_immunity']}"
        assert metrics["habituation_decay"] >= 0.99, f"Expected habituation >= 0.99, got {metrics['habituation_decay']}"
        assert metrics["state_coverage"] >= 0.20, f"Expected coverage >= 0.20, got {metrics['state_coverage']}"

