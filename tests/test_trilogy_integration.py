"""
Integration tests for the Triad/Trilogy RL system:
  1. Curiosity Novelty Engine (ICM + Episodic RND Habituation + Directional Frontier)
  2. Adaptive Pseudo-Huber Expectile Loss (tau=0.598, Robust MAD scale estimation, EMA beta)
  3. Novel Directed Epistemic Exploration Policy (Sticky-action momentum, collision sensing, UCB)

Validates unit behaviors, end-to-end multi-step rollouts, service layer, and REST API.
"""

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.core.agent import DQNAgent
from src.core.exploration import ExplorationPolicy
from src.core.loss import AdaptiveLoss
from src.routers.dqn import router
from src.schema.dqn import (
    AdaptiveLossConfig,
    CuriosityConfig,
    DQNActRequest,
    DQNConfig,
    DQNInitRequest,
    DQNStepRequest,
    DQNTrainRequest,
    ExplorationConfig,
)
from src.services.dqn import DQNService


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Adaptive Loss Unit Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveLoss:
    def test_loss_asymmetric_expectile_property(self):
        """Underestimation (current < target) should yield higher penalty than overestimation."""
        loss_mod = AdaptiveLoss(config=AdaptiveLossConfig(initial_beta=0.5, tau=0.598))
        
        # Scenario A: underestimation by 2.0 (current=0.0, target=2.0 -> td_error = -2.0)
        curr_under = torch.tensor([0.0], dtype=torch.float32)
        tgt_under = torch.tensor([2.0], dtype=torch.float32)
        l_under = loss_mod(curr_under, tgt_under).item()
        
        # Scenario B: overestimation by 2.0 (current=2.0, target=0.0 -> td_error = +2.0)
        curr_over = torch.tensor([2.0], dtype=torch.float32)
        tgt_over = torch.tensor([0.0], dtype=torch.float32)
        l_over = loss_mod(curr_over, tgt_over).item()
        
        # Ratio should closely match tau / (1 - tau) = 0.598 / 0.402 = 1.487
        expected_ratio = 0.598 / (1.0 - 0.598)
        actual_ratio = l_under / l_over
        assert actual_ratio > 1.4
        assert abs(actual_ratio - expected_ratio) < 0.05

    def test_loss_backward_differentiability(self):
        """Loss module should compute valid non-zero gradients via autograd."""
        loss_mod = AdaptiveLoss()
        pred = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
        tgt = torch.tensor([1.5, 1.0, 4.0], dtype=torch.float32)
        
        loss = loss_mod(pred, tgt)
        loss.backward()
        
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()
        assert not torch.isinf(pred.grad).any()
        assert (pred.grad != 0).any()

    def test_loss_mad_scale_adaptation(self):
        """Beta should adapt smoothly based on robust scale updates."""
        loss_mod = AdaptiveLoss(config=AdaptiveLossConfig(initial_beta=0.5, min_beta=0.05, max_beta=2.0))
        
        # Feed high variance errors
        high_errors = np.random.normal(0, 3.0, size=100)
        for _ in range(10):
            loss_mod.update_stats(high_errors)
        beta_high = loss_mod.beta
        
        # Feed low variance errors
        low_errors = np.random.normal(0, 0.05, size=100)
        for _ in range(30):
            loss_mod.update_stats(low_errors)
        beta_low = loss_mod.beta
        
        assert beta_high > beta_low
        metrics = loss_mod.get_metrics()
        assert metrics["total_updates"] == 40
        assert "current_beta" in metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Exploration Policy Unit Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestExplorationPolicy:
    def test_sticky_action_persistence(self):
        """Policy should persist action across consecutive steps when exploring."""
        policy = ExplorationPolicy(
            state_dim=4,
            action_dim=4,
            config=ExplorationConfig(sticky_min=3, sticky_max=5, q_margin_threshold=100.0),
            seed=42,
        )
        
        q_uniform = np.zeros(4)
        s1 = np.array([0.1, 0.2, 0.3, 0.4])
        a1 = policy.select_action(q_uniform, s1, step=0, epsilon=1.0)
        
        # In the next step, since sticky_counter > 0 and state differs, action should be identical
        s2 = np.array([0.15, 0.25, 0.35, 0.45])
        a2 = policy.select_action(q_uniform, s2, step=1, epsilon=1.0)
        assert a2 == a1

    def test_collision_sensing_breaks_sticky(self):
        """Hitting an obstacle (state invariance) must cancel sticky momentum and apply wall penalty."""
        policy = ExplorationPolicy(
            state_dim=2,
            action_dim=4,
            config=ExplorationConfig(sticky_min=5, sticky_max=5, wall_penalty=-2.0, q_margin_threshold=100.0),
            seed=42,
        )
        
        q = np.zeros(4)
        s_pos = np.array([1.0, 1.0])
        action = policy.select_action(q, s_pos, step=0, epsilon=1.0)
        
        # Agent repeats exact same state (collision with wall)
        next_action = policy.select_action(q, s_pos, step=1, epsilon=0.0)
        # Because of wall penalty on `action`, momentum is broken and a different direction is selected
        assert next_action != action

    def test_confident_q_margin_gate(self):
        """Clear superiority in Q-values should trigger instant deterministic exploitation."""
        policy = ExplorationPolicy(
            state_dim=3,
            action_dim=4,
            config=ExplorationConfig(q_margin_threshold=0.20),
            seed=123,
        )
        
        # Action 2 has clear margin: 1.0 vs 0.1 (margin = 0.90 >> 0.20)
        q_clear = np.array([0.0, 0.1, 1.0, 0.05])
        state = np.array([0.5, 0.5, 0.5])
        
        # Even with high epsilon, confident gate should exploit
        chosen_actions = [policy.select_action(q_clear, state, step=i, epsilon=0.9) for i in range(15)]
        assert chosen_actions.count(2) >= 14


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DQNAgent Full Trilogy Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDQNAgentTrilogy:
    @pytest.fixture
    def trilogy_agent(self):
        """Instantiate DQNAgent with all three champion innovations enabled."""
        cfg = DQNConfig(
            state_size=4,
            action_size=3,
            batch_size=8,
            memory_capacity=100,
            train_interval=1,
            curiosity=CuriosityConfig(enabled=True, intrinsic_reward_scale=0.1),
            adaptive_loss=AdaptiveLossConfig(enabled=True, initial_beta=0.5, tau=0.598),
            exploration=ExplorationConfig(enabled=True, sticky_min=2, sticky_max=4),
        )
        agent = DQNAgent(config=cfg)
        agent.init()
        return agent

    def test_all_modules_instantiated(self, trilogy_agent: DQNAgent):
        assert trilogy_agent.curiosity is not None
        assert trilogy_agent.adaptive_loss is not None
        assert trilogy_agent.exploration is not None
        assert isinstance(trilogy_agent.criterion, AdaptiveLoss)

    def test_trilogy_rollout_and_training(self, trilogy_agent: DQNAgent):
        """Execute action selection, experience ingestion, curiosity computation, and batch updates."""
        rng = np.random.default_rng(42)
        state = rng.uniform(-1.0, 1.0, size=4).tolist()
        
        for step in range(25):
            action = trilogy_agent.act(state)
            assert 0 <= action < 3
            
            next_state = (np.array(state) + rng.normal(0, 0.1, size=4)).tolist()
            reward = 1.0 if step % 5 == 0 else 0.0
            done = step == 24
            
            trilogy_agent.remember(state, action, reward, next_state, done)
            trilogy_agent.train_batch()
            state = next_state

        assert trilogy_agent.step_counter == 25
        assert trilogy_agent.train_step_count > 0
        assert trilogy_agent.training_loss > 0.0

        # Check sub-module metrics
        c_metrics = trilogy_agent.get_curiosity_metrics()
        assert c_metrics is not None
        assert c_metrics["total_transition_steps"] >= 25

        l_metrics = trilogy_agent.get_loss_metrics()
        assert l_metrics is not None
        assert l_metrics["total_updates"] > 0
        assert l_metrics["tau_optimism"] == pytest.approx(0.598, abs=1e-3)

        e_metrics = trilogy_agent.get_exploration_metrics()
        assert e_metrics is not None
        assert e_metrics["total_exploration_steps"] >= 25


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DQNService and FastAPI REST API Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrilogyServiceAndAPI:
    def test_service_trilogy_lifecycle(self):
        """Test DQNService end-to-end with curiosity, adaptive loss, and exploration enabled."""
        service = DQNService()
        session_id = "trilogy_service_test"
        
        cfg = DQNConfig(
            state_size=4,
            action_size=2,
            batch_size=4,
            memory_capacity=50,
            train_interval=1,
            curiosity=CuriosityConfig(enabled=True),
            adaptive_loss=AdaptiveLossConfig(enabled=True),
            exploration=ExplorationConfig(enabled=True),
        )
        
        init_res = service.init_agent(DQNInitRequest(session_id=session_id, config=cfg))
        assert init_res["curiosity_enabled"] is True
        assert init_res["adaptive_loss_enabled"] is True
        assert init_res["exploration_enabled"] is True
        
        # Step transitions
        for i in range(10):
            step_res = service.step(
                DQNStepRequest(
                    session_id=session_id,
                    state=[0.1 * i, 0.2, 0.3, 0.4],
                    action=i % 2,
                    reward=0.5,
                    next_state=[0.1 * (i + 1), 0.2, 0.3, 0.4],
                    done=False,
                    auto_train=True,
                )
            )
            assert step_res.effective_reward >= 0.5

        # Query metrics
        metrics = service.get_metrics(session_id=session_id)
        assert metrics.curiosity_enabled is True
        assert metrics.curiosity_metrics is not None
        assert metrics.adaptive_loss_enabled is True
        assert metrics.loss_metrics is not None
        assert metrics.exploration_enabled is True
        assert metrics.exploration_metrics is not None

    def test_fastapi_trilogy_endpoints(self):
        """Test FastAPI router with complete trilogy configuration."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        session = "api_trilogy_session"
        
        # 1. Initialize session
        init_payload = {
            "session_id": session,
            "config": {
                "state_size": 4,
                "action_size": 3,
                "batch_size": 4,
                "curiosity": {"enabled": True},
                "adaptive_loss": {"enabled": True},
                "exploration": {"enabled": True},
            },
        }
        res_init = client.post("/api/dqn/init", json=init_payload)
        assert res_init.status_code == 200
        data_init = res_init.json()
        assert data_init["curiosity_enabled"] is True
        assert data_init["adaptive_loss_enabled"] is True
        assert data_init["exploration_enabled"] is True
        
        # 2. Query Action
        res_act = client.post("/api/dqn/act", json={"session_id": session, "state": [0.5, -0.2, 0.1, 0.9]})
        assert res_act.status_code == 200
        assert 0 <= res_act.json()["action"] < 3
        
        # 3. Step transition
        step_payload = {
            "session_id": session,
            "state": [0.5, -0.2, 0.1, 0.9],
            "action": res_act.json()["action"],
            "reward": 1.0,
            "next_state": [0.6, -0.1, 0.2, 0.8],
            "done": False,
            "auto_train": True,
        }
        res_step = client.post("/api/dqn/step", json=step_payload)
        assert res_step.status_code == 200
        assert "intrinsic_reward" in res_step.json()
        
        # 4. Metrics endpoint
        res_metrics = client.get(f"/api/dqn/metrics?session_id={session}")
        assert res_metrics.status_code == 200
        m = res_metrics.json()
        assert m["curiosity_enabled"] is True
        assert m["adaptive_loss_enabled"] is True
        assert m["exploration_enabled"] is True
        assert "loss_metrics" in m and m["loss_metrics"] is not None
        assert "exploration_metrics" in m and m["exploration_metrics"] is not None
