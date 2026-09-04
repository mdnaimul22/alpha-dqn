"""
Unit & Integration Tests for DQNService and FastAPI DQN Endpoints.
Validates session management, vectorized step processing, curiosity exploration,
metrics retrieval, and weight serialization.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from main import app
from src.schema.dqn import (
    CuriosityConfig,
    DQNActRequest,
    DQNConfig,
    DQNInitRequest,
    DQNPersistenceRequest,
    DQNStepRequest,
    DQNTrainRequest,
)
from src.services.dqn import DQNService


class TestDQNService:
    """Test suite for DQNService business logic."""

    def test_service_initialization_and_act(self):
        service = DQNService()
        config = DQNConfig(
            state_size=4,
            action_size=2,
            curiosity=CuriosityConfig(enabled=True, feature_dim=16),
        )
        init_res = service.init_agent(DQNInitRequest(session_id="test_sess", config=config))
        assert init_res["status"] == "initialized"
        assert init_res["curiosity_enabled"] is True

        act_res = service.act(
            DQNActRequest(session_id="test_sess", state=[0.1, 0.2, 0.3, 0.4])
        )
        assert act_res.action in [0, 1]

    def test_service_step_with_curiosity(self):
        service = DQNService()
        config = DQNConfig(
            state_size=4,
            action_size=2,
            batch_size=4,
            train_interval=1,
            curiosity=CuriosityConfig(enabled=True, intrinsic_reward_scale=1.0),
        )
        service.init_agent(DQNInitRequest(session_id="curious_sess", config=config))

        step_res = service.step(
            DQNStepRequest(
                session_id="curious_sess",
                state=[0.0, 0.0, 0.0, 0.0],
                action=0,
                reward=1.0,
                next_state=[0.5, 0.5, 0.5, 0.5],
                done=False,
                auto_train=True,
            )
        )

        assert step_res.session_id == "curious_sess"
        # Intrinsic curiosity bonus should be positive for novel transition
        assert step_res.intrinsic_reward > 0.0
        assert step_res.effective_reward >= step_res.intrinsic_reward

    def test_service_metrics_and_reset(self):
        service = DQNService()
        config = DQNConfig(
            state_size=4,
            action_size=2,
            curiosity=CuriosityConfig(enabled=True),
        )
        service.init_agent(DQNInitRequest(session_id="metrics_sess", config=config))

        # Perform 2 steps
        service.step(
            DQNStepRequest(
                session_id="metrics_sess",
                state=[0.1, 0.2, 0.3, 0.4],
                action=1,
                reward=0.5,
                next_state=[0.2, 0.3, 0.4, 0.5],
                done=False,
            )
        )

        metrics = service.get_metrics(session_id="metrics_sess")
        assert metrics.curiosity_enabled is True
        assert metrics.curiosity_metrics is not None
        assert metrics.curiosity_metrics["unique_states_episode"] >= 1

        # Reset episode
        reset_res = service.reset_episode(session_id="metrics_sess")
        assert reset_res["status"] == "episode_reset"

        # After reset, episode unique states should be 0
        metrics_after = service.get_metrics(session_id="metrics_sess")
        assert metrics_after.curiosity_metrics["unique_states_episode"] == 0

    def test_service_persistence(self, tmp_path):
        service = DQNService()
        save_file = str(tmp_path / "weights.json")
        config = DQNConfig(state_size=4, action_size=2, disk_path=save_file)
        service.init_agent(DQNInitRequest(session_id="persist_sess", config=config))

        save_res = service.save_weights(
            DQNPersistenceRequest(session_id="persist_sess", file_path=save_file)
        )
        assert save_res.success is True

        load_res = service.load_weights(
            DQNPersistenceRequest(session_id="persist_sess", file_path=save_file)
        )
        assert load_res.success is True


@pytest.mark.asyncio
class TestFastAPIDQNEndpoints:
    """Integration test suite for FastAPI REST API endpoints."""

    async def test_api_dqn_lifecycle(self, tmp_path):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1. Health check
            health = await client.get("/health")
            assert health.status_code == 200

            # 2. Init Agent
            init_payload = {
                "session_id": "api_test",
                "config": {
                    "state_size": 4,
                    "action_size": 3,
                    "batch_size": 4,
                    "train_interval": 1,
                    "curiosity": {
                        "enabled": True,
                        "feature_dim": 16,
                        "intrinsic_reward_scale": 0.5,
                    },
                },
            }
            init_res = await client.post("/api/dqn/init", json=init_payload)
            assert init_res.status_code == 200
            assert init_res.json()["curiosity_enabled"] is True

            # 3. Act
            act_res = await client.post(
                "/api/dqn/act",
                json={"session_id": "api_test", "state": [0.1, 0.2, 0.3, 0.4]},
            )
            assert act_res.status_code == 200
            assert act_res.json()["action"] in [0, 1, 2]

            # 4. Step
            step_res = await client.post(
                "/api/dqn/step",
                json={
                    "session_id": "api_test",
                    "state": [0.0, 0.0, 0.0, 0.0],
                    "action": 1,
                    "reward": 2.0,
                    "next_state": [0.5, 0.2, 0.1, 0.9],
                    "done": False,
                    "auto_train": True,
                },
            )
            assert step_res.status_code == 200
            step_data = step_res.json()
            assert step_data["intrinsic_reward"] > 0.0
            assert step_data["effective_reward"] >= 2.0

            # 5. Metrics
            metrics_res = await client.get("/api/dqn/metrics?session_id=api_test")
            assert metrics_res.status_code == 200
            metrics_data = metrics_res.json()
            assert metrics_data["curiosity_enabled"] is True
            assert metrics_data["curiosity_metrics"]["unique_states_episode"] >= 1

            # 6. Reset episode
            reset_res = await client.post("/api/dqn/reset-episode?session_id=api_test")
            assert reset_res.status_code == 200
            assert reset_res.json()["status"] == "episode_reset"

            # 7. Persistence Save & Load
            save_path = str(tmp_path / "api_weights.json")
            save_res = await client.post(
                "/api/dqn/save",
                json={"session_id": "api_test", "file_path": save_path},
            )
            assert save_res.status_code == 200
            assert save_res.json()["success"] is True

            load_res = await client.post(
                "/api/dqn/load",
                json={"session_id": "api_test", "file_path": save_path},
            )
            assert load_res.status_code == 200
            assert load_res.json()["success"] is True
