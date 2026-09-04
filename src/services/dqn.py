"""
Reinforcement Learning DQN Service Layer.
Orchestrates DQNAgent sessions, experience intake, vectorized batch updates,
and intrinsic curiosity novelty rewards.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.config import Settings, setup_logger
from src.core.agent import DQNAgent
from src.schema.dqn import (
    DQNActRequest,
    DQNActResponse,
    DQNConfig,
    DQNInitRequest,
    DQNMetricsResponse,
    DQNPersistenceRequest,
    DQNPersistenceResponse,
    DQNStepRequest,
    DQNStepResponse,
    DQNTrainRequest,
    DQNTrainResponse,
)

logger = setup_logger(Settings.LOG_DIR / "service.log", name="alphadqn.services.dqn")


class DQNService:
    """
    Session-aware service orchestrating Double-DQN agents with integrated Curiosity Engines.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, DQNAgent] = {}
        logger.info("Initialized DQNService session registry")

    def get_agent(self, session_id: str = "default") -> DQNAgent:
        """Retrieve existing agent or instantiate default baseline."""
        if session_id not in self._agents:
            logger.info(f"Instantiating default DQNAgent for session='{session_id}'")
            agent = DQNAgent()
            agent.init()
            self._agents[session_id] = agent
        return self._agents[session_id]

    def init_agent(self, request: DQNInitRequest) -> Dict[str, Any]:
        """Initialize or reconfigure an agent session."""
        session_id = request.session_id
        config = request.config or DQNConfig()

        agent = DQNAgent(config=config)
        agent.init()
        self._agents[session_id] = agent

        curiosity_enabled = bool(config.curiosity and config.curiosity.enabled)
        adaptive_loss_enabled = bool(config.adaptive_loss and config.adaptive_loss.enabled)
        exploration_enabled = bool(config.exploration and config.exploration.enabled)
        logger.info(
            f"Configured agent session='{session_id}' (state_size={config.state_size}, "
            f"action_size={config.action_size}, curiosity={curiosity_enabled}, "
            f"adaptive_loss={adaptive_loss_enabled}, exploration={exploration_enabled})"
        )

        return {
            "session_id": session_id,
            "status": "initialized",
            "state_size": config.state_size,
            "action_size": config.action_size,
            "curiosity_enabled": curiosity_enabled,
            "adaptive_loss_enabled": adaptive_loss_enabled,
            "exploration_enabled": exploration_enabled,
        }

    def act(self, request: DQNActRequest) -> DQNActResponse:
        """Select discrete action for state observation."""
        agent = self.get_agent(request.session_id)
        action = agent.act(request.state, force_pure_neural=request.force_pure_neural)
        return DQNActResponse(session_id=request.session_id, action=action)

    def step(self, request: DQNStepRequest) -> DQNStepResponse:
        """
        Record environment transition, compute intrinsic curiosity bonus,
        and optionally run vectorized mini-batch gradient descent.
        """
        agent = self.get_agent(request.session_id)

        # 1. Intrinsic Curiosity Reward Bonus Calculation
        intrinsic_reward = 0.0
        if agent.curiosity is not None and agent.config.curiosity:
            intrinsic_reward = float(
                agent.curiosity.compute_intrinsic_reward(
                    request.state, request.action, request.next_state
                )
            )

        # Scale bonus if enabled
        curiosity_scale = agent.config.curiosity.intrinsic_reward_scale if agent.config.curiosity else 0.0
        effective_reward = float(request.reward) + curiosity_scale * intrinsic_reward

        # 2. Update exploration policy transitions if enabled
        if agent.exploration is not None:
            agent.exploration.update(
                request.state, request.action, request.reward, request.next_state, request.done
            )

        # 3. Store directly in circular replay buffer
        agent.memory.remember(
            request.state,
            request.action,
            effective_reward,
            request.next_state,
            request.done,
        )

        # 4. Automatic training step
        if request.auto_train:
            agent.train_batch()

        # 5. Handle episodic boundary
        if request.done:
            agent.reset_episode()

        return DQNStepResponse(
            session_id=request.session_id,
            intrinsic_reward=round(intrinsic_reward, 6),
            effective_reward=round(effective_reward, 6),
            training_loss=round(agent.training_loss, 6),
            epsilon=round(agent.epsilon, 4),
            step_counter=agent.step_counter,
            train_step_count=agent.train_step_count,
        )

    def train(self, request: DQNTrainRequest) -> DQNTrainResponse:
        """Explicitly trigger one gradient descent mini-batch update."""
        agent = self.get_agent(request.session_id)
        agent.train_batch()
        return DQNTrainResponse(
            session_id=request.session_id,
            training_loss=round(agent.training_loss, 6),
            epsilon=round(agent.epsilon, 4),
            train_step_count=agent.train_step_count,
        )

    def get_metrics(self, session_id: str = "default") -> DQNMetricsResponse:
        """Query real-time metrics of agent, curiosity, adaptive loss, and exploration modules."""
        agent = self.get_agent(session_id)
        curiosity_metrics = agent.get_curiosity_metrics()
        loss_metrics = agent.get_loss_metrics()
        exploration_metrics = agent.get_exploration_metrics()

        return DQNMetricsResponse(
            session_id=session_id,
            step_counter=agent.step_counter,
            train_step_count=agent.train_step_count,
            epsilon=round(agent.epsilon, 4),
            training_loss=round(agent.training_loss, 6),
            memory_size=len(agent.memory),
            memory_capacity=agent.memory.capacity,
            curiosity_enabled=agent.curiosity is not None,
            curiosity_metrics=curiosity_metrics,
            adaptive_loss_enabled=agent.adaptive_loss is not None,
            loss_metrics=loss_metrics,
            exploration_enabled=agent.exploration is not None,
            exploration_metrics=exploration_metrics,
        )

    def reset_episode(self, session_id: str = "default") -> Dict[str, Any]:
        """Reset episodic state and curiosity frontier tracker."""
        agent = self.get_agent(session_id)
        agent.reset_episode()
        logger.info(f"Reset episode for session='{session_id}'")
        return {"session_id": session_id, "status": "episode_reset"}

    def reset_agent(self, session_id: str = "default") -> Dict[str, Any]:
        """Reset agent to pristine initialization state."""
        agent = self.get_agent(session_id)
        agent.reset()
        logger.info(f"Full agent reset executed for session='{session_id}'")
        return {"session_id": session_id, "status": "agent_reset"}

    def save_weights(self, request: DQNPersistenceRequest) -> DQNPersistenceResponse:
        """Save model weights to JSON manifest."""
        agent = self.get_agent(request.session_id)
        path = request.file_path or agent.config.disk_path
        success = agent.save(path)
        message = "Model saved successfully" if success else "Failed to save model"
        return DQNPersistenceResponse(
            session_id=request.session_id,
            success=success,
            message=message,
            file_path=path,
        )

    def load_weights(self, request: DQNPersistenceRequest) -> DQNPersistenceResponse:
        """Load model weights from JSON manifest."""
        agent = self.get_agent(request.session_id)
        path = request.file_path or agent.config.disk_path
        success = agent.load(path)
        message = "Model loaded successfully" if success else "Failed to load model"
        return DQNPersistenceResponse(
            session_id=request.session_id,
            success=success,
            message=message,
            file_path=path,
        )


# Global singleton service instance
dqn_service = DQNService()
