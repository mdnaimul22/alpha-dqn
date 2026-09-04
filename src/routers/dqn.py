"""
Router — Deep Q-Network (Double-DQN) Reinforcement Learning Engine.
Exposes REST endpoints for agent initialization, action inference,
transition stepping, batch training, curiosity metrics, and persistence.
"""

from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter

from src.config import Settings, setup_logger
from src.schema.dqn import (
    DQNActRequest,
    DQNActResponse,
    DQNInitRequest,
    DQNMetricsResponse,
    DQNPersistenceRequest,
    DQNPersistenceResponse,
    DQNStepRequest,
    DQNStepResponse,
    DQNTrainRequest,
    DQNTrainResponse,
)
from src.services.dqn import dqn_service

logger = setup_logger(Settings.LOG_DIR / "router.log", name="alphadqn.routers.dqn")

router = APIRouter(prefix="/api/dqn", tags=["dqn"])


@router.post("/init")
async def init_agent(body: DQNInitRequest) -> Dict[str, Any]:
    """Initialize or reconfigure a DQN agent session with custom architecture or curiosity settings."""
    logger.info(f"POST /api/dqn/init for session='{body.session_id}'")
    return dqn_service.init_agent(body)


@router.post("/act", response_model=DQNActResponse)
async def act(body: DQNActRequest) -> DQNActResponse:
    """Select action from state vector using epsilon-greedy or neural inference."""
    return dqn_service.act(body)


@router.post("/step", response_model=DQNStepResponse)
async def step(body: DQNStepRequest) -> DQNStepResponse:
    """Record environment transition, compute intrinsic curiosity bonus, and perform gradient step."""
    return dqn_service.step(body)


@router.post("/train", response_model=DQNTrainResponse)
async def train(body: DQNTrainRequest) -> DQNTrainResponse:
    """Trigger an explicit mini-batch gradient descent update."""
    return dqn_service.train(body)


@router.get("/metrics", response_model=DQNMetricsResponse)
async def get_metrics(session_id: str = "default") -> DQNMetricsResponse:
    """Query live metrics including training loss, epsilon decay, and curiosity exploration statistics."""
    return dqn_service.get_metrics(session_id=session_id)


@router.post("/reset-episode")
async def reset_episode(session_id: str = "default") -> Dict[str, Any]:
    """Reset episodic trackers, temporary visit caches, and outward frontier records."""
    return dqn_service.reset_episode(session_id=session_id)


@router.post("/reset-agent")
async def reset_agent(session_id: str = "default") -> Dict[str, Any]:
    """Reset full agent networks and replay memory to baseline pristine state."""
    return dqn_service.reset_agent(session_id=session_id)


@router.post("/save", response_model=DQNPersistenceResponse)
async def save_weights(body: DQNPersistenceRequest) -> DQNPersistenceResponse:
    """Persist agent neural network weights and metadata to JSON manifest."""
    return dqn_service.save_weights(body)


@router.post("/load", response_model=DQNPersistenceResponse)
async def load_weights(body: DQNPersistenceRequest) -> DQNPersistenceResponse:
    """Restore agent neural network weights and metadata from JSON manifest."""
    return dqn_service.load_weights(body)
