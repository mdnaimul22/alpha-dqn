"""
Unit tests for the DQN Core Engine (QNetwork, ReplayBuffer, DQNAgent, Persistence).
"""

from __future__ import annotations

import json
import os
import tempfile
import pytest
import torch

from src.config import PROJECT_ROOT, delete, exists, read_json
from src.core.agent import DQNAgent
from src.core.buffer import ReplayBuffer
from src.core.network import QNetwork
from src.schema.dqn import DQNConfig, Experience, LayerConfig


class DummyExpertPolicy:
    """Mock expert policy to test DQfD blending."""

    def __init__(self, action_size: int = 4) -> None:
        self.action_size = action_size

    def get_q_values(self, state: list[float]) -> list[float]:
        # Favor action 1
        q = [0.1] * self.action_size
        q[1] = 5.0
        return q

    def normalize(self, expert_q: list[float]) -> list[float]:
        max_v = max(expert_q)
        return [v / max_v for v in expert_q]

    def blend_schedule(self, epsilon: float) -> float:
        return 0.5


def test_qnetwork_architecture():
    """Verify QNetwork shapes and layers."""
    layers = [
        LayerConfig(units=16, activation="relu"),
        LayerConfig(units=8, activation="relu"),
    ]
    net = QNetwork(state_size=4, action_size=2, layers=layers)

    # 1D single state inference
    s_single = torch.randn(4)
    q_single = net(s_single)
    assert q_single.shape == (1, 2)

    # 2D batch state inference
    s_batch = torch.randn(5, 4)
    q_batch = net(s_batch)
    assert q_batch.shape == (5, 2)


def test_replay_buffer_circular():
    """Verify ReplayBuffer circular capacity and sampling."""
    buf = ReplayBuffer(capacity=3)
    assert len(buf) == 0

    buf.remember([1.0], 0, 1.0, [2.0], False)
    buf.remember([2.0], 1, 2.0, [3.0], False)
    buf.remember([3.0], 0, 3.0, [4.0], True)
    assert len(buf) == 3

    # Adding 4th item should overwrite index 0
    buf.remember([4.0], 1, 4.0, [5.0], False)
    assert len(buf) == 3
    # Check that item with state [1.0] was overwritten
    states = [e.state[0] for e in buf.memory]
    assert 1.0 not in states
    assert 4.0 in states

    # Test sampling
    batch = buf.sample(batch_size=2)
    assert len(batch) == 2


def test_replay_buffer_critical_filter():
    """Verify critical filter prioritized sampling."""
    buf = ReplayBuffer(capacity=10)
    for i in range(10):
        # Even i are marked critical
        buf.remember([float(i)], 0, float(i), [float(i + 1)], i % 2 == 0)

    critical_filter = lambda e: e.done is True
    batch = buf.sample(batch_size=4, critical_filter=critical_filter)
    assert len(batch) == 4
    critical_count = sum(1 for e in batch if e.done is True)
    # Target critical count is min(4//2, critical_pool=5) = 2
    assert critical_count >= 2


def test_dqn_agent_lifecycle():
    """Verify DQNAgent initialization, action selection, and training step."""
    cfg = DQNConfig(
        state_size=4,
        action_size=3,
        layers=[LayerConfig(units=16, activation="relu")],
        batch_size=4,
        memory_capacity=20,
        train_interval=1,
        learning_rate=0.01,
    )
    agent = DQNAgent(config=cfg, device="cpu")
    agent.init()

    # 1. Action selection with force_pure_neural
    s = [0.5, -0.2, 0.1, 1.0]
    action_pure = agent.act(s, force_pure_neural=True)
    assert 0 <= action_pure < 3

    # 2. Add experiences to replay memory
    for i in range(10):
        agent.remember([0.1 * i, 0.2, -0.1, 0.5], i % 3, 1.0 if i % 2 == 0 else -1.0, [0.2, 0.1, 0.0, 0.1], False)

    # 3. Train batch
    initial_steps = agent.train_step_count
    agent.train_batch()
    assert agent.train_step_count == initial_steps + 1
    assert isinstance(agent.training_loss, float)


def test_dqn_expert_policy_blend():
    """Verify DQfD expert policy hooks in agent act and train."""
    cfg = DQNConfig(
        state_size=4,
        action_size=4,
        layers=[LayerConfig(units=16, activation="relu")],
        batch_size=4,
        memory_capacity=10,
        train_interval=1,
        epsilon=1.0,  # 100% exploration
    )
    agent = DQNAgent(config=cfg, device="cpu")
    expert = DummyExpertPolicy(action_size=4)
    agent.set_expert_policy(expert)

    # In exploratory mode, 70% of calls follow expert (action=1)
    actions = [agent.act([0.1, 0.2, 0.3, 0.4]) for _ in range(50)]
    assert actions.count(1) > 20  # Statistically guaranteed with 70% bias

    # Add memory and train with expert blend
    for i in range(6):
        agent.remember([0.1, 0.2, 0.3, 0.4], 1, 1.0, [0.2, 0.3, 0.4, 0.5], False)

    agent.train_batch()
    assert agent.train_step_count >= 1


def test_json_persistence_compatibility():
    """Verify saving to and loading from JS-compatible JSON manifest."""
    test_json_path = "models/test_weights.json"

    try:
        # 1. Test Standard MLP Mode (exact 1:1 JS format)
        cfg_standard = DQNConfig(
            state_size=4,
            action_size=2,
            layers=[LayerConfig(units=8, activation="relu")],
            dueling=False,
            disk_path=test_json_path,
        )
        agent_std = DQNAgent(config=cfg_standard, device="cpu")
        agent_std.step_counter = 150
        agent_std.train_step_count = 42
        agent_std.training_loss = 0.0125
        agent_std.epsilon = 0.55

        ok = agent_std.save(test_json_path)
        assert ok is True
        assert exists(test_json_path)

        raw_data = read_json(test_json_path)
        assert raw_data["format"] == "dqn_dense_weights"
        assert raw_data["stateSize"] == 4
        assert raw_data["actionSize"] == 2
        assert raw_data["metadata"]["stepsTrained"] == 42
        assert raw_data["metadata"]["environmentSteps"] == 150
        assert raw_data["metadata"]["epsilon"] == 0.55
        assert len(raw_data["weights"]) == 4  # 2 layers * (kernel + bias)

        # Load back into fresh standard agent
        fresh_std = DQNAgent(config=cfg_standard, device="cpu")
        assert fresh_std.load(test_json_path) is True
        assert fresh_std.step_counter == 150
        test_s = [0.5, 0.1, -0.3, 0.8]
        assert agent_std.act(test_s, force_pure_neural=True) == fresh_std.act(test_s, force_pure_neural=True)

        # 2. Test Dueling Architecture Mode
        cfg_dueling = DQNConfig(
            state_size=4,
            action_size=2,
            layers=[LayerConfig(units=8, activation="relu")],
            dueling=True,
            disk_path=test_json_path,
        )
        agent_duel = DQNAgent(config=cfg_dueling, device="cpu")
        assert agent_duel.save(test_json_path) is True
        raw_duel = read_json(test_json_path)
        assert len(raw_duel["weights"]) == 6  # dense_0 + value_stream + advantage_stream

        fresh_duel = DQNAgent(config=cfg_dueling, device="cpu")
        assert fresh_duel.load(test_json_path) is True
        assert agent_duel.act(test_s, force_pure_neural=True) == fresh_duel.act(test_s, force_pure_neural=True)

    finally:
        if exists(test_json_path):
            delete(test_json_path)


def test_pytorch_checkpoint_persistence():
    """Verify native PyTorch checkpoint save and restore."""
    ckpt_path = "models/test_checkpoint.pt"

    try:
        cfg = DQNConfig(
            state_size=4,
            action_size=2,
            layers=[LayerConfig(units=8, activation="relu")],
        )
        agent = DQNAgent(config=cfg, device="cpu")
        agent.train_step_count = 88
        agent.epsilon = 0.33

        assert agent.save_checkpoint(ckpt_path) is True
        assert exists(ckpt_path)

        fresh_agent = DQNAgent(config=cfg, device="cpu")
        assert fresh_agent.load_checkpoint(ckpt_path) is True
        assert fresh_agent.train_step_count == 88
        assert fresh_agent.epsilon == 0.33

    finally:
        if exists(ckpt_path):
            delete(ckpt_path)
