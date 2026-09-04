"""
Deep Q-Network (Double-DQN) Agent.
Game-agnostic reinforcement learning engine featuring:
  - Double-DQN target evaluation
  - Configurable deep MLP network with optional Dueling architecture (V and A streams)
  - High-throughput pre-allocated circular replay buffer
  - 100% Vectorized PyTorch Bellman updates
  - Epsilon-greedy exploration with exponential decay
  - DQfD-style expert policy blending
  - Cross-platform JSON weight persistence and PyTorch checkpoints
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from src.config import setup_logger, Settings
from src.core.buffer import ReplayBuffer
from src.core.curiosity import CuriosityEngine
from src.core.exploration import ExplorationPolicy
from src.core.loss import AdaptiveLoss
from src.core.network import QNetwork
from src.core.persistence import load_checkpoint, load_from_json, save_checkpoint, save_to_json
from src.schema.dqn import DQNConfig, Experience


logger = setup_logger(Settings.LOG_DIR / "core.log", name="alphadqn.core.agent")


class DQNAgent:
    """
    Production-grade Double-DQN Reinforcement Learning Agent.
    """

    def __init__(self, config: Optional[DQNConfig] = None, device: Optional[str] = None) -> None:
        self.config: DQNConfig = config or DQNConfig()

        # Device selection: use CUDA if available and not explicitly overridden
        chosen_device = device or self.config.device
        if chosen_device:
            self.device = torch.device(chosen_device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Models & Optimizer
        self.model: QNetwork = QNetwork(
            state_size=self.config.state_size,
            action_size=self.config.action_size,
            layers=self.config.layers,
            dueling=self.config.dueling,
        ).to(self.device)

        self.target_model: QNetwork = QNetwork(
            state_size=self.config.state_size,
            action_size=self.config.action_size,
            layers=self.config.layers,
            dueling=self.config.dueling,
        ).to(self.device)

        self.optimizer: torch.optim.Adam = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        # Adaptive Loss Formulation (Evolved Champion)
        self.adaptive_loss: Optional[AdaptiveLoss] = None
        if self.config.adaptive_loss and self.config.adaptive_loss.enabled:
            self.adaptive_loss = AdaptiveLoss(config=self.config.adaptive_loss)
            self.criterion: nn.Module = self.adaptive_loss
        else:
            self.criterion: nn.Module = nn.MSELoss()

        # Runtime State
        self.step_counter: int = 0
        self.train_step_count: int = 0
        self.training_loss: float = 0.0
        self.last_action: int = 0
        self.epsilon: float = self.config.epsilon
        self.is_training_busy: bool = False
        self.is_weights_loaded: bool = False

        # High-Throughput Pre-allocated Replay Buffer
        self.memory: ReplayBuffer = ReplayBuffer(
            capacity=self.config.memory_capacity,
            state_size=self.config.state_size,
        )

        # Pluggable Hooks
        self._expert_policy: Optional[Any] = None
        self._critical_filter: Optional[Callable[[Experience], bool]] = None

        # Intrinsic Curiosity Engine (Evolved Formulation)
        self.curiosity: Optional[CuriosityEngine] = None
        if self.config.curiosity and self.config.curiosity.enabled:
            self.curiosity = CuriosityEngine(
                state_dim=self.config.state_size,
                action_dim=self.config.action_size,
                config=self.config.curiosity,
            )

        # Novel Exploration Policy (Evolved Formulation)
        self.exploration: Optional[ExplorationPolicy] = None
        if self.config.exploration and self.config.exploration.enabled:
            self.exploration = ExplorationPolicy(
                state_dim=self.config.state_size,
                action_dim=self.config.action_size,
                config=self.config.exploration,
            )

        # Synchronize target network initially
        self.update_target_model()


    # ═══════════════════════════════════════════════════════════════
    # Lifecycle & Initialization
    # ═══════════════════════════════════════════════════════════════

    def init(self) -> bool:
        """
        Initialize agent networks, sync target network, and attempt to load existing weights.
        """
        self.update_target_model()
        # Attempt loading from disk if available
        loaded = load_from_json(self, self.config.disk_path)
        if loaded:
            logger.info(
                f"Initialized DQNAgent on device='{self.device}' with existing weights from {self.config.disk_path}"
            )
        else:
            logger.info(
                f"Initialized DQNAgent on device='{self.device}' with fresh baseline weights (CUDA={torch.cuda.is_available()})"
            )
        return True


    def reset(self) -> bool:
        """
        Full agent reset to pristine Double-DQN state.
        """
        self.step_counter = 0
        self.train_step_count = 0
        self.training_loss = 0.0
        self.epsilon = self.config.epsilon
        self.memory.clear()

        # Re-initialize models
        self.model = QNetwork(
            state_size=self.config.state_size,
            action_size=self.config.action_size,
            layers=self.config.layers,
            dueling=self.config.dueling,
        ).to(self.device)

        self.target_model = QNetwork(
            state_size=self.config.state_size,
            action_size=self.config.action_size,
            layers=self.config.layers,
            dueling=self.config.dueling,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        self.update_target_model()

        # Reset submodule states
        if self.curiosity is not None:
            self.curiosity.reset()
        if self.exploration is not None:
            self.exploration.reset()
        if self.adaptive_loss is not None:
            self.adaptive_loss.reset()

        logger.info("DQNAgent successfully reset to fresh baseline state")
        return True


    # ═══════════════════════════════════════════════════════════════
    # Plugin Hooks
    # ═══════════════════════════════════════════════════════════════

    def set_expert_policy(self, policy: Any) -> None:
        """
        Set expert policy for DQfD-style guided exploration and training.
        Expected interface:
          - get_q_values(state: list[float]) -> list[float]
          - Optional normalize(expert_q: list[float]) -> list[float]
          - Optional blend_schedule(epsilon: float) -> float
        """
        self._expert_policy = policy

    def set_critical_filter(self, filter_fn: Callable[[Experience], bool]) -> None:
        """Set critical experience filter for prioritized replay sampling."""
        self._critical_filter = filter_fn

    # ═══════════════════════════════════════════════════════════════
    # Action Selection
    # ═══════════════════════════════════════════════════════════════

    def act(self, state: List[float], force_pure_neural: bool = False) -> int:
        """
        Select action using epsilon-greedy policy with optional expert blend.
        :param state: Input state feature vector.
        :param force_pure_neural: If True, bypass exploration and expert policy.
        :return: Discrete action index [0, action_size - 1].
        """
        if not state or len(state) < self.config.state_size:
            return 0

        # 1. Pure Neural Inference (validation / deterministic testing)
        if force_pure_neural:
            self.model.eval()
            with torch.no_grad():
                tensor_s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_vals = self.model(tensor_s)
                action = int(q_vals.argmax(dim=-1).item())
                self.last_action = action
                return action

        # 2. Evolved Exploration Policy (Sticky Momentum + Epistemic UCB)
        if self.exploration is not None:
            self.model.eval()
            with torch.no_grad():
                tensor_s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_vals_np = self.model(tensor_s).squeeze(0).detach().cpu().numpy()
            action = self.exploration.select_action(
                q_values=q_vals_np,
                state=np.array(state, dtype=np.float64),
                step=self.step_counter,
                epsilon=self.epsilon,
            )
            self.last_action = action
            return action

        # 3. Standard Exploration with probability epsilon
        if random.random() < self.epsilon:

            # Guided exploration: 70% expert policy if available, 30% random
            if self._expert_policy is not None and random.random() < 0.70:
                try:
                    expert_q = self._expert_policy.get_q_values(state)
                    best_action = int(max(range(len(expert_q)), key=lambda i: expert_q[i]))
                    self.last_action = best_action
                    return best_action
                except Exception as exc:
                    logger.warning(f"Expert policy evaluation failed, falling back to random: {exc}")

            action = random.randint(0, self.config.action_size - 1)
            self.last_action = action
            return action

        # 3. Exploitation using Neural Network
        self.model.eval()
        try:
            with torch.no_grad():
                tensor_s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_vals = self.model(tensor_s)
                action = int(q_vals.argmax(dim=-1).item())
                self.last_action = action
                return action
        except Exception as exc:
            logger.error(f"Neural network prediction error, falling back: {exc}")

        # Fallback to expert if present
        if self._expert_policy is not None:
            try:
                expert_q = self._expert_policy.get_q_values(state)
                best_action = int(max(range(len(expert_q)), key=lambda i: expert_q[i]))
                self.last_action = best_action
                return best_action
            except Exception as exc:
                logger.debug(f"Expert policy fallback failed: {exc}")

        action = random.randint(0, self.config.action_size - 1)
        self.last_action = action
        return action

    # ═══════════════════════════════════════════════════════════════
    # Experience Replay
    # ═══════════════════════════════════════════════════════════════

    def remember(
        self,
        state: List[float],
        action: int,
        reward: float,
        next_state: List[float],
        done: bool,
    ) -> None:
        """Add experience transition to circular replay buffer, computing curiosity bonus if enabled."""
        effective_reward = float(reward)
        if self.curiosity is not None and self.config.curiosity:
            r_intrinsic = self.curiosity.compute_intrinsic_reward(state, action, next_state)
            effective_reward += self.config.curiosity.intrinsic_reward_scale * r_intrinsic

        if self.exploration is not None:
            self.exploration.update(state, action, reward, next_state, done)

        self.memory.remember(state, action, effective_reward, next_state, done)


    # ═══════════════════════════════════════════════════════════════
    # Training
    # ═══════════════════════════════════════════════════════════════

    def update_target_model(self) -> None:
        """Synchronize Target Model weights from Main Online Model."""
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def train_batch(self) -> None:
        """
        Execute one mini-batch gradient descent step using Double-DQN.
        100% Vectorized PyTorch execution without per-item Python loops.
        """
        self.step_counter += 1

        if (
            len(self.memory) < self.config.batch_size
            or self.is_training_busy
            or self.step_counter % self.config.train_interval != 0
        ):
            return

        self.is_training_busy = True

        try:
            # 1. Decay Epsilon
            if self.epsilon > self.config.epsilon_min:
                decay_factor = math.pow(self.config.epsilon_decay, self.config.train_interval)
                self.epsilon = max(self.config.epsilon_min, self.epsilon * decay_factor)

            # 2. Vectorized Mini-Batch Extraction directly into PyTorch tensors
            states, actions, rewards, next_states, dones = self.memory.sample_tensors(
                self.config.batch_size, self.device, critical_filter=self._critical_filter
            )

            # 3. Predict Current Q-values for Chosen Actions
            self.model.train()
            current_q = self.model(states)
            state_action_q = current_q.gather(1, actions).squeeze(1)

            # 4. Vectorized Double-DQN Target Computation
            # Online network selects best action; Target network evaluates value
            with torch.no_grad():
                online_next_q = self.model(next_states)
                best_next_actions = online_next_q.argmax(dim=1, keepdim=True)
                target_next_q = self.target_model(next_states)
                double_dqn_values = target_next_q.gather(1, best_next_actions).squeeze(1)

                reward_scale = self.config.reward_scale
                reward_clamp = self.config.reward_clamp
                q_clamp = self.config.q_clamp
                gamma = self.config.gamma

                if self.config.enable_clamping:
                    scaled_r = torch.clamp(rewards * reward_scale, -reward_clamp, reward_clamp)
                    double_targets = torch.clamp(
                        scaled_r + gamma * double_dqn_values, -q_clamp, q_clamp
                    )
                else:
                    scaled_r = rewards * reward_scale
                    double_targets = scaled_r + gamma * double_dqn_values

                td_targets = torch.where(dones, scaled_r, double_targets)

            # 5. Optional DQfD Expert Policy Blending
            has_expert = self._expert_policy is not None
            if has_expert:
                if hasattr(self._expert_policy, "blend_schedule") and callable(self._expert_policy.blend_schedule):
                    expert_weight = float(self._expert_policy.blend_schedule(self.epsilon))
                else:
                    expert_weight = float(0.40 + 0.40 * min(1.0, self.epsilon / 1.0))

                if expert_weight > 0.0:
                    td_weight = 1.0 - expert_weight
                    states_np = states.detach().cpu().numpy()
                    actions_np = actions.squeeze(1).detach().cpu().numpy()
                    expert_targets = []
                    for s_vec, a_idx in zip(states_np, actions_np):
                        eq = self._expert_policy.get_q_values(s_vec.tolist())
                        if hasattr(self._expert_policy, "normalize") and callable(self._expert_policy.normalize):
                            eq = self._expert_policy.normalize(eq)
                        expert_targets.append(eq[a_idx])
                    expert_tensor = torch.tensor(expert_targets, dtype=torch.float32, device=self.device)
                    td_targets = td_weight * td_targets + expert_weight * expert_tensor

            # 6. Compute Loss & Optimize with Gradient Clipping
            loss = self.criterion(state_action_q, td_targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update Adaptive Loss statistics
            if self.adaptive_loss is not None:
                td_errors = (state_action_q - td_targets).detach().cpu().numpy()
                self.adaptive_loss.update_stats(td_errors)

            # 7. Online update of Curiosity Engine from experienced batch
            if self.curiosity is not None:
                self.curiosity.update(
                    states.detach().cpu().numpy(),
                    actions.squeeze(1).detach().cpu().numpy(),
                    next_states.detach().cpu().numpy(),
                )

            self.training_loss = float(loss.item())
            self.train_step_count += 1

            # 8. Periodic Target Sync
            if self.train_step_count % self.config.target_update_interval == 0:
                self.update_target_model()

            # 9. Periodic Auto-save
            if self.config.auto_save_interval > 0 and self.train_step_count % self.config.auto_save_interval == 0:
                self.save()

        except Exception as exc:
            logger.error(f"Error during Double-DQN batch training: {exc}")

        finally:
            self.is_training_busy = False

    def reset_episode(self) -> None:
        """Reset episode-specific temporary counters, caches, and frontier tracking."""
        if self.curiosity is not None:
            self.curiosity.reset_episode()
        if self.exploration is not None:
            self.exploration.reset_episode()

    def get_curiosity_metrics(self) -> Optional[Dict[str, Any]]:
        """Return operational curiosity metrics if enabled, else None."""
        if self.curiosity is not None:
            return self.curiosity.get_metrics()
        return None

    def get_loss_metrics(self) -> Optional[Dict[str, Any]]:
        """Return operational adaptive loss metrics if enabled, else None."""
        if self.adaptive_loss is not None:
            return self.adaptive_loss.get_metrics()
        return None

    def get_exploration_metrics(self) -> Optional[Dict[str, Any]]:
        """Return operational exploration metrics if enabled, else None."""
        if self.exploration is not None:
            return self.exploration.get_metrics()
        return None


    # ═══════════════════════════════════════════════════════════════
    # Persistence
    # ═══════════════════════════════════════════════════════════════

    def save(self, file_path: Optional[str] = None) -> bool:
        """Save model weights and metadata to JSON manifest."""
        return save_to_json(self, file_path or self.config.disk_path)

    def load(self, file_path: Optional[str] = None) -> bool:
        """Load model weights and metadata from JSON manifest."""
        return load_from_json(self, file_path or self.config.disk_path)

    def save_checkpoint(self, file_path: str) -> bool:
        """Save native PyTorch training checkpoint."""
        return save_checkpoint(self, file_path)

    def load_checkpoint(self, file_path: str) -> bool:
        """Load native PyTorch training checkpoint."""
        return load_checkpoint(self, file_path)
