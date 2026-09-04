"""
High-Throughput Pre-allocated Experience Replay Buffer.
Utilizes contiguous memory buffers for zero-overhead transition storage
and vectorized tensor extraction.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch

from src.helpers import ValidationError
from src.schema.dqn import Experience


class ReplayBuffer:
    """
    High-Performance Pre-allocated Experience Replay Buffer.
    Provides O(1) in-place array writes and zero-copy tensor slicing.
    """

    def __init__(self, capacity: int, state_size: int = 75) -> None:
        if capacity <= 0:
            raise ValidationError(f"ReplayBuffer capacity must be positive, got {capacity}")

        self.capacity: int = capacity
        self.state_size: int = state_size
        self._index: int = 0
        self._size: int = 0

        # Pre-allocate contiguous NumPy memory blocks
        self._states: np.ndarray = np.zeros((capacity, state_size), dtype=np.float32)
        self._actions: np.ndarray = np.zeros(capacity, dtype=np.int64)
        self._rewards: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self._next_states: np.ndarray = np.zeros((capacity, state_size), dtype=np.float32)
        self._dones: np.ndarray = np.zeros(capacity, dtype=np.bool_)

        # Optional list of critical index tracker for fast prioritized sampling
        self._critical_mask: np.ndarray = np.zeros(capacity, dtype=np.bool_)

    def remember(
        self,
        state: List[float] | np.ndarray,
        action: int,
        reward: float,
        next_state: List[float] | np.ndarray,
        done: bool,
    ) -> None:
        """In-place O(1) transition write."""
        idx = self._index
        self._states[idx] = state
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_states[idx] = next_state
        self._dones[idx] = done
        self._critical_mask[idx] = done  # Default critical mark on episode boundary

        self._index = (idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample_tensors(
        self,
        batch_size: int,
        device: torch.device,
        critical_filter: Optional[Callable[[Experience], bool]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ultra-fast vectorized sampling directly into PyTorch tensors.
        Avoids creating intermediate Python objects or list comprehensions.
        """
        if self._size < batch_size:
            raise ValidationError(
                f"Cannot sample batch of size {batch_size} from buffer with {self._size} items"
            )

        if critical_filter is not None:
            # Slower path if custom python filter is requested
            indices = self._sample_with_filter(batch_size, critical_filter)
        else:
            indices = np.random.randint(0, self._size, size=batch_size)

        s = torch.from_numpy(self._states[indices]).to(device)
        a = torch.from_numpy(self._actions[indices]).to(device).unsqueeze(1)
        r = torch.from_numpy(self._rewards[indices]).to(device)
        ns = torch.from_numpy(self._next_states[indices]).to(device)
        d = torch.from_numpy(self._dones[indices]).to(device)

        return s, a, r, ns, d

    def _sample_with_filter(
        self, batch_size: int, critical_filter: Callable[[Experience], bool]
    ) -> np.ndarray:
        # Build matching indices
        critical_indices = []
        for i in range(self._size):
            exp = Experience(
                state=self._states[i].tolist(),
                action=int(self._actions[i]),
                reward=float(self._rewards[i]),
                next_state=self._next_states[i].tolist(),
                done=bool(self._dones[i]),
            )
            if critical_filter(exp):
                critical_indices.append(i)

        target_critical_count = min(batch_size // 2, len(critical_indices))
        batch_indices = []
        if target_critical_count > 0:
            batch_indices.extend(random.choices(critical_indices, k=target_critical_count))

        remaining = batch_size - len(batch_indices)
        batch_indices.extend(np.random.randint(0, self._size, size=remaining).tolist())
        return np.array(batch_indices, dtype=np.int64)

    def sample(
        self,
        batch_size: int,
        critical_filter: Optional[Callable[[Experience], bool]] = None,
    ) -> List[Experience]:
        """Backward-compatible object sampling interface."""
        if self._size < batch_size:
            raise ValidationError(
                f"Cannot sample batch of size {batch_size} from buffer with {self._size} items"
            )

        if critical_filter is not None:
            indices = self._sample_with_filter(batch_size, critical_filter)
        else:
            indices = np.random.randint(0, self._size, size=batch_size)

        experiences: List[Experience] = []
        for i in indices:
            experiences.append(
                Experience(
                    state=self._states[i].tolist(),
                    action=int(self._actions[i]),
                    reward=float(self._rewards[i]),
                    next_state=self._next_states[i].tolist(),
                    done=bool(self._dones[i]),
                )
            )
        return experiences

    @property
    def memory(self) -> List[Experience]:
        """Provides backward-compatible memory property inspection."""
        exps: List[Experience] = []
        for i in range(self._size):
            exps.append(
                Experience(
                    state=self._states[i].tolist(),
                    action=int(self._actions[i]),
                    reward=float(self._rewards[i]),
                    next_state=self._next_states[i].tolist(),
                    done=bool(self._dones[i]),
                )
            )
        return exps

    def clear(self) -> None:
        """Clear replay buffer contents."""
        self._index = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size
