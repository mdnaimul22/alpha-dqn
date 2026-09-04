"""
Configurable PyTorch Neural Network for Deep Q-Network (QNetwork).
Supports standard MLP and Dueling Q-Network architecture (decoupled V and A streams).
"""

from __future__ import annotations

from typing import List
import torch
import torch.nn as nn

from src.schema.dqn import LayerConfig


class QNetwork(nn.Module):
    """
    Configurable Deep Q-Network.
    Builds dense sequential MLP layers with He/Kaiming normal initialization.
    Supports Dueling architecture: Q(s, a) = V(s) + (A(s, a) - mean(A(s, :))).
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        layers: List[LayerConfig],
        dueling: bool = False,
    ) -> None:
        super().__init__()
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.dueling: bool = dueling

        hidden_layers: List[nn.Module] = []
        in_dim = state_size

        for layer_cfg in layers:
            out_dim = layer_cfg.units
            linear = nn.Linear(in_dim, out_dim)

            # He / Kaiming normal initialization matching TF.js heNormal
            act_name = layer_cfg.activation.lower()
            nonlinearity = "relu" if act_name in ("relu", "leaky_relu") else "linear"
            nn.init.kaiming_normal_(linear.weight, nonlinearity=nonlinearity)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

            hidden_layers.append(linear)
            hidden_layers.append(self._get_activation(act_name))
            in_dim = out_dim

        self.feature_extractor = nn.Sequential(*hidden_layers)

        if self.dueling:
            # Value stream: V(s) -> scalar state value
            self.value_stream = nn.Linear(in_dim, 1)
            nn.init.kaiming_normal_(self.value_stream.weight, nonlinearity="linear")
            if self.value_stream.bias is not None:
                nn.init.zeros_(self.value_stream.bias)

            # Advantage stream: A(s, a) -> action-specific advantages
            self.advantage_stream = nn.Linear(in_dim, action_size)
            nn.init.kaiming_normal_(self.advantage_stream.weight, nonlinearity="linear")
            if self.advantage_stream.bias is not None:
                nn.init.zeros_(self.advantage_stream.bias)
        else:
            # Standard single output stream
            self.output_layer = nn.Linear(in_dim, action_size)
            nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity="linear")
            if self.output_layer.bias is not None:
                nn.init.zeros_(self.output_layer.bias)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "tanh":
            return nn.Tanh()
        if name == "sigmoid":
            return nn.Sigmoid()
        if name == "elu":
            return nn.ELU()
        if name == "leaky_relu":
            return nn.LeakyReLU()
        return nn.ReLU()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param state: Tensor of shape (batch_size, state_size) or (state_size,)
        :return: Q-values tensor of shape (batch_size, action_size)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.feature_extractor(state)

        if self.dueling:
            values = self.value_stream(features)  # (B, 1)
            advantages = self.advantage_stream(features)  # (B, action_size)
            # Q(s, a) = V(s) + (A(s, a) - mean_a(A(s, a)))
            return values + (advantages - advantages.mean(dim=-1, keepdim=True))

        return self.output_layer(features)
