"""
DQN Model Persistence Layer.
Handles serialization and deserialization of DQN weights and training state:
  - JSON format: 100% interoperable with ModelPersistence.js (dqn_dense_weights)
  - PyTorch Checkpoint format: Native .pt state dicts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import torch

from src.config import ensure_dir, exists, get_abs_path, read_json, setup_logger, write_json, Settings
from src.helpers import time_now_iso
from src.schema.dqn import (
    HyperparametersManifest,
    ModelMetadata,
    ModelWeightsManifest,
    WeightTensorManifest,
)

if TYPE_CHECKING:
    from src.core.agent import DQNAgent
    from src.core.network import QNetwork

logger = setup_logger(Settings.LOG_DIR / "core.log", name="alphadqn.core.persistence")


def _get_linear_layers(model: QNetwork) -> List[Tuple[str, torch.nn.Linear]]:
    """Retrieve named linear layers preserving architecture semantics."""
    named_layers: List[Tuple[str, torch.nn.Linear]] = []
    dense_idx = 0

    for module in model.feature_extractor:
        if isinstance(module, torch.nn.Linear):
            named_layers.append((f"dense_{dense_idx}", module))
            dense_idx += 1

    if getattr(model, "dueling", False):
        named_layers.append(("value_stream", model.value_stream))
        named_layers.append(("advantage_stream", model.advantage_stream))
    else:
        named_layers.append(("output", model.output_layer))

    return named_layers


def save_to_json(agent: DQNAgent, file_path: Optional[str] = None) -> bool:
    """
    Export neural network weights and training metadata to disk as a JSON manifest.
    Strictly follows the 'dqn_dense_weights' format for cross-runtime interoperability.
    """
    target_path = file_path or agent.config.disk_path
    try:
        named_linear_layers = _get_linear_layers(agent.model)
        weights_data: List[Dict[str, Any]] = []

        for layer_name, layer in named_linear_layers:
            # 1. Kernel weight: TF.js expects [in_features, out_features]
            # PyTorch holds [out_features, in_features]
            weight_tensor = layer.weight.detach().cpu()
            kernel = weight_tensor.t().contiguous()
            weights_data.append({
                "name": f"{layer_name}/kernel",
                "shape": list(kernel.shape),
                "dtype": "float32",
                "values": [round(float(v), 7) for v in kernel.flatten().tolist()],
            })

            # 2. Bias: shape [out_features]
            if layer.bias is not None:
                bias_tensor = layer.bias.detach().cpu().flatten()
                weights_data.append({
                    "name": f"{layer_name}/bias",
                    "shape": list(bias_tensor.shape),
                    "dtype": "float32",
                    "values": [round(float(v), 7) for v in bias_tensor.tolist()],
                })

        manifest = {
            "modelName": agent.config.model_name,
            "version": agent.config.model_version,
            "format": "dqn_dense_weights",
            "stateSize": agent.config.state_size,
            "actionSize": agent.config.action_size,
            "dueling": getattr(agent.config, "dueling", False),
            "savedAt": time_now_iso(),
            "hyperparameters": {
                "gamma": agent.config.gamma,
                "epsilonMin": agent.config.epsilon_min,
                "learningRate": agent.config.learning_rate,
            },
            "metadata": {
                "trainingLoss": float(agent.training_loss),
                "stepsTrained": int(agent.train_step_count),
                "environmentSteps": int(agent.step_counter),
                "epsilon": float(agent.epsilon),
            },
            "weights": weights_data,
        }

        # Derive parent directory and ensure it exists
        parent_dir = target_path.rsplit("/", 1)[0] if "/" in target_path else ""
        if parent_dir:
            ensure_dir(parent_dir)

        write_json(target_path, manifest)
        logger.info(f"Model successfully saved to JSON manifest at {target_path}")
        return True

    except Exception as exc:
        logger.error(f"Failed to save model weights to {target_path}: {exc}")
        return False


def load_from_json(agent: DQNAgent, file_path: Optional[str] = None) -> bool:
    """
    Import neural network weights and metadata from JSON manifest.
    Validates state size compatibility and converts TF.js weight tensors to PyTorch format.
    """
    target_path = file_path or agent.config.disk_path
    if not exists(target_path):
        logger.warning(f"Model JSON manifest does not exist at {target_path}")
        return False

    try:
        data = read_json(target_path)
        if not data or not isinstance(data, dict):
            logger.warning(f"Invalid JSON content in model manifest at {target_path}")
            return False

        # State size validation
        state_size = data.get("stateSize") or data.get("state_size")
        if state_size and state_size != agent.config.state_size:
            logger.warning(
                f"Incompatible state size: manifest has {state_size}, agent expects {agent.config.state_size}. Skipping."
            )
            return False

        # Restore metadata
        metadata = data.get("metadata", {})
        if "stepsTrained" in metadata:
            agent.train_step_count = int(metadata["stepsTrained"])
        if "environmentSteps" in metadata:
            agent.step_counter = int(metadata["environmentSteps"])
        if "trainingLoss" in metadata:
            agent.training_loss = float(metadata["trainingLoss"])
        if "epsilon" in metadata:
            agent.epsilon = float(metadata["epsilon"])

        # Restore weights
        weights = data.get("weights", [])
        if weights and agent.model:
            named_linear_layers = _get_linear_layers(agent.model)
            weight_map = {w["name"]: w for w in weights if "name" in w}

            for layer_name, layer in named_linear_layers:
                kernel_entry = weight_map.get(f"{layer_name}/kernel")
                bias_entry = weight_map.get(f"{layer_name}/bias")

                # Fallback mapping: if loading standard weights into standard model or legacy output
                if not kernel_entry and layer_name == "output":
                    kernel_entry = weight_map.get("dense_3/kernel") or weight_map.get("dense_2/kernel")

                if kernel_entry:
                    # TF.js shape: [in_features, out_features] -> PyTorch: [out_features, in_features]
                    k_shape = kernel_entry["shape"]
                    k_vals = kernel_entry["values"]
                    k_tensor = torch.tensor(k_vals, dtype=torch.float32).reshape(k_shape).t().contiguous()
                    if k_tensor.shape == layer.weight.shape:
                        layer.weight.data.copy_(k_tensor.to(layer.weight.device))

                if bias_entry and layer.bias is not None:
                    b_vals = bias_entry["values"]
                    b_tensor = torch.tensor(b_vals, dtype=torch.float32).reshape(bias_entry["shape"])
                    if b_tensor.shape == layer.bias.shape:
                        layer.bias.data.copy_(b_tensor.to(layer.bias.device))

            agent.update_target_model()
            agent.is_weights_loaded = True

        logger.info(f"Loaded unified model from {target_path} (Steps={agent.step_counter}, Loss={agent.training_loss:.4f}, Epsilon={agent.epsilon:.4f})")
        return True

    except Exception as exc:
        logger.error(f"Failed to load model weights from {target_path}: {exc}")
        return False


def save_checkpoint(agent: DQNAgent, file_path: str) -> bool:
    """Save native PyTorch state dict and optimizer checkpoint."""
    try:
        parent_dir = file_path.rsplit("/", 1)[0] if "/" in file_path else ""
        if parent_dir:
            ensure_dir(parent_dir)

        abs_path = get_abs_path(file_path)
        checkpoint = {
            "model_state_dict": agent.model.state_dict(),
            "target_model_state_dict": agent.target_model.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "step_counter": agent.step_counter,
            "train_step_count": agent.train_step_count,
            "training_loss": agent.training_loss,
            "epsilon": agent.epsilon,
            "config": agent.config.model_dump(),
        }
        torch.save(checkpoint, abs_path)
        logger.info(f"PyTorch native checkpoint saved to {file_path}")
        return True
    except Exception as exc:
        logger.error(f"Failed to save checkpoint to {file_path}: {exc}")
        return False


def load_checkpoint(agent: DQNAgent, file_path: str) -> bool:
    """Load native PyTorch state dict and resume agent training state."""
    if not exists(file_path):
        logger.warning(f"Checkpoint file not found: {file_path}")
        return False

    try:
        abs_path = get_abs_path(file_path)
        checkpoint = torch.load(abs_path, map_location=agent.device, weights_only=False)
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        agent.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.step_counter = checkpoint.get("step_counter", agent.step_counter)
        agent.train_step_count = checkpoint.get("train_step_count", agent.train_step_count)
        agent.training_loss = checkpoint.get("training_loss", agent.training_loss)
        agent.epsilon = checkpoint.get("epsilon", agent.epsilon)
        agent.is_weights_loaded = True
        logger.info(f"PyTorch checkpoint restored from {file_path}")
        return True
    except Exception as exc:
        logger.error(f"Failed to load checkpoint from {file_path}: {exc}")
        return False
