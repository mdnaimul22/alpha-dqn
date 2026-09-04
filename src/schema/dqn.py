"""
Pydantic data schemas and contracts for Deep Q-Network (DQN) reinforcement learning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LayerConfig(BaseModel):
    """Configuration for a single dense/hidden neural network layer."""
    units: int = Field(gt=0, description="Number of units in the layer")
    activation: str = Field(default="relu", description="Activation function (e.g., relu, tanh, sigmoid)")
    kernel_initializer: str = Field(default="heNormal", description="Weight initialization strategy")


class CuriosityConfig(BaseModel):
    """Configuration hyperparameters for the Evolved Curiosity Exploration Module."""
    enabled: bool = Field(default=False, description="Enable intrinsic curiosity reward exploration bonus")
    feature_dim: int = Field(default=32, gt=0, description="Dimension of QR-orthogonal feature embedding space")
    learning_rate: float = Field(default=0.1, gt=0.0, description="Online learning rate for predictor and IDM matrices")
    intrinsic_reward_scale: float = Field(default=0.5, ge=0.0, description="Weighting multiplier for intrinsic reward bonus")
    frontier_boost: float = Field(default=1.75, ge=1.0, description="Multiplier when discovering an outward distance record")
    outward_boost_scale: float = Field(default=0.75, ge=0.0, description="Amplitude for directional outward progress multiplier")
    global_boost_max: float = Field(default=2.0, ge=1.0, description="Multiplier for lifetime first-visit state discovery")
    episodic_decay_power: float = Field(default=1.4, gt=1.0, description="Superlinear exponent for episodic habituation decay")
    controllability_floor: float = Field(default=0.25, ge=0.0, le=1.0, description="Minimum controllability floor for noise rejection")
    running_error_init: float = Field(default=0.02, gt=0.0, description="Initial running prediction error scale")


class AdaptiveLossConfig(BaseModel):
    """Configuration hyperparameters for the Evolved Adaptive Pseudo-Huber Expectile Loss."""
    enabled: bool = Field(default=True, description="Enable evolved adaptive loss instead of basic MSE")
    initial_beta: float = Field(default=1.0, gt=0.0, description="Initial Pseudo-Huber curvature scale threshold")
    tau: float = Field(default=0.598, gt=0.5, lt=1.0, description="Asymmetric optimism factor for underestimation penalty")
    ema_alpha: float = Field(default=0.43, gt=0.0, le=1.0, description="Exponential moving average smoothing factor for beta")
    beta_scale: float = Field(default=0.82, gt=0.0, description="Multiplier for robust MAD scale in beta adaptation")
    min_beta: float = Field(default=0.007, gt=0.0, description="Lower clamping bound for beta")
    max_beta: float = Field(default=20.0, gt=0.0, description="Upper clamping bound for beta")


class ExplorationConfig(BaseModel):
    """Configuration hyperparameters for the Evolved Directed Epistemic Exploration Policy."""
    enabled: bool = Field(default=False, description="Enable evolved sticky momentum & UCB exploration policy")
    q_margin_threshold: float = Field(default=0.58, ge=0.0, description="Q-value gap threshold to trigger pure exploitation")
    sticky_min: int = Field(default=3, ge=1, description="Minimum duration of sticky action momentum")
    sticky_max: int = Field(default=6, ge=1, description="Maximum duration of sticky action momentum")
    sa_ucb_scale: float = Field(default=0.82, ge=0.0, description="Scale for state-action visitation UCB bonus")
    s_novelty_scale: float = Field(default=0.52, ge=0.0, description="Scale for state rarity exploration bonus")
    global_bal_scale: float = Field(default=0.22, ge=0.0, description="Scale for global action starvation balance")
    inertia_scale: float = Field(default=0.58, ge=0.0, description="Directional forward inertia momentum weight")
    wall_penalty: float = Field(default=-1.8, description="Negative inertia penalty upon detecting collision")
    grid_discretize_scale: float = Field(default=3.5, gt=0.0, description="Discretization factor for continuous state UCB bins")


class DQNConfig(BaseModel):
    """Configuration hyperparameters for the Double-DQN Agent."""
    state_size: int = Field(default=75, gt=0, description="Dimension of state vector input")
    action_size: int = Field(default=10, gt=0, description="Dimension of discrete action space")
    layers: List[LayerConfig] = Field(
        default_factory=lambda: [
            LayerConfig(units=128, activation="relu"),
            LayerConfig(units=128, activation="relu"),
            LayerConfig(units=64, activation="relu"),
        ],
        description="List of hidden layer configurations",
    )
    gamma: float = Field(default=0.95, ge=0.0, le=1.0, description="Discount factor for future rewards")
    epsilon: float = Field(default=1.0, ge=0.0, le=1.0, description="Initial exploration rate")
    epsilon_min: float = Field(default=0.08, ge=0.0, le=1.0, description="Minimum exploration rate floor")
    epsilon_decay: float = Field(default=0.9998, gt=0.0, le=1.0, description="Per-step decay rate for epsilon")
    learning_rate: float = Field(default=0.001, gt=0.0, description="Adam optimizer learning rate")
    batch_size: int = Field(default=32, gt=0, description="Mini-batch size for training")
    memory_capacity: int = Field(default=5000, gt=0, description="Max capacity of circular replay buffer")
    train_interval: int = Field(default=10, gt=0, description="Step frequency for mini-batch training")
    target_update_interval: int = Field(default=50, gt=0, description="Training step frequency to sync target network")
    reward_scale: float = Field(default=0.02, description="Reward scaling multiplier")
    reward_clamp: float = Field(default=2.0, gt=0.0, description="Clamping boundary for scaled reward [-clamp, +clamp]")
    q_clamp: float = Field(default=3.0, gt=0.0, description="Clamping boundary for target Q-values [-clamp, +clamp]")
    dueling: bool = Field(default=True, description="Enable Dueling DQN architecture (decoupled V and A streams)")
    enable_clamping: bool = Field(default=False, description="Enable artificial reward and Q-target clamping")
    device: Optional[str] = Field(default=None, description="Compute device override ('cuda', 'cpu', or None for auto-detection)")
    curiosity: Optional[CuriosityConfig] = Field(default=None, description="Optional intrinsic curiosity exploration module")
    adaptive_loss: Optional[AdaptiveLossConfig] = Field(default_factory=AdaptiveLossConfig, description="Evolved adaptive loss configuration")
    exploration: Optional[ExplorationConfig] = Field(default=None, description="Optional evolved directed exploration policy")
    auto_save_interval: int = Field(default=200, ge=0, description="Training step frequency to auto-save weights")
    model_name: str = Field(default="DQNAgent", description="Human-readable model identifier")
    model_version: str = Field(default="1.0.0", description="Model semantic version")
    storage_key: str = Field(default="localstorage://dqn_model", description="Storage key for browser sync")
    disk_path: str = Field(default="models/dqn_weights.json", description="Relative disk path for model weights JSON")




class Experience(BaseModel):
    """Single transition tuple stored in the replay memory."""
    state: List[float] = Field(description="State vector at time t")
    action: int = Field(ge=0, description="Action index chosen at time t")
    reward: float = Field(description="Scalar reward received")
    next_state: List[float] = Field(description="State vector at time t+1")
    done: bool = Field(description="Episode termination flag")


class WeightTensorManifest(BaseModel):
    """Manifest entry for a single neural network parameter tensor."""
    name: str
    shape: List[int]
    dtype: str = "float32"
    values: List[float]


class HyperparametersManifest(BaseModel):
    gamma: float
    epsilon_min: float = Field(validation_alias="epsilonMin")
    learning_rate: float = Field(validation_alias="learningRate")


class ModelMetadata(BaseModel):
    training_loss: float = Field(default=0.0, validation_alias="trainingLoss")
    steps_trained: int = Field(default=0, validation_alias="stepsTrained")
    environment_steps: int = Field(default=0, validation_alias="environmentSteps")
    epsilon: float = Field(default=1.0)


class ModelWeightsManifest(BaseModel):
    """
    Unified cross-platform model persistence schema.
    100% compatible with ModelPersistence.js (format: dqn_dense_weights).
    """
    model_name: str = Field(default="DQNAgent", validation_alias="modelName")
    version: str = "1.0.0"
    format: str = "dqn_dense_weights"
    state_size: int = Field(validation_alias="stateSize")
    action_size: int = Field(validation_alias="actionSize")
    saved_at: Optional[str] = Field(default=None, validation_alias="savedAt")
    hyperparameters: HyperparametersManifest
    metadata: ModelMetadata
    weights: List[WeightTensorManifest]


class ActOptions(BaseModel):
    """Options passed to act() for action inference."""
    force_pure_neural: bool = Field(default=False, description="Bypass epsilon exploration and expert policy")


class DQNInitRequest(BaseModel):
    """Request payload to initialize or re-configure a DQN agent."""
    session_id: str = Field(default="default", description="Unique session/agent identifier")
    config: Optional[DQNConfig] = Field(default=None, description="Optional agent and curiosity configuration")


class DQNActRequest(BaseModel):
    """Request payload to query an action from the agent."""
    session_id: str = Field(default="default", description="Unique session/agent identifier")
    state: List[float] = Field(description="Continuous state vector observation")
    force_pure_neural: bool = Field(default=False, description="Bypass exploration/expert policies")


class DQNActResponse(BaseModel):
    """Response payload with chosen discrete action."""
    session_id: str
    action: int


class DQNStepRequest(BaseModel):
    """Request payload to record a transition into replay memory and compute intrinsic reward."""
    session_id: str = Field(default="default", description="Unique session/agent identifier")
    state: List[float] = Field(description="State vector at time t")
    action: int = Field(ge=0, description="Discrete action taken")
    reward: float = Field(description="Extrinsic scalar reward")
    next_state: List[float] = Field(description="State vector at time t+1")
    done: bool = Field(description="Episode completion flag")
    auto_train: bool = Field(default=True, description="Automatically trigger train_batch if threshold is met")


class DQNStepResponse(BaseModel):
    """Response payload after processing transition step."""
    session_id: str
    intrinsic_reward: float
    effective_reward: float
    training_loss: float
    epsilon: float
    step_counter: int
    train_step_count: int


class DQNTrainRequest(BaseModel):
    """Request payload to trigger explicit batch training."""
    session_id: str = Field(default="default", description="Unique session/agent identifier")


class DQNTrainResponse(BaseModel):
    """Response payload after batch training."""
    session_id: str
    training_loss: float
    epsilon: float
    train_step_count: int


class DQNMetricsResponse(BaseModel):
    """Detailed operational metrics of the DQN agent and curiosity engine."""
    session_id: str
    step_counter: int
    train_step_count: int
    epsilon: float
    training_loss: float
    memory_size: int
    memory_capacity: int
    curiosity_enabled: bool
    curiosity_metrics: Optional[Dict[str, Any]] = None
    adaptive_loss_enabled: bool = False
    loss_metrics: Optional[Dict[str, Any]] = None
    exploration_enabled: bool = False
    exploration_metrics: Optional[Dict[str, Any]] = None



class DQNPersistenceRequest(BaseModel):
    """Request payload to save or load weights from disk."""
    session_id: str = Field(default="default", description="Unique session/agent identifier")
    file_path: Optional[str] = Field(default=None, description="Optional custom file path")


class DQNPersistenceResponse(BaseModel):
    """Response payload for persistence actions."""
    session_id: str
    success: bool
    message: str
    file_path: str

