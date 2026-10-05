"""CyberHeal AI World Model Package.

Predicts future continuous Network State vectors S_hat_{t+k} in R^280:
    S_hat_{t+k} = W_theta(S_t, k, Delta tau)
conditioned on discrete window horizon k and continuous elapsed time Delta tau.
"""

from .baselines import LinearDynamicsBaseline, PersistenceBaseline
from .checkpoint import load_world_model_checkpoint, save_world_model_checkpoint
from .evaluator import WorldModelEvaluator
from .explainability import (
    FeatureChange,
    ForecastExplainabilityEngine,
    ForecastExplanation,
    GroupAttribution,
)
from .inference import WorldModelInference
from .candidates import GRUDynamicsWorldModel, TemporalContextResidualModel
from .loss import (
    SLICE_CONTINUOUS,
    SLICE_ENDPOINTS,
    SLICE_FLAGS,
    SLICE_MACRO,
    SLICE_SHAPE,
    WorldModelLoss,
)
from .model import ResidualBlock, StateResidualDynamicsModel
from .risk_estimator import (
    HorizonRiskEstimate,
    PredictiveRiskEstimator,
    RiskTrajectory,
    StateChangeSummary,
)
from .stage_mapper import AttackStage, AttackStageMapper, StageMappingResult
from .time_conditioning import TimeHorizonConditioner
from .trainer import WorldModelTrainer
from .trigger_gate import (
    ActionSafety,
    AssetCriticality,
    PredictiveTriggerGate,
    TriggerDecision,
    TriggerDecisionType,
    TriggerPolicy,
)

__all__ = [
    # Architecture
    "StateResidualDynamicsModel",
    "TemporalContextResidualModel",
    "GRUDynamicsWorldModel",
    "ResidualBlock",
    "TimeHorizonConditioner",
    # Loss & Slices
    "WorldModelLoss",
    "SLICE_CONTINUOUS",
    "SLICE_FLAGS",
    "SLICE_SHAPE",
    "SLICE_ENDPOINTS",
    "SLICE_MACRO",
    # Baselines
    "PersistenceBaseline",
    "LinearDynamicsBaseline",
    # Training & Evaluation
    "WorldModelTrainer",
    "WorldModelEvaluator",
    # Inference & Checkpointing
    "WorldModelInference",
    "save_world_model_checkpoint",
    "load_world_model_checkpoint",
    # Predictive Intelligence Layer (Phase 1)
    "PredictiveRiskEstimator",
    "HorizonRiskEstimate",
    "RiskTrajectory",
    "StateChangeSummary",
    "AttackStage",
    "AttackStageMapper",
    "StageMappingResult",
    "ForecastExplainabilityEngine",
    "ForecastExplanation",
    "FeatureChange",
    "GroupAttribution",
    # Predictive Trigger Gate (Phase 2)
    "PredictiveTriggerGate",
    "TriggerDecision",
    "TriggerDecisionType",
    "TriggerPolicy",
    "AssetCriticality",
    "ActionSafety",
]

