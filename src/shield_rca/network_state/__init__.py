"""CyberHeal AI Network State Representation Package.

Transforms raw historical observation windows X_{t-T+1:t} in R^{T x 78}
into a standardized, compact, and deterministic Network State vector S_t in R^280,
and constructs mathematically valid future-state prediction pairs (S_t, S_{t+k})
across configurable horizons k in {1, ..., K}.
"""

from .future_dataset import (
    FutureStateDatasetBuilder,
    FutureStatePairBatch,
    MultiHorizonFutureDataset,
)
from .preprocessor import NetworkStatePreprocessor
from .quality_report import generate_future_dataset_quality_report
from .state_encoder import NetworkState, NetworkStateBatch, NetworkStateEncoder
from .state_schema import (
    STATE_FEATURE_DIM,
    STATE_FEATURE_NAMES,
    STATE_SCHEMA_VERSION,
    export_state_schema_json,
    get_state_schema_metadata,
)

__all__ = [
    # State encoding & containers
    "NetworkState",
    "NetworkStateBatch",
    "NetworkStateEncoder",
    # Schema metadata
    "STATE_SCHEMA_VERSION",
    "STATE_FEATURE_DIM",
    "STATE_FEATURE_NAMES",
    "get_state_schema_metadata",
    "export_state_schema_json",
    # Future-state dataset building
    "FutureStatePairBatch",
    "MultiHorizonFutureDataset",
    "FutureStateDatasetBuilder",
    # State-level preprocessing
    "NetworkStatePreprocessor",
    # Auditing and quality reporting
    "generate_future_dataset_quality_report",
]
