"""CyberHeal AI Network Telemetry Package.

Handles raw PCAP/packet decoding, 5-tuple stateful flow aggregation,
CIC-standard statistical feature extraction (F=78), canonical schema validation,
leakage-safe preprocessing, chronological splitting, and temporal window batching.
"""

from .dataset_builder import DatasetBuilder, WindowBatch
from .feature_schema import (
    CANONICAL_FEATURE_DIM,
    CANONICAL_FEATURES,
    CIC_COLUMN_ALIASES,
    METADATA_COLUMNS,
    SCHEMA_VERSION,
    align_dataframe_to_schema,
    get_schema_metadata,
    validate_feature_vector,
)
from .flow_aggregator import FlowKey, FlowRecord, aggregate_flows
from .flow_features import extract_feature_vector, extract_flow_features
from .packet_feature_schema import (
    MISSING_TTL_SENTINEL,
    PACKET_FEATURE_DIM,
    PACKET_FEATURE_NAMES,
    PACKET_SCHEMA_VERSION,
    PacketFeatureVector,
    validate_packet_features,
)
from .packet_features import (
    PacketFeatureExtractor,
    build_enriched_features,
    extract_packet_feature_vector,
    extract_packet_features,
)
from .pcap_parser import RawPacketRecord, iter_packets, iter_packets_from_bytes
from .preprocessor import TelemetryPreprocessor
from .split import DatasetSplits, chronological_split, scenario_aware_split
from .validator import DataValidator, TelemetryValidationError
from .windows import (
    WindowDataset,
    WindowRingBuffer,
    build_timeseries_windows,
    encode_categorical_cols,
)

__all__ = [
    # PCAP parsing
    "RawPacketRecord",
    "iter_packets",
    "iter_packets_from_bytes",
    # Flow aggregation
    "FlowKey",
    "FlowRecord",
    "aggregate_flows",
    # Flow features
    "extract_flow_features",
    "extract_feature_vector",
    # Packet-level features & schema (Phase 8)
    "PACKET_FEATURE_NAMES",
    "PACKET_FEATURE_DIM",
    "PACKET_SCHEMA_VERSION",
    "MISSING_TTL_SENTINEL",
    "PacketFeatureVector",
    "validate_packet_features",
    "PacketFeatureExtractor",
    "extract_packet_features",
    "extract_packet_feature_vector",
    "build_enriched_features",
    # Canonical schema
    "CANONICAL_FEATURES",
    "CANONICAL_FEATURE_DIM",
    "SCHEMA_VERSION",
    "METADATA_COLUMNS",
    "CIC_COLUMN_ALIASES",
    "align_dataframe_to_schema",
    "validate_feature_vector",
    "get_schema_metadata",
    # Validation & Preprocessing
    "DataValidator",
    "TelemetryValidationError",
    "TelemetryPreprocessor",
    # Splitting & Dataset building
    "DatasetSplits",
    "chronological_split",
    "scenario_aware_split",
    "DatasetBuilder",
    "WindowBatch",
    # Windows & Buffers
    "WindowDataset",
    "WindowRingBuffer",
    "build_timeseries_windows",
    "encode_categorical_cols",
]
