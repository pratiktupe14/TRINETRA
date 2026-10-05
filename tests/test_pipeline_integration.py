"""End-to-end integration test for the CyberHeal telemetry pipeline.

Validates the full deterministic chain:
synthetic PCAP -> packets -> flow aggregation -> feature extraction -> temporal windowing -> metrics
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from shield_rca.evaluation.metrics import compute_multiclass_metrics
from shield_rca.models.label_maps import INT_TO_LABEL
from shield_rca.telemetry.flow_aggregator import aggregate_flows
from shield_rca.telemetry.flow_features import extract_flow_features
from shield_rca.telemetry.pcap_parser import iter_packets
from shield_rca.telemetry.windows import build_timeseries_windows
from tests.fixtures.generate_test_pcap import create_synthetic_pcap


@pytest.fixture(scope="module")
def synthetic_pcap(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("pcaps") / "test_pipeline.pcap"
    return create_synthetic_pcap(tmp)


def test_full_telemetry_pipeline_integration(synthetic_pcap: Path):
    # 1. Packet parsing
    packets = list(iter_packets(synthetic_pcap))
    assert len(packets) == 18

    # 2. Flow aggregation
    flows = list(aggregate_flows(iter(packets)))
    assert len(flows) == 4

    # 3. Statistical feature extraction
    flow_rows = []
    for flow in flows:
        feats = extract_flow_features(flow)
        feats["label_int"] = 0  # mock benign for test
        feats["scenario_id"] = "test_run"
        flow_rows.append(feats)

    df = pd.DataFrame(flow_rows)
    assert len(df) == 4
    assert "tot_fwd_pkts" in df.columns
    assert "flow_duration_us" in df.columns

    # 4. Temporal windowing
    window_dataset = build_timeseries_windows(
        df,
        window_size=2,
        stride=1,
        min_points_per_window=1,
        sequence_label_rule="majority",
    )
    assert window_dataset.X.shape[1] == 2  # T=2
    assert window_dataset.X.shape[0] >= 1

    # 5. Metrics computation verification
    metrics = compute_multiclass_metrics(
        y_true=np.array([0, 1]),
        y_pred=np.array([0, 1]),
        label_names=["BENIGN", "Bot"],
        num_classes=2,
    )
    assert metrics["macro_f1"] == 1.0
