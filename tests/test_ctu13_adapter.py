"""Focused tests for the CTU-13 Dataset Adapter and TRINETRA Pipeline Compatibility."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from shield_rca.evaluation.ctu13_adapter import (
    CTU13_OFFICIAL_SCENARIOS,
    Ctu13Adapter,
    Ctu13ScenarioSplitter,
    parse_ctu_port,
    parse_ctu_protocol,
)
from shield_rca.network_state.future_dataset import FutureStateDatasetBuilder
from shield_rca.network_state.state_encoder import NetworkStateEncoder
from shield_rca.network_state.state_schema import STATE_FEATURE_DIM
from shield_rca.telemetry.feature_schema import CANONICAL_FEATURE_DIM, CANONICAL_FEATURES


@pytest.fixture
def synthetic_binetflow_file(tmp_path: Path) -> Path:
    """Create a synthetic CTU-13 .binetflow file with diverse flows and labels."""
    file_path = tmp_path / "capture20110810.binetflow"
    content = (
        "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label\n"
        "2011/08/10 09:45:01.000000,1.0234,tcp,147.32.84.165,1025,->,147.32.80.9,80,S_RA,0,0,10,1500,500,flow=From-Botnet-V42-TCP-Attempt-SPAM\n"
        "2011/08/10 09:45:02.500000,0.5000,udp,147.32.84.165,domain,<->,147.32.80.9,53,CON,0,0,2,240,120,flow=From-Normal-V42-Grill\n"
        "2011/08/10 09:45:03.000000,0.0100,icmp,147.32.84.165,0,->,147.32.80.9,0,INT,0,0,1,64,64,flow=Background-Established-cmp\n"
        "2011/08/10 09:45:04.000000,2.5000,tcp,147.32.84.165,0x0050,->,147.32.80.9,443,FSRPA_FSRPA,0,0,20,3200,1600,flow=From-Botnet-V42-CC\n"
        "2011/08/10 09:45:05.000000,0.2000,tcp,147.32.84.165,http,->,147.32.80.9,8080,RST,0,0,4,400,200,flow=From-Normal-Grill\n"
        "2011/08/10 09:45:06.000000,0.0500,udp,147.32.84.165,123,->,147.32.80.9,123,INT,0,0,2,100,50,flow=Background-UDP-Attempt\n"
        "2011/08/10 09:45:07.000000,1.2000,tcp,147.32.84.165,1026,->,147.32.80.9,22,S_RA,0,0,8,800,400,flow=From-Botnet-PortScan\n"
        "2011/08/10 09:45:08.000000,0.8000,tcp,147.32.84.165,1027,->,147.32.80.9,25,S_A,0,0,6,600,300,flow=From-Botnet-SPAM\n"
        "2011/08/10 09:45:09.000000,0.1000,tcp,147.32.84.165,443,->,147.32.80.9,443,CON,0,0,4,400,200,flow=From-Normal-SSL\n"
        "2011/08/10 09:45:10.000000,0.0500,udp,147.32.84.165,53,->,147.32.80.9,53,INT,0,0,2,150,75,flow=Background-DNS\n"
    )
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_file_discovery_and_scenario_detection() -> None:
    """Verify official CTU-13 scenarios and scenario detection logic."""
    assert len(CTU13_OFFICIAL_SCENARIOS) == 13
    assert CTU13_OFFICIAL_SCENARIOS["scenario_1"]["botnet"] == "Neris"
    assert CTU13_OFFICIAL_SCENARIOS["scenario_10"]["botnet"] == "Rbot"

    # Test detection from filenames
    assert Ctu13Adapter.detect_scenario_id("data/ctu13/capture20110810.binetflow") == "scenario_1"
    assert Ctu13Adapter.detect_scenario_id("capture20110815-2.binetflow") == "scenario_5"
    assert Ctu13Adapter.detect_scenario_id("path/to/scenario_10/flows.csv") == "scenario_10"
    assert Ctu13Adapter.detect_scenario_id("capture20110818.binetflow") == "scenario_10"


def test_port_and_protocol_parsing() -> None:
    """Verify parsing of service names, hexadecimal, and numeric ports/protocols."""
    assert parse_ctu_port("http") == 80
    assert parse_ctu_port("https") == 443
    assert parse_ctu_port("domain") == 53
    assert parse_ctu_port("0x0050") == 80
    assert parse_ctu_port(8080) == 8080
    assert parse_ctu_port("invalid") == 0
    assert parse_ctu_port(None) == 0

    assert parse_ctu_protocol("tcp") == 6
    assert parse_ctu_protocol("udp") == 17
    assert parse_ctu_protocol("icmp") == 1
    assert parse_ctu_protocol(17) == 17
    assert parse_ctu_protocol(None) == 6


def test_label_normalization() -> None:
    """Verify label classification across Botnet, Normal, and Background."""
    adapter = Ctu13Adapter()
    raw = pd.Series([
        "flow=From-Botnet-V42-TCP-Attempt-SPAM",
        "flow=From-Botnet-V42-CC",
        "flow=From-Normal-V42-Grill",
        "flow=Background-Established-cmp",
        "flow=From-Botnet-PortScan",
        "flow=Background-UDP-Attempt",
    ])
    bin_lbls, cat_lbls = adapter.normalize_labels(raw)

    assert list(bin_lbls) == [1, 1, 0, 0, 1, 0]
    assert cat_lbls[0] == "Bot" or "Spam" in cat_lbls[0]
    assert cat_lbls[1] == "Bot"
    assert cat_lbls[2] == "Normal"
    assert cat_lbls[3] == "Background"
    assert cat_lbls[4] == "PortScan"


def test_load_and_normalize(synthetic_binetflow_file: Path) -> None:
    """Verify end-to-end ingestion, feature mapping, and quality report."""
    adapter = Ctu13Adapter()
    clean_df, quality, provenance = adapter.load_and_normalize(
        data_paths=[synthetic_binetflow_file],
        max_rows=10,
    )

    assert len(clean_df) == 10
    assert quality.usable_rows == 10
    assert quality.benign_count == 6
    assert quality.attack_count == 4
    assert quality.attack_ratio == 0.4

    # Check canonical feature dimensions
    assert clean_df.attrs["features_holder"].array.shape == (10, CANONICAL_FEATURE_DIM)
    assert np.all(np.isfinite(clean_df.attrs["features_holder"].array))
    assert clean_df["scenario_id"].iloc[0] == "scenario_1"


def test_chronological_ordering(tmp_path: Path) -> None:
    """Verify out-of-order flows are chronologically sorted without leakage."""
    out_of_order_file = tmp_path / "out_of_order.binetflow"
    content = (
        "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label\n"
        "2011/08/10 10:00:00.000000,1.0,tcp,1.1.1.1,80,->,2.2.2.2,80,CON,0,0,10,1000,500,flow=Normal\n"
        "2011/08/10 09:00:00.000000,1.0,tcp,1.1.1.1,80,->,2.2.2.2,80,CON,0,0,10,1000,500,flow=Normal\n"
        "2011/08/10 09:30:00.000000,1.0,tcp,1.1.1.1,80,->,2.2.2.2,80,CON,0,0,10,1000,500,flow=Botnet\n"
    )
    out_of_order_file.write_text(content, encoding="utf-8")

    adapter = Ctu13Adapter()
    clean_df, _, _ = adapter.load_and_normalize([out_of_order_file])

    ts = clean_df["flow_start_time"].to_numpy()
    assert np.all(ts[:-1] <= ts[1:]), "Timestamps must be non-decreasing"
    assert clean_df["label_int"].tolist() == [0, 1, 0]  # 09:00 (Normal), 09:30 (Botnet), 10:00 (Normal)


def test_280d_network_state_compatibility(synthetic_binetflow_file: Path) -> None:
    """Verify CTU-13 can be deterministically encoded into 280-D Network State vectors."""
    adapter = Ctu13Adapter()
    clean_df, _, _ = adapter.load_and_normalize([synthetic_binetflow_file])
    feats = clean_df.attrs["features_holder"].array

    # Build windows of length T=2
    T_flow = 2
    win_X = [feats[i : i + T_flow] for i in range(len(feats) - T_flow + 1)]
    win_X = np.stack(win_X, axis=0)

    encoder = NetworkStateEncoder()
    batch = encoder.encode_batch(win_X)

    assert batch.states.shape == (len(win_X), STATE_FEATURE_DIM)
    assert batch.states.shape[1] == 280
    assert np.all(np.isfinite(batch.states))


def test_temporal_pipeline_and_future_pairs(synthetic_binetflow_file: Path) -> None:
    """Verify temporal sequence generation (T=5) and future pairs (k=1, 3, 5)."""
    adapter = Ctu13Adapter()
    clean_df, _, _ = adapter.load_and_normalize([synthetic_binetflow_file])
    feats = clean_df.attrs["features_holder"].array

    # Replicate features to form 20 windows
    replicated = np.tile(feats, (3, 1))[:25]
    win_X = [replicated[i : i + 2] for i in range(len(replicated) - 1)]
    win_X = np.stack(win_X, axis=0)

    encoder = NetworkStateEncoder()
    states_batch = encoder.encode_batch(win_X)

    builder = FutureStateDatasetBuilder(horizons=[1, 3, 5], sequence_length=5)
    ds = builder.build_future_pairs(states_batch, sequence_length=5)

    assert ds.total_usable_pairs > 0
    for k in [1, 3, 5]:
        b = ds.batches[k]
        assert b.current_states.ndim == 3
        assert b.current_states.shape[1] == 5
        assert b.current_states.shape[2] == 280
        assert b.target_states.shape[1] == 280


def test_unseen_scenario_split_isolation() -> None:
    """Verify that unseen scenario is completely disjoint from train and val."""
    split = Ctu13ScenarioSplitter.get_unseen_split_definition()

    unseen = set(split["unseen_test_scenarios"])
    train = set(split["train_scenarios"])
    val = set(split["val_scenarios"])
    known_test = set(split["known_test_scenarios"])

    assert len(unseen.intersection(train)) == 0, "LEAKAGE: Unseen scenario found in train"
    assert len(unseen.intersection(val)) == 0, "LEAKAGE: Unseen scenario found in val"
    assert len(unseen.intersection(known_test)) == 0, "LEAKAGE: Unseen scenario in known test"
    assert len(train.intersection(val)) == 0, "LEAKAGE: Train and val scenarios overlap"
    assert split["unseen_scenario"] == "scenario_10"
