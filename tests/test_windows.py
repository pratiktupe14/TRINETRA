"""Tests for temporal windowing in CyberHeal AI."""

import numpy as np
import pandas as pd
import pytest

from shield_rca.telemetry.windows import (
    WindowDataset,
    WindowRingBuffer,
    build_timeseries_windows,
    encode_categorical_cols,
)


def test_build_timeseries_windows_basic():
    # Construct dummy flows DataFrame
    data = {
        "flow_start_time": [10.0 + i for i in range(50)],
        "tot_fwd_pkts": [float(i * 2) for i in range(50)],
        "tot_bwd_pkts": [float(i * 3) for i in range(50)],
        "label_int": [0] * 40 + [1] * 10,
        "scenario_id": ["group_1"] * 50,
    }
    df = pd.DataFrame(data)

    dataset = build_timeseries_windows(
        df,
        feature_cols=["tot_fwd_pkts", "tot_bwd_pkts"],
        window_size=10,
        stride=5,
        min_points_per_window=5,
        sequence_label_rule="majority",
    )

    assert isinstance(dataset, WindowDataset)
    assert dataset.X.ndim == 3
    assert dataset.X.shape[1] == 10  # T=10
    assert dataset.X.shape[2] == 2   # F=2
    assert len(dataset.y) == dataset.X.shape[0]


def test_encode_categorical_cols():
    df = pd.DataFrame({"proto": [6, 17, 1, 6, 99]})
    specs = [{"col": "proto", "known_values": [6, 17, 1], "value_names": ["tcp", "udp", "icmp"]}]
    encoded = encode_categorical_cols(df, specs)

    assert "proto_tcp" in encoded.columns
    assert "proto_udp" in encoded.columns
    assert "proto_icmp" in encoded.columns
    assert "proto_other" in encoded.columns
    assert encoded["proto_tcp"].tolist() == [1, 0, 0, 1, 0]
    assert encoded["proto_other"].tolist() == [0, 0, 0, 0, 1]


def test_window_ring_buffer():
    buf = WindowRingBuffer(window_size=5)
    assert not buf.is_ready
    assert buf.get_window() is None

    for i in range(5):
        buf.append(np.array([i, i * 2], dtype=np.float32))

    assert buf.is_ready
    w = buf.get_window()
    assert w is not None
    assert w.shape == (1, 5, 2)
