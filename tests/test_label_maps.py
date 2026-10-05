"""Tests for canonical attack label mappings in CyberHeal AI."""

import pandas as pd
import pytest

from shield_rca.models.label_maps import (
    INT_TO_LABEL,
    LABEL_TO_INT,
    assign_labels,
    get_all_labels,
    get_label_int,
    get_label_name,
    is_attack,
)


def test_label_mappings_consistency():
    labels = get_all_labels()
    assert len(labels) == 15
    assert labels[0] == "BENIGN"

    for i in range(15):
        name = get_label_name(i)
        assert get_label_int(name) == i

    assert not is_attack("BENIGN")
    assert not is_attack(0)
    assert is_attack("DDoS")
    assert is_attack(2)


def test_aliases_resolution():
    assert get_label_int("DoS attacks-Hulk") == 4
    assert get_label_int("FTP-BruteForce") == 7
    assert get_label_int("Benign") == 0


def test_assign_labels():
    df = pd.DataFrame({"attack_class": ["BENIGN", "DDoS", "FTP-BruteForce"]})
    assigned = assign_labels(df, raw_label_col="attack_class", int_label_col="label_int")
    assert assigned["label_int"].tolist() == [0, 2, 7]
