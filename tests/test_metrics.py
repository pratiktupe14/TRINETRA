"""Tests for evaluation metrics in CyberHeal AI."""

import numpy as np
import pytest

from shield_rca.evaluation.metrics import compute_multiclass_metrics


def test_compute_multiclass_metrics_perfect():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    label_names = ["BENIGN", "Bot", "DDoS"]

    res = compute_multiclass_metrics(y_true, y_pred, label_names=label_names, num_classes=3)

    assert res["macro_f1"] == 1.0
    assert res["weighted_f1"] == 1.0
    assert res["confusion_matrix"].shape == (3, 3)
    assert np.trace(res["confusion_matrix"]) == 6


def test_compute_multiclass_metrics_with_proba():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.8, 0.2],
        [0.1, 0.9],
    ])
    label_names = ["BENIGN", "Bot"]

    res = compute_multiclass_metrics(y_true, y_pred, y_proba=y_proba, label_names=label_names, num_classes=2)

    assert "auroc_ovr" in res
    assert res["auroc_ovr"] == 1.0
