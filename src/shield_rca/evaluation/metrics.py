"""Pure multiclass evaluation metric functions for CyberHeal AI.

All functions are stateless — no I/O, no global state, no GUI side effects.
Used for offline verification and model validation.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    label_names: list[str] | None = None,
    num_classes: int | None = None,
) -> dict:
    """Compute the full multiclass evaluation suite.

    Args:
        y_true: Ground-truth integer labels, shape (N,).
        y_pred: Predicted integer labels, shape (N,).
        y_proba: Predicted class probabilities, shape (N, C). Optional.
        label_names: Human-readable class names aligned to integer indices.
        num_classes: Number of classes (inferred from y_true if None).

    Returns:
        Dict with keys:
          macro_f1, weighted_f1, per_class (dict), confusion_matrix,
          auroc_ovr (if y_proba), aupr (if y_proba), classification_report.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if num_classes is None:
        max_idx = max(int(y_true.max()) if len(y_true) > 0 else 0, int(y_pred.max()) if len(y_pred) > 0 else 0)
        num_classes = max_idx + 1

    classes = list(range(num_classes))
    names = label_names if label_names else [str(c) for c in classes]

    macro_f1: float = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1: float = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    report: dict = classification_report(
        y_true,
        y_pred,
        labels=classes,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )

    per_class: dict[str, dict[str, float]] = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }
        for name in names
        if name in report
    }

    cm: np.ndarray = confusion_matrix(y_true, y_pred, labels=classes)

    result: dict = {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    if y_proba is not None and y_proba.ndim == 2:
        result.update(_compute_roc_pr(y_true, y_proba, classes, names))

    return result


def _compute_roc_pr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: list[int],
    names: list[str],
) -> dict:
    """Compute per-class AUROC (OVR) and AUPR."""
    y_bin = label_binarize(y_true, classes=classes)
    if len(classes) == 2 and y_bin.shape[1] == 1:
        y_bin = np.column_stack([1 - y_bin[:, 0], y_bin[:, 0]])

    auroc_per_class: dict[str, float] = {}
    aupr_per_class: dict[str, float] = {}

    for i, name in enumerate(names):
        if i >= y_proba.shape[1] or y_bin.shape[1] <= i:
            continue
        col_true = y_bin[:, i]
        col_score = y_proba[:, i]
        if col_true.sum() == 0 or (1 - col_true).sum() == 0:
            continue

        try:
            auroc_per_class[name] = float(roc_auc_score(col_true, col_score))
        except ValueError:
            pass

        try:
            precision, recall, _ = precision_recall_curve(col_true, col_score)
            trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
            if trapz_fn is not None:
                aupr_per_class[name] = float(trapz_fn(precision, recall))
            else:
                from scipy.integrate import trapezoid as scipy_trapz
                aupr_per_class[name] = float(scipy_trapz(precision, recall))
        except (ValueError, ZeroDivisionError):
            pass

    auroc_macro = (
        float(np.mean(list(auroc_per_class.values()))) if auroc_per_class else 0.0
    )
    aupr_macro = (
        float(np.mean(list(aupr_per_class.values()))) if aupr_per_class else 0.0
    )

    return {
        "auroc_ovr": auroc_macro,
        "aupr": aupr_macro,
        "auroc_per_class": auroc_per_class,
        "aupr_per_class": aupr_per_class,
    }
