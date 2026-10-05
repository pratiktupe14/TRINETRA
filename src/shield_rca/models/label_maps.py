"""Canonical attack label mapping and taxonomy for CyberHeal AI.

Standardizes 15 multi-class attack categories derived from CIC-IDS2017/2018 datasets.
Used for translating numeric model predictions into human-readable attack descriptions
for SOC analysts and autonomous RCA.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical 15-class integer to label mapping
INT_TO_LABEL: dict[int, str] = {
    0: "BENIGN",
    1: "Bot",
    2: "DDoS",
    3: "DoS GoldenEye",
    4: "DoS Hulk",
    5: "DoS Slowhttptest",
    6: "DoS slowloris",
    7: "FTP-Patator",
    8: "Heartbleed",
    9: "Infiltration",
    10: "PortScan",
    11: "SSH-Patator",
    12: "Web Attack - Brute Force",
    13: "Web Attack - Sql Injection",
    14: "Web Attack - XSS",
}

# Inverted label to integer mapping
LABEL_TO_INT: dict[str, int] = {v: k for k, v in INT_TO_LABEL.items()}

# Common label aliases from different dataset versions and logging formats
LABEL_ALIASES: dict[str, str] = {
    "Benign": "BENIGN",
    "benign": "BENIGN",
    "Normal": "BENIGN",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "DDOS attack-HOIC": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",
    "DoS attacks-GoldenEye": "DoS GoldenEye",
    "DoS attacks-Hulk": "DoS Hulk",
    "DoS attacks-SlowHTTPTest": "DoS Slowhttptest",
    "DoS attacks-Slowloris": "DoS slowloris",
    "FTP-BruteForce": "FTP-Patator",
    "SSH-Bruteforce": "SSH-Patator",
    "Web Attack \x96 Brute Force": "Web Attack - Brute Force",
    "Web Attack \x96 XSS": "Web Attack - XSS",
    "Web Attack \x96 Sql Injection": "Web Attack - Sql Injection",
    "Web Attack – Brute Force": "Web Attack - Brute Force",
    "Web Attack – XSS": "Web Attack - XSS",
    "Web Attack – Sql Injection": "Web Attack - Sql Injection",
    "Brute Force -Web": "Web Attack - Brute Force",
    "Brute Force -XSS": "Web Attack - XSS",
    "SQL Injection": "Web Attack - Sql Injection",
}


def get_label_name(idx: int) -> str:
    """Retrieve string label name from integer class index."""
    return INT_TO_LABEL.get(idx, f"UNKNOWN_ATTACK_{idx}")


def get_label_int(name: str) -> int:
    """Retrieve integer class index from attack name (resolving aliases)."""
    clean_name = name.strip()
    canonical = LABEL_ALIASES.get(clean_name, clean_name)
    if canonical in LABEL_TO_INT:
        return LABEL_TO_INT[canonical]

    # Lowercase lookup
    lowered = canonical.lower()
    for k, v in LABEL_TO_INT.items():
        if k.lower() == lowered:
            return v

    raise KeyError(f"Unknown attack class label: '{name}'")


def is_attack(label: int | str) -> bool:
    """Check whether a label corresponds to an attack (non-benign)."""
    if isinstance(label, int):
        return label != 0
    clean = label.strip()
    canonical = LABEL_ALIASES.get(clean, clean).upper()
    return canonical != "BENIGN"


def get_all_labels() -> list[str]:
    """Return all 15 canonical attack class labels in index order."""
    return [INT_TO_LABEL[i] for i in range(len(INT_TO_LABEL))]


def assign_labels(
    df: pd.DataFrame,
    raw_label_col: str = "attack_class",
    int_label_col: str = "label_int",
    unknown_strategy: str = "drop",
) -> pd.DataFrame:
    """Map raw string attack labels in a DataFrame to integer indices.

    Args:
        df: Input DataFrame containing string labels.
        raw_label_col: Source column containing label strings.
        int_label_col: Destination column for integer indices.
        unknown_strategy: "drop" to filter out unrecognized labels, "error" to raise.

    Returns:
        DataFrame with integer label column added.
    """
    if raw_label_col not in df.columns:
        raise KeyError(f"Column '{raw_label_col}' not found in DataFrame")

    raw = df[raw_label_col].astype(str).str.strip()
    canonical = raw.map(LABEL_ALIASES).fillna(raw)
    mapped = canonical.map(LABEL_TO_INT)

    # Case-insensitive fallback
    if mapped.isna().any():
        lower_map = {k.lower(): v for k, v in LABEL_TO_INT.items()}
        mapped = mapped.fillna(canonical.str.lower().map(lower_map))

    unknown_mask = mapped.isna()
    if unknown_mask.any():
        unknown_labels = canonical[unknown_mask].unique().tolist()
        if unknown_strategy == "error":
            raise KeyError(f"Unknown labels found: {unknown_labels}")
        logger.warning("Dropping %d rows with unrecognized labels: %s", unknown_mask.sum(), unknown_labels)
        df = df[~unknown_mask].copy()
        mapped = mapped[~unknown_mask]

    df[int_label_col] = mapped.astype(int)
    return df
