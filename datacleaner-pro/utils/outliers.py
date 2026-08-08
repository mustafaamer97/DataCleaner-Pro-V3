"""
outliers.py — Safe Outlier Detection for DataCleaner Pro V3.

Uses IQR and Z-score methods.
NEVER automatically deletes outliers — only flags them for review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_outliers_iqr(series: pd.Series) -> dict:
    """
    Detect outliers using the IQR (Interquartile Range) method.

    Returns a dict with outlier count, bounds, and outlier values.
    """
    clean = series.dropna()
    if len(clean) < 4:
        return {"method": "iqr", "count": 0, "lower": None, "upper": None, "values": []}

    q1  = float(clean.quantile(0.25))
    q3  = float(clean.quantile(0.75))
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask    = (clean < lower) | (clean > upper)
    outlier_vals = clean[mask].tolist()

    return {
        "method":   "iqr",
        "count":    int(mask.sum()),
        "lower":    round(lower, 4),
        "upper":    round(upper, 4),
        "q1":       round(q1, 4),
        "q3":       round(q3, 4),
        "values":   outlier_vals[:20],  # cap preview
    }


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> dict:
    """
    Detect outliers using the Z-score method.

    Returns a dict with outlier count and values.
    """
    clean = series.dropna()
    if len(clean) < 4:
        return {"method": "zscore", "count": 0, "threshold": threshold, "values": []}

    mean  = float(clean.mean())
    std   = float(clean.std())

    if std == 0:
        return {"method": "zscore", "count": 0, "threshold": threshold, "values": []}

    zscores = (clean - mean) / std
    mask    = zscores.abs() > threshold
    outlier_vals = clean[mask].tolist()

    return {
        "method":    "zscore",
        "count":     int(mask.sum()),
        "threshold": threshold,
        "mean":      round(mean, 4),
        "std":       round(std, 4),
        "values":    outlier_vals[:20],
    }


def detect_all_outliers(df: pd.DataFrame) -> dict[str, dict]:
    """
    Run outlier detection on all numeric columns.

    Returns a dict mapping column name → outlier report.
    Only includes columns where outliers were found.
    """
    results: dict[str, dict] = {}
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        iqr_result = detect_outliers_iqr(df[col])
        if iqr_result["count"] > 0:
            results[col] = iqr_result

    return results
