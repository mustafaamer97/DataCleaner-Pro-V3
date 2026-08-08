"""
outliers.py — Safe Outlier Detection for DataCleaner Pro V3.

Uses IQR and Z-score methods.
NEVER automatically deletes outliers — only flags them for review.
"""

from __future__ import annotations

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# IQR OUTLIERS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_outliers_iqr(series: pd.Series) -> dict:
    """
    Detect outliers using the IQR (Interquartile Range) method.

    Non-finite values (+/-inf) are ignored during statistical calculations.

    Returns:
        Dict containing:
        - method
        - count
        - lower / upper bounds
        - q1 / q3
        - preview of outlier values
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")

    # Keep only finite numeric values.
    clean = pd.to_numeric(series, errors="coerce")
    clean = clean[clean.notna() & clean.isfinite()]

    if len(clean) < 4:
        return {
            "method": "iqr",
            "count": 0,
            "lower": None,
            "upper": None,
            "q1": None,
            "q3": None,
            "values": [],
        }

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1

    # Constant / zero-variance data has no IQR outliers.
    if iqr == 0:
        return {
            "method": "iqr",
            "count": 0,
            "lower": round(q1, 4),
            "upper": round(q3, 4),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "values": [],
        }

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (clean < lower) | (clean > upper)
    outlier_vals = clean[mask].tolist()

    return {
        "method": "iqr",
        "count": int(mask.sum()),
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "values": outlier_vals[:20],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Z-SCORE OUTLIERS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0,
) -> dict:
    """
    Detect outliers using the Z-score method.

    Non-finite values (+/-inf) are ignored during statistical calculations.

    Args:
        series:
            Numeric pandas Series.
        threshold:
            Absolute Z-score above which a value is considered an outlier.

    Returns:
        Dict containing:
        - method
        - count
        - threshold
        - mean / std
        - preview of outlier values
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")

    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = 3.0

    if threshold <= 0:
        threshold = 3.0

    clean = pd.to_numeric(series, errors="coerce")
    clean = clean[clean.notna() & clean.isfinite()]

    if len(clean) < 4:
        return {
            "method": "zscore",
            "count": 0,
            "threshold": threshold,
            "values": [],
        }

    mean = float(clean.mean())
    std = float(clean.std())

    # Constant / zero-variance data has no Z-score outliers.
    if std == 0 or pd.isna(std):
        return {
            "method": "zscore",
            "count": 0,
            "threshold": threshold,
            "mean": round(mean, 4),
            "std": 0.0,
            "values": [],
        }

    zscores = (clean - mean) / std
    mask = zscores.abs() > threshold
    outlier_vals = clean[mask].tolist()

    return {
        "method": "zscore",
        "count": int(mask.sum()),
        "threshold": threshold,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "values": outlier_vals[:20],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ALL NUMERIC COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_all_outliers(
    df: pd.DataFrame,
) -> dict[str, dict]:
    """
    Run IQR outlier detection on all numeric columns.

    Returns:
        Dict mapping column name → outlier report.

    Notes:
        - Only numeric columns are analyzed.
        - Columns with zero detected outliers are omitted.
        - Outliers are NEVER deleted automatically.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    results: dict[str, dict] = {}

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        result = detect_outliers_iqr(df[col])

        if result["count"] > 0:
            results[col] = result

    return results
