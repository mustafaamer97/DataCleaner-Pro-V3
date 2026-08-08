"""
profiling.py — Smart Data Profiling Engine for DataCleaner Pro V3.

Analyzes a DataFrame and returns a structured profile with
detected column types, warnings, and cleaning recommendations.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


# ── Regex Patterns ────────────────────────────────────────────────────────────

_EMAIL_RE   = re.compile(r"^[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}$")
_PHONE_RE   = re.compile(r"^\+?[\d\s\-().]{7,20}$")
_URL_RE     = re.compile(r"^https?://|^www\.", re.IGNORECASE)
_DATE_RE    = re.compile(
    r"\b(\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4})\b|"
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{1,2}[,\s]+\d{2,4}\b",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(r"^[\$€£¥₹]?\s?\d[\d,\.]*$")
_PCT_RE      = re.compile(r"^\d+\.?\d*\s?%$")


# ── Column Type Detection ─────────────────────────────────────────────────────

def _sample_col(series: pd.Series, n: int = 200) -> pd.Series:
    """Return a non-null sample of a Series for pattern matching."""
    non_null = series.dropna()
    return non_null.sample(min(n, len(non_null)), random_state=42) if len(non_null) > n else non_null


def _match_ratio(series: pd.Series, pattern: re.Pattern, n: int = 200) -> float:
    """Return the fraction of sampled non-null values matching a regex."""
    sample = _sample_col(series, n).astype(str)
    if sample.empty:
        return 0.0
    return float(sample.str.match(pattern).mean())


def detect_column_type(series: pd.Series) -> str:
    """
    Detect the semantic type of a column.

    Returns one of:
        'email', 'phone', 'url', 'date', 'currency',
        'percentage', 'id', 'numeric', 'categorical', 'text', 'unknown'
    """
    # Already numeric dtype
    if pd.api.types.is_numeric_dtype(series):
        # Potential ID: all unique integers
        if series.nunique() == len(series.dropna()) and pd.api.types.is_integer_dtype(series):
            return "id"
        return "numeric"

    # Datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    # Object / string — use regex heuristics
    if _match_ratio(series, _EMAIL_RE) >= 0.7:
        return "email"
    if _match_ratio(series, _PHONE_RE) >= 0.6:
        return "phone"
    if _match_ratio(series, _URL_RE) >= 0.5:
        return "url"

    # Date: try pandas parser on a sample
    sample = _sample_col(series, 100).astype(str)
    if not sample.empty:
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() >= 0.7:
                return "date"
        except Exception:
            pass

    if _match_ratio(series, _CURRENCY_RE) >= 0.6:
        return "currency"
    if _match_ratio(series, _PCT_RE) >= 0.6:
        return "percentage"

    # Cardinality-based
    nunique = series.nunique()
    nrows   = len(series.dropna())
    if nrows == 0:
        return "unknown"

    ratio = nunique / nrows
    if ratio < 0.05 or nunique <= 20:
        return "categorical"
    if ratio > 0.9:
        return "text"
    return "categorical"


# ── Main Profiler ─────────────────────────────────────────────────────────────

def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run a full profile on a DataFrame.

    Returns a structured dict with metadata, column profiles,
    warnings, and cleaning recommendations.
    """
    rows, cols = df.shape

    # ── Basic stats ──────────────────────────────────────
    total_cells     = rows * cols
    missing_count   = int(df.isnull().sum().sum())
    missing_pct     = round(missing_count / max(total_cells, 1) * 100, 2)
    dup_rows        = int(df.duplicated().sum())
    dup_row_pct     = round(dup_rows / max(rows, 1) * 100, 2)

    # ── Column-level profiling ───────────────────────────
    col_profiles: dict[str, dict] = {}
    type_groups: dict[str, list[str]] = {
        "numeric": [], "categorical": [], "date": [],
        "email": [], "phone": [], "url": [],
        "currency": [], "percentage": [], "id": [],
        "text": [], "unknown": [],
    }
    constant_cols:  list[str] = []
    empty_cols:     list[str] = []
    dup_col_groups: list[list[str]] = []

    for col in df.columns:
        series = df[col]
        col_missing     = int(series.isnull().sum())
        col_missing_pct = round(col_missing / max(rows, 1) * 100, 2)
        nunique         = int(series.nunique())
        sem_type        = detect_column_type(series)

        col_profiles[col] = {
            "dtype":       str(series.dtype),
            "semantic":    sem_type,
            "missing":     col_missing,
            "missing_pct": col_missing_pct,
            "unique":      nunique,
            "sample":      _get_sample_values(series),
        }

        type_groups.setdefault(sem_type, []).append(col)

        if col_missing == rows:
            empty_cols.append(col)
        if nunique <= 1:
            constant_cols.append(col)

    # ── Duplicate columns (by content hash) ─────────────
    seen_hashes: dict[int, str] = {}
    for col in df.columns:
        try:
            h = hash(df[col].to_json())
        except Exception:
            continue
        if h in seen_hashes:
            # Group them
            found = False
            for grp in dup_col_groups:
                if seen_hashes[h] in grp:
                    grp.append(col)
                    found = True
                    break
            if not found:
                dup_col_groups.append([seen_hashes[h], col])
        else:
            seen_hashes[h] = col

    # ── Warnings & Recommendations ───────────────────────
    warnings:         list[str] = []
    recommendations:  list[str] = []

    if missing_pct > 0:
        warnings.append(f"{missing_pct}% missing values across the dataset")
        recommendations.append("Review and fill or drop missing values")

    if dup_rows > 0:
        warnings.append(f"{dup_rows:,} exact duplicate rows detected ({dup_row_pct}%)")
        recommendations.append("Remove exact duplicate rows")

    if constant_cols:
        warnings.append(
            f"{len(constant_cols)} constant column(s): {', '.join(constant_cols)}"
        )
        recommendations.append("Remove constant/zero-variance columns")

    if empty_cols:
        warnings.append(f"{len(empty_cols)} completely empty column(s): {', '.join(empty_cols)}")
        recommendations.append("Remove empty columns")

    if dup_col_groups:
        warnings.append(f"{len(dup_col_groups)} duplicated column group(s) detected")
        recommendations.append("Remove duplicated columns")

    if type_groups["email"]:
        recommendations.append(
            f"Normalize email addresses in: {', '.join(type_groups['email'])}"
        )

    if type_groups["date"]:
        recommendations.append(
            f"Standardize date formats in: {', '.join(type_groups['date'])}"
        )

    if type_groups["phone"]:
        recommendations.append(
            f"Normalize phone numbers in: {', '.join(type_groups['phone'])}"
        )

    return {
        "rows":              rows,
        "columns":           cols,
        "missing_count":     missing_count,
        "missing_pct":       missing_pct,
        "duplicate_rows":    dup_rows,
        "duplicate_row_pct": dup_row_pct,
        "constant_cols":     constant_cols,
        "empty_cols":        empty_cols,
        "dup_col_groups":    dup_col_groups,
        "col_profiles":      col_profiles,
        "type_groups":       type_groups,
        "warnings":          warnings,
        "recommendations":   recommendations,
    }


def _get_sample_values(series: pd.Series, n: int = 3) -> list[str]:
    """Return up to n non-null sample values as strings."""
    non_null = series.dropna()
    samples  = non_null.head(n).tolist()
    return [str(s)[:60] for s in samples]
