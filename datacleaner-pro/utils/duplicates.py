"""
duplicates.py — Duplicate Detection for DataCleaner Pro V3.

Exact duplicate detection is automatic.
Fuzzy duplicate detection NEVER silently deletes records —
it returns candidates for user review.
"""

from __future__ import annotations

import pandas as pd

from utils.helpers import MAX_ROWS_FUZZY


# ── Exact Duplicates ──────────────────────────────────────────────────────────

def get_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return all rows that are exact duplicates (keeping the first occurrence).
    """
    return df[df.duplicated(keep="first")]


def remove_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove exact duplicate rows.

    Returns:
        (cleaned_df, number_of_rows_removed)
    """
    before = len(df)
    df = df.drop_duplicates(keep="first")
    return df, before - len(df)


# ── Fuzzy Duplicates ──────────────────────────────────────────────────────────

def find_fuzzy_duplicates(
    df: pd.DataFrame,
    columns: list[str],
    threshold: float = 0.85,
) -> pd.DataFrame | None:
    """
    Find potential fuzzy duplicates in specified string columns.

    Uses a conservative normalized token-sort comparison.

    IMPORTANT:
        - This function NEVER deletes records.
        - It returns a DataFrame of CANDIDATE pairs for user review.
        - Returns None if the dataset is too large (> MAX_ROWS_FUZZY).

    Args:
        df:        Input DataFrame.
        columns:   Columns to compare (must be string/object dtype).
        threshold: Similarity threshold 0.0–1.0. Default 0.85.

    Returns:
        DataFrame with columns [idx_a, idx_b, row_a_preview, row_b_preview, similarity]
        or None if dataset too large.
    """
    if len(df) > MAX_ROWS_FUZZY:
        return None  # Caller must warn the user

    try:
        from rapidfuzz import fuzz  # optional dependency
    except ImportError:
        return None  # rapidfuzz not installed — skip silently

    # Normalize: lowercase, strip, collapse spaces
    def normalize(val) -> str:
        if pd.isna(val):
            return ""
        return " ".join(str(val).lower().split())

    # Build a combined key from selected columns
    keys = df[columns].apply(
        lambda row: " | ".join(normalize(row[c]) for c in columns),
        axis=1,
    ).tolist()

    n       = len(keys)
    results = []

    # O(n²) — only safe for small datasets (guarded above)
    for i in range(n):
        for j in range(i + 1, n):
            score = fuzz.token_sort_ratio(keys[i], keys[j]) / 100.0
            if score >= threshold and keys[i] != keys[j]:
                results.append(
                    {
                        "Index A":    df.index[i],
                        "Index B":    df.index[j],
                        "Key A":      keys[i],
                        "Key B":      keys[j],
                        "Similarity": round(score, 3),
                    }
                )

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("Similarity", ascending=False)
