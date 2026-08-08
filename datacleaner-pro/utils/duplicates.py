"""
duplicates.py — Duplicate Detection for DataCleaner Pro V3.

Exact duplicate detection is automatic.
Fuzzy duplicate detection NEVER silently deletes records —
it returns candidates for user review.

The fuzzy engine is intentionally bounded to prevent O(n²)
work from freezing the application.
"""

from __future__ import annotations

import pandas as pd

from utils.helpers import MAX_ROWS_FUZZY


# Maximum number of fuzzy candidate pairs returned.
# Prevents a highly repetitive dataset from generating millions
# of records and exhausting memory.
MAX_FUZZY_RESULTS = 10_000


# ── Exact Duplicates ──────────────────────────────────────────────────────────

def get_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return all rows that are exact duplicates, excluding the first occurrence.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    return df[df.duplicated(keep="first")]


def remove_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove exact duplicate rows.

    Returns:
        (cleaned_df, number_of_rows_removed)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    before = len(df)
    cleaned = df.drop_duplicates(keep="first")

    return cleaned, before - len(cleaned)


# ── Fuzzy Duplicates ──────────────────────────────────────────────────────────

def find_fuzzy_duplicates(
    df: pd.DataFrame,
    columns: list[str],
    threshold: float = 0.85,
) -> pd.DataFrame | None:
    """
    Find potential fuzzy duplicates in selected columns.

    Fuzzy matching NEVER deletes records. It only returns candidate
    pairs for user review.

    The implementation uses RapidFuzz's optimized matching instead
    of a naive O(n²) Python loop.

    Args:
        df:
            Input DataFrame.

        columns:
            Columns used to build the comparison key.

        threshold:
            Similarity threshold from 0.0 to 1.0.
            Default is 0.85.

    Returns:
        DataFrame containing:

            Index A
            Index B
            Key A
            Key B
            Similarity

        Returns None when:
            - dataset exceeds MAX_ROWS_FUZZY
            - RapidFuzz is unavailable

        Raises:
            TypeError:
                If df is not a DataFrame.

            ValueError:
                If columns are invalid or threshold is outside
                the allowed range.
    """

    # ── Validate DataFrame ────────────────────────────────────────────────

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    # ── Validate threshold ────────────────────────────────────────────────

    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise ValueError("threshold must be a number between 0.0 and 1.0.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0.")

    # ── Validate columns ──────────────────────────────────────────────────

    if not columns:
        raise ValueError("At least one column must be selected.")

    missing_columns = [
        col for col in columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Selected columns do not exist: "
            + ", ".join(map(str, missing_columns))
        )

    # Remove duplicates while preserving column order.
    columns = list(dict.fromkeys(columns))

    # ── Dataset size guard ────────────────────────────────────────────────

    if len(df) > MAX_ROWS_FUZZY:
        return None

    # Nothing to compare.
    if len(df) < 2:
        return pd.DataFrame(
            columns=[
                "Index A",
                "Index B",
                "Key A",
                "Key B",
                "Similarity",
            ]
        )

    # ── Optional dependency ───────────────────────────────────────────────

    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        return None

    # ── Normalize values ──────────────────────────────────────────────────

    def normalize(value) -> str:
        """
        Normalize a cell for fuzzy comparison.

        Null-like values become empty strings.
        Whitespace is collapsed and text is lowercased.
        """
        if pd.isna(value):
            return ""

        return " ".join(str(value).lower().split())

    # ── Build comparison keys ─────────────────────────────────────────────

    keys: list[str] = []

    for _, row in df[columns].iterrows():
        key = " | ".join(
            normalize(row[col])
            for col in columns
        )
        keys.append(key)

    # Avoid comparing completely empty keys.
    valid_positions = [
        i
        for i, key in enumerate(keys)
        if key
    ]

    if len(valid_positions) < 2:
        return pd.DataFrame(
            columns=[
                "Index A",
                "Index B",
                "Key A",
                "Key B",
                "Similarity",
            ]
        )

    # ── Fuzzy matching ───────────────────────────────────────────────────
    #
    # RapidFuzz performs the similarity calculations much more efficiently
    # than a nested Python loop.
    #
    # We still explicitly ensure each pair is emitted only once.

    results: list[dict] = []

    choices = {
        position: keys[position]
        for position in valid_positions
    }

    score_cutoff = threshold * 100

    for position in valid_positions:

        query = keys[position]

        matches = process.extract(
            query,
            choices,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=score_cutoff,
            limit=None,
        )

        for _, score, matched_position in matches:

            # Same row.
            if matched_position == position:
                continue

            # Prevent duplicate A/B pairs.
            if matched_position <= position:
                continue

            matched_key = keys[matched_position]

            # Exact normalized matches are already handled by
            # exact duplicate detection.
            if query == matched_key:
                continue

            similarity = round(score / 100.0, 3)

            results.append(
                {
                    "Index A": df.index[position],
                    "Index B": df.index[matched_position],
                    "Key A": query,
                    "Key B": matched_key,
                    "Similarity": similarity,
                }
            )

            # Hard safety limit.
            if len(results) >= MAX_FUZZY_RESULTS:
                break

        if len(results) >= MAX_FUZZY_RESULTS:
            break

    # ── Return results ────────────────────────────────────────────────────

    if not results:
        return pd.DataFrame(
            columns=[
                "Index A",
                "Index B",
                "Key A",
                "Key B",
                "Similarity",
            ]
        )

    return (
        pd.DataFrame(results)
        .sort_values(
            "Similarity",
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )
