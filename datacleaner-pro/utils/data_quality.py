"""
data_quality.py — Data Quality Intelligence Engine for DataCleaner Pro V3.

Analyzes a pandas DataFrame and returns a structured quality report
including an overall score, category scores, detected issues, and
actionable recommendations.

All analysis is deterministic, testable, and independent of Streamlit.
No external APIs or LLMs are required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Score band thresholds (inclusive lower bound)
SCORE_EXCELLENT  = 90
SCORE_GOOD       = 70
SCORE_ATTENTION  = 50
# Below SCORE_ATTENTION is considered Poor

# Issue severity labels
SEV_HIGH   = "HIGH"
SEV_MEDIUM = "MEDIUM"
SEV_LOW    = "LOW"

# Thresholds used during analysis
NEARLY_EMPTY_THRESHOLD       = 0.70   # >70 % missing → nearly empty
HIGH_MISSING_THRESHOLD       = 0.15   # >15 % missing → high missingness
CATEGORICAL_UNIQUE_THRESHOLD = 0.05   # unique/rows < 5 % → categorical candidate
KEY_UNIQUE_THRESHOLD         = 0.98   # unique/rows > 98 % → key candidate
KEY_MISSING_THRESHOLD        = 0.01   # < 1 % missing    → key candidate
MIXED_TYPE_STRING_RATIO      = 0.10   # > 10 % look-like-other-type → mixed

# Column name patterns that hint at identifier columns
_KEY_NAME_RE = re.compile(
    r"(^id$|[-_]id$|^id[-_]|uuid|guid|[-_]key$|^email$|[-_]email$)",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualityIssue:
    """A single detected data-quality issue."""
    severity:       str          # SEV_HIGH / SEV_MEDIUM / SEV_LOW
    category:       str          # completeness / uniqueness / consistency / validity
    column:         str | None   # None when issue is dataset-level
    title:          str
    detail:         str
    recommendation: str


@dataclass
class QualityReport:
    """
    Full data-quality report returned by analyze_data_quality().

    score        : 0–100 overall quality score.
    categories   : dict with keys completeness / uniqueness /
                   consistency / validity, each 0–100.
    issues       : list of QualityIssue ordered by severity.
    key_candidates         : column names that look like primary keys.
    categorical_candidates : column names that could be categorical.
    stats        : lightweight summary statistics for display.
    """
    score:                  int
    categories:             dict[str, int]
    issues:                 list[QualityIssue]
    key_candidates:         list[str]
    categorical_candidates: list[str]
    stats:                  dict[str, Any]

    # Derived convenience properties
    @property
    def score_label(self) -> str:
        if self.score >= SCORE_EXCELLENT:
            return "Excellent"
        if self.score >= SCORE_GOOD:
            return "Good"
        if self.score >= SCORE_ATTENTION:
            return "Needs Attention"
        return "Poor"

    @property
    def score_emoji(self) -> str:
        if self.score >= SCORE_EXCELLENT:
            return "🟢"
        if self.score >= SCORE_GOOD:
            return "🟡"
        if self.score >= SCORE_ATTENTION:
            return "🟠"
        return "🔴"

    @property
    def high_priority_issues(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == SEV_HIGH]

    @property
    def recommendations(self) -> list[str]:
        return [i.recommendation for i in self.issues]


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> int:
    """Clamp a float to [lo, hi] and return as int."""
    return int(max(lo, min(hi, value)))


def _pct(num: float, denom: float) -> float:
    """Return num/denom as a percentage, or 0 when denom is 0."""
    return (num / denom * 100.0) if denom > 0 else 0.0


def _looks_numeric(series: pd.Series, sample: int = 200) -> float:
    """
    Return the fraction of non-null string values that look numeric.
    Operates on a sample for performance.
    """
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return 0.0
    s = non_null.sample(min(sample, len(non_null)), random_state=42)
    numeric_re = re.compile(r"^[-+]?\d+(\.\d+)?([eE][-+]?\d+)?$")
    matched = s.apply(lambda v: bool(numeric_re.fullmatch(v.strip()))).sum()
    return float(matched) / len(s)


def _looks_date(series: pd.Series, sample: int = 200) -> float:
    """
    Return the fraction of non-null string values that look like dates.
    Operates on a sample for performance.
    """
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return 0.0
    s = non_null.sample(min(sample, len(non_null)), random_state=42)
    date_re = re.compile(
        r"(\d{4}[-/]\d{2}[-/]\d{2}"
        r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
        r"|(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d)",
        re.IGNORECASE,
    )
    matched = s.apply(lambda v: bool(date_re.search(v.strip()))).sum()
    return float(matched) / len(s)


def _severity_order(issue: QualityIssue) -> int:
    return {SEV_HIGH: 0, SEV_MEDIUM: 1, SEV_LOW: 2}.get(issue.severity, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY SCORERS
# ═══════════════════════════════════════════════════════════════════════════════

def _score_completeness(
    df: pd.DataFrame,
    issues: list[QualityIssue],
) -> int:
    """
    Completeness = 100 - weighted_missing_pct.

    Each column contributes equally.  Nearly-empty columns (>70 % missing)
    add a HIGH issue; columns above the high-missing threshold add MEDIUM.
    """
    rows, cols = df.shape
    if rows == 0 or cols == 0:
        return 100

    total_cells  = rows * cols
    total_missing = int(df.isnull().sum().sum())
    overall_pct  = _pct(total_missing, total_cells)

    # Per-column issues
    for col in df.columns:
        miss = int(df[col].isnull().sum())
        pct  = _pct(miss, rows)

        if pct > NEARLY_EMPTY_THRESHOLD * 100:
            issues.append(QualityIssue(
                severity=SEV_HIGH,
                category="completeness",
                column=col,
                title=f"Nearly empty column: '{col}'",
                detail=(
                    f"Column '{col}' is {pct:.1f}% empty "
                    f"({miss:,} of {rows:,} values missing)."
                ),
                recommendation=(
                    f"Consider removing column '{col}' as it contains "
                    f"very little usable data ({100 - pct:.1f}% filled)."
                ),
            ))
        elif pct > HIGH_MISSING_THRESHOLD * 100:
            issues.append(QualityIssue(
                severity=SEV_MEDIUM,
                category="completeness",
                column=col,
                title=f"High missingness in '{col}'",
                detail=(
                    f"Column '{col}' has {pct:.1f}% missing values "
                    f"({miss:,} of {rows:,} rows)."
                ),
                recommendation=(
                    f"Review missing records in '{col}'. "
                    "Consider filling, flagging, or dropping them "
                    "depending on their significance."
                ),
            ))

    score = _clamp(100.0 - overall_pct)
    return score


def _score_uniqueness(
    df: pd.DataFrame,
    issues: list[QualityIssue],
) -> int:
    """
    Uniqueness = 100 - duplicate_row_pct.

    Adds a HIGH issue when duplicate rows are detected.
    """
    rows = len(df)
    if rows == 0:
        return 100

    dup_count = int(df.duplicated().sum())
    dup_pct   = _pct(dup_count, rows)

    if dup_count > 0:
        sev = SEV_HIGH if dup_pct > 5.0 else SEV_MEDIUM
        issues.append(QualityIssue(
            severity=sev,
            category="uniqueness",
            column=None,
            title="Duplicate rows detected",
            detail=(
                f"{dup_count:,} duplicate row(s) found "
                f"({dup_pct:.1f}% of {rows:,} rows)."
            ),
            recommendation=(
                "Remove duplicate rows using the Auto-Clean feature "
                "to avoid skewed analysis results."
            ),
        ))

    return _clamp(100.0 - dup_pct)


def _score_consistency(
    df: pd.DataFrame,
    issues: list[QualityIssue],
    categorical_candidates: list[str],
) -> int:
    """
    Consistency covers:
    - Mixed-type columns (object columns containing both numeric/date
      strings and plain text).
    - Repeated-value (low-cardinality) columns that could be categorical.

    Penalty per mixed-type column: 5 points (max 25).
    Penalty per low-cardinality column: 1 point (max 10, informational).
    """
    rows = len(df)
    penalty = 0.0

    obj_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in obj_cols:
        series   = df[col].dropna()
        n        = len(series)
        nunique  = series.nunique()

        if n == 0:
            continue

        # ── Mixed-type detection ──────────────────────────────────────────────
        numeric_ratio = _looks_numeric(series)
        date_ratio    = _looks_date(series)

        is_mixed = False

        if 0.0 < numeric_ratio < (1.0 - MIXED_TYPE_STRING_RATIO):
            is_mixed = True
            issues.append(QualityIssue(
                severity=SEV_MEDIUM,
                category="consistency",
                column=col,
                title=f"Mixed numeric and text values in '{col}'",
                detail=(
                    f"Column '{col}' appears to contain both numeric values "
                    f"and non-numeric text ({numeric_ratio * 100:.0f}% numeric). "
                    "This may indicate data entry errors."
                ),
                recommendation=(
                    f"Review column '{col}' for mixed data types. "
                    "Non-numeric entries such as 'unknown' or 'N/A' "
                    "should be standardized before numeric analysis."
                ),
            ))
            penalty += 5.0

        elif 0.0 < date_ratio < (1.0 - MIXED_TYPE_STRING_RATIO) and not is_mixed:
            issues.append(QualityIssue(
                severity=SEV_MEDIUM,
                category="consistency",
                column=col,
                title=f"Inconsistent date formats in '{col}'",
                detail=(
                    f"Column '{col}' contains values that look like dates "
                    f"mixed with non-date text ({date_ratio * 100:.0f}% date-like)."
                ),
                recommendation=(
                    f"Standardize date values in '{col}' to a single format "
                    "(e.g. YYYY-MM-DD) using the date normalization feature."
                ),
            ))
            penalty += 5.0

        # ── Categorical candidate ─────────────────────────────────────────────
        unique_ratio = nunique / n if n > 0 else 1.0

        if (
            unique_ratio < CATEGORICAL_UNIQUE_THRESHOLD
            and nunique > 1
            and rows >= 50
        ):
            categorical_candidates.append(col)
            issues.append(QualityIssue(
                severity=SEV_LOW,
                category="consistency",
                column=col,
                title=f"Low-cardinality column: '{col}'",
                detail=(
                    f"Column '{col}' has only {nunique} unique value(s) "
                    f"across {n:,} rows ({unique_ratio * 100:.1f}% unique)."
                ),
                recommendation=(
                    f"Column '{col}' is a good candidate for categorical "
                    "encoding, which can reduce memory usage and improve "
                    "processing speed."
                ),
            ))
            penalty += 1.0

    return _clamp(100.0 - min(penalty, 35.0))


def _score_validity(
    df: pd.DataFrame,
    issues: list[QualityIssue],
    key_candidates: list[str],
) -> int:
    """
    Validity covers:
    - Columns that look like identifiers but contain duplicates.
    - Detection of potential primary-key candidates.

    Penalty per duplicated candidate-key column: 10 points.
    """
    rows    = len(df)
    penalty = 0.0

    if rows == 0:
        return 100

    for col in df.columns:
        series   = df[col].dropna()
        n        = len(series)
        nunique  = int(series.nunique())

        if n == 0:
            continue

        unique_ratio  = nunique / n
        missing_ratio = _pct(df[col].isnull().sum(), rows) / 100.0

        # ── Key candidate detection ───────────────────────────────────────────
        is_key_name = bool(_KEY_NAME_RE.search(str(col)))
        is_high_unique = unique_ratio >= KEY_UNIQUE_THRESHOLD
        is_low_missing = missing_ratio <= KEY_MISSING_THRESHOLD

        if is_high_unique and is_low_missing and (is_key_name or unique_ratio == 1.0):
            key_candidates.append(col)

            # Check for duplicates in the candidate key
            dup_in_key = n - nunique
            if dup_in_key > 0:
                dup_pct = _pct(dup_in_key, n)
                issues.append(QualityIssue(
                    severity=SEV_HIGH,
                    category="validity",
                    column=col,
                    title=f"Duplicate values in key candidate '{col}'",
                    detail=(
                        f"Column '{col}' looks like a unique identifier "
                        f"but contains {dup_in_key:,} duplicate value(s) "
                        f"({dup_pct:.1f}%)."
                    ),
                    recommendation=(
                        f"Investigate duplicate values in '{col}'. "
                        "If this column is intended to be a primary key, "
                        "duplicates must be resolved before use."
                    ),
                ))
                penalty += 10.0

    return _clamp(100.0 - min(penalty, 40.0))


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_data_quality(df: pd.DataFrame) -> QualityReport:
    """
    Analyze a DataFrame and return a full QualityReport.

    Args:
        df: The DataFrame to analyze.  It is never modified.

    Returns:
        QualityReport with score (0–100), category scores, issues,
        candidates, and summary statistics.

    Performance:
        Uses pandas vectorized operations throughout.
        Per-column sampling is capped (default 200 values) to keep
        the analysis fast even for datasets approaching MAX_ROWS.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    rows, cols = df.shape

    # Collect issues and candidates across all scorers.
    issues:                 list[QualityIssue] = []
    key_candidates:         list[str]          = []
    categorical_candidates: list[str]          = []

    # ── Handle edge cases ─────────────────────────────────────────────────────
    if rows == 0 or cols == 0:
        return QualityReport(
            score=100,
            categories={
                "completeness": 100,
                "uniqueness":   100,
                "consistency":  100,
                "validity":     100,
            },
            issues=[],
            key_candidates=[],
            categorical_candidates=[],
            stats={
                "rows":          rows,
                "cols":          cols,
                "missing_cells": 0,
                "missing_pct":   0.0,
                "duplicate_rows": 0,
                "duplicate_pct": 0.0,
            },
        )

    # ── Run category scorers ──────────────────────────────────────────────────
    completeness = _score_completeness(df, issues)
    uniqueness   = _score_uniqueness(df, issues)
    consistency  = _score_consistency(df, issues, categorical_candidates)
    validity     = _score_validity(df, issues, key_candidates)

    # ── Overall score (weighted average) ──────────────────────────────────────
    # Weights reflect typical data-quality priorities:
    #   completeness 35 %, uniqueness 25 %, consistency 20 %, validity 20 %
    overall = _clamp(
        completeness * 0.35
        + uniqueness   * 0.25
        + consistency  * 0.20
        + validity     * 0.20
    )

    # ── Summary statistics ────────────────────────────────────────────────────
    total_cells   = rows * cols
    missing_cells = int(df.isnull().sum().sum())
    dup_rows      = int(df.duplicated().sum())

    stats: dict[str, Any] = {
        "rows":          rows,
        "cols":          cols,
        "missing_cells": missing_cells,
        "missing_pct":   round(_pct(missing_cells, total_cells), 2),
        "duplicate_rows": dup_rows,
        "duplicate_pct": round(_pct(dup_rows, rows), 2),
    }

    # ── Sort issues by severity ───────────────────────────────────────────────
    issues.sort(key=_severity_order)

    # ── De-duplicate key / categorical candidates ─────────────────────────────
    seen_k: set[str] = set()
    unique_keys = [c for c in key_candidates if not (c in seen_k or seen_k.add(c))]  # type: ignore[func-returns-value]

    seen_c: set[str] = set()
    unique_cats = [c for c in categorical_candidates if not (c in seen_c or seen_c.add(c))]  # type: ignore[func-returns-value]

    return QualityReport(
        score=overall,
        categories={
            "completeness": completeness,
            "uniqueness":   uniqueness,
            "consistency":  consistency,
            "validity":     validity,
        },
        issues=issues,
        key_candidates=unique_keys,
        categorical_candidates=unique_cats,
        stats=stats,
    )


def score_label(score: int) -> str:
    """Return a human-readable label for a quality score."""
    if score >= SCORE_EXCELLENT:
        return "Excellent"
    if score >= SCORE_GOOD:
        return "Good"
    if score >= SCORE_ATTENTION:
        return "Needs Attention"
    return "Poor"


def score_emoji(score: int) -> str:
    """Return an emoji indicator for a quality score."""
    if score >= SCORE_EXCELLENT:
        return "🟢"
    if score >= SCORE_GOOD:
        return "🟡"
    if score >= SCORE_ATTENTION:
        return "🟠"
    return "🔴"
