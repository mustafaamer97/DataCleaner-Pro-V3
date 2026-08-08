"""
profiling.py — Smart Data Profiling Engine for DataCleaner Pro V3.

Detects column semantic types using BOTH column name heuristics
and content sampling. Date detection always runs before phone
detection to prevent date strings being misclassified as phone numbers.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Column Name Keyword Maps
#  These are checked BEFORE content sampling.
#  A strong name match short-circuits the content analysis.
# ─────────────────────────────────────────────────────────────────────────────

_NAME_EMAIL_KW = re.compile(
    r"\b(email|e[-_]?mail|mail)\b", re.IGNORECASE
)
_NAME_PHONE_KW = re.compile(
    r"\b(phone|mobile|cell|tel|fax|contact[-_]?no|phone[-_]?no)\b",
    re.IGNORECASE,
)
_NAME_DATE_KW = re.compile(
    r"\b(date|datetime|timestamp|time|created|updated|modified|"
    r"signup|birth|dob|joined|registered|at|on)\b",
    re.IGNORECASE,
)
_NAME_ID_KW = re.compile(
    r"(^id$|[-_]id$|^id[-_]|[-_]?uuid|[-_]?guid|[-_]?key$)",
    re.IGNORECASE,
)
_NAME_URL_KW = re.compile(
    r"\b(url|link|website|site|href|uri)\b", re.IGNORECASE
)
_NAME_CURRENCY_KW = re.compile(
    r"\b(price|cost|amount|salary|wage|revenue|fee|charge|"
    r"payment|budget|total|subtotal)\b",
    re.IGNORECASE,
)
_NAME_PCT_KW = re.compile(
    r"\b(percent|pct|rate|ratio|share|proportion)\b", re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
#  Content Regex Patterns
# ─────────────────────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(
    r"^[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}$"
)

# Phone: must start with optional +, then digits with limited separators.
# Crucially does NOT match date separators like 2024-01-15 or Feb 3 2024.
_PHONE_RE = re.compile(
    r"^\+?[\d]{1,4}?[-.\s]?"          # optional country code
    r"\(?\d{2,4}\)?"                   # optional area code with parens
    r"[-.\s]?\d{2,4}"                  # main number block
    r"[-.\s]?\d{2,4}"                  # second block
    r"[-.\s]?\d{0,4}$"                 # optional trailing block
)

_URL_RE = re.compile(
    r"^(https?://|www\.)", re.IGNORECASE
)

# Strict currency: optional symbol, then digits (no pure date-like strings)
_CURRENCY_RE = re.compile(
    r"^[\$€£¥₹]?\s?\d[\d,]*(\.\d{1,4})?$"
)

_PCT_RE = re.compile(
    r"^\d+\.?\d*\s?%$"
)

# Date patterns — comprehensive list used for POSITIVE date identification
_DATE_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2})?"),           # 2024-01-15
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),                       # 15/01/2024
    re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$"),                         # 15-01-2024
    re.compile(
        r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[a-z]*[\s,]+\d{1,2}[,\s]+\d{2,4}$",
        re.IGNORECASE,
    ),                                                                 # Feb 3 2024
    re.compile(
        r"^\d{1,2}\s+"
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[a-z]*\s+\d{4}$",
        re.IGNORECASE,
    ),                                                                 # 3 Feb 2024
]


# ─────────────────────────────────────────────────────────────────────────────
#  Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_col(series: pd.Series, n: int = 200) -> pd.Series:
    """Return a non-null sample of a Series for pattern matching."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return non_null
    return (
        non_null.sample(min(n, len(non_null)), random_state=42)
        if len(non_null) > n
        else non_null
    )


def _match_ratio(series: pd.Series, pattern: re.Pattern, n: int = 200) -> float:
    """
    Return the fraction of sampled non-null string values matching a regex.
    Returns 0.0 if there are no non-null values.
    """
    sample = _sample_col(series, n).astype(str)
    if sample.empty:
        return 0.0
    return float(sample.str.fullmatch(pattern).mean())


def _date_content_ratio(series: pd.Series, n: int = 200) -> float:
    """
    Return the fraction of sampled non-null values that look like dates.
    Uses multiple date patterns and pandas parser as a fallback.
    """
    sample = _sample_col(series, n).astype(str)
    if sample.empty:
        return 0.0

    def _is_date(val: str) -> bool:
        for pat in _DATE_PATTERNS:
            if pat.match(val.strip()):
                return True
        # pandas fallback
        try:
            pd.to_datetime(val, infer_datetime_format=True)
            return True
        except Exception:
            return False

    return float(sample.apply(_is_date).mean())


def _phone_content_ratio(series: pd.Series, n: int = 200) -> float:
    """
    Return the fraction of sampled non-null values that look like phone numbers.
    Values that already look like dates are excluded from matching.
    """
    sample = _sample_col(series, n).astype(str)
    if sample.empty:
        return 0.0

    def _is_phone(val: str) -> bool:
        v = val.strip()
        # Exclude anything that looks like a date
        for pat in _DATE_PATTERNS:
            if pat.match(v):
                return False
        # Must not be pure numeric with > 8 digits (likely an ID or zip)
        digits_only = re.sub(r"[^\d]", "", v)
        if len(digits_only) > 15:
            return False
        if len(digits_only) < 7:
            return False
        return bool(_PHONE_RE.fullmatch(v))

    return float(sample.apply(_is_phone).mean())


# ─────────────────────────────────────────────────────────────────────────────
#  Main Type Detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_column_type(series: pd.Series, col_name: str = "") -> str:
    """
    Detect the semantic type of a column.

    Strategy (in priority order):
        1. Pandas dtype — if already datetime → 'date', if numeric → check ID
        2. Column name keywords — strong prior, checked before content
        3. Content sampling — regex matching on a sample of non-null values

    Date detection always runs before phone detection to prevent
    date strings (which contain digits and separators) being misclassified.

    Returns one of:
        'email', 'phone', 'url', 'date', 'currency', 'percentage',
        'id', 'numeric', 'categorical', 'text', 'unknown'
    """
    name = str(col_name).strip()

    # ── 1. Dtype shortcuts ─────────────────────────────
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    if pd.api.types.is_numeric_dtype(series):
        # Potential ID: integer column where every non-null value is unique
        if (
            pd.api.types.is_integer_dtype(series)
            and _NAME_ID_KW.search(name)
        ):
            return "id"
        if (
            pd.api.types.is_integer_dtype(series)
            and series.nunique() == len(series.dropna())
            and len(series.dropna()) > 0
        ):
            return "id"
        return "numeric"

    # From here the column is object / string dtype
    # ── 2a. Strong name → date ──────────────────────────
    if _NAME_DATE_KW.search(name):
        # Confirm with content (must look date-like ≥ 50%)
        if _date_content_ratio(series) >= 0.50:
            return "date"
        # Even without full content confirmation, date keyword wins over phone
        return "date"

    # ── 2b. Strong name → email ─────────────────────────
    if _NAME_EMAIL_KW.search(name):
        return "email"

    # ── 2c. Strong name → phone ─────────────────────────
    if _NAME_PHONE_KW.search(name):
        return "phone"

    # ── 2d. Strong name → URL ───────────────────────────
    if _NAME_URL_KW.search(name):
        return "url"

    # ── 2e. Strong name → currency ──────────────────────
    if _NAME_CURRENCY_KW.search(name):
        return "currency"

    # ── 2f. Strong name → percentage ────────────────────
    if _NAME_PCT_KW.search(name):
        return "percentage"

    # ── 2g. Strong name → id ───────────────────────────
    if _NAME_ID_KW.search(name):
        return "id"

    # ── 3. Content sampling (no strong name match) ──────
    # Date MUST be checked before phone
    if _date_content_ratio(series) >= 0.70:
        return "date"

    if _match_ratio(series, _EMAIL_RE) >= 0.70:
        return "email"

    if _phone_content_ratio(series) >= 0.60:
        return "phone"

    if _match_ratio(series, _URL_RE) >= 0.50:
        return "url"

    if _match_ratio(series, _CURRENCY_RE) >= 0.60:
        return "currency"

    if _match_ratio(series, _PCT_RE) >= 0.60:
        return "percentage"

    # ── 4. Cardinality-based fallback ───────────────────
    nunique = series.nunique()
    nrows   = len(series.dropna())
    if nrows == 0:
        return "unknown"

    ratio = nunique / nrows
    if ratio < 0.05 or nunique <= 20:
        return "categorical"
    if ratio > 0.90:
        return "text"
    return "categorical"


# ─────────────────────────────────────────────────────────────────────────────
#  Main Profiler
# ─────────────────────────────────────────────────────────────────────────────

def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run a full profile on a DataFrame.

    Returns a structured dict with metadata, per-column profiles,
    type group lists, warnings, and cleaning recommendations.
    """
    rows, cols = df.shape

    total_cells   = rows * cols
    missing_count = int(df.isnull().sum().sum())
    missing_pct   = round(missing_count / max(total_cells, 1) * 100, 2)
    dup_rows      = int(df.duplicated().sum())
    dup_row_pct   = round(dup_rows / max(rows, 1) * 100, 2)

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
        series      = df[col]
        col_missing = int(series.isnull().sum())
        col_miss_pct = round(col_missing / max(rows, 1) * 100, 2)
        nunique     = int(series.nunique())

        # Pass the column name to the detector
        sem_type = detect_column_type(series, col_name=col)

        col_profiles[col] = {
            "dtype":       str(series.dtype),
            "semantic":    sem_type,
            "missing":     col_missing,
            "missing_pct": col_miss_pct,
            "unique":      nunique,
            "sample":      _get_sample_values(series),
        }

        type_groups.setdefault(sem_type, []).append(col)

        if col_missing == rows:
            empty_cols.append(col)
        if nunique <= 1:
            constant_cols.append(col)

    # ── Duplicate columns ────────────────────────────────
    seen_hashes: dict[int, str] = {}
    for col in df.columns:
        try:
            h = hash(df[col].to_json())
        except Exception:
            continue
        if h in seen_hashes:
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
    warnings:        list[str] = []
    recommendations: list[str] = []

    if missing_pct > 0:
        warnings.append(
            f"{missing_pct}% missing values across the dataset"
        )
        recommendations.append("Review and fill or drop missing values")

    if dup_rows > 0:
        warnings.append(
            f"{dup_rows:,} exact duplicate rows detected ({dup_row_pct}%)"
        )
        recommendations.append("Remove exact duplicate rows")

    if constant_cols:
        warnings.append(
            f"{len(constant_cols)} constant column(s): {', '.join(constant_cols)}"
        )
        recommendations.append("Remove constant/zero-variance columns")

    if empty_cols:
        warnings.append(
            f"{len(empty_cols)} completely empty column(s): {', '.join(empty_cols)}"
        )
        recommendations.append("Remove empty columns")

    if dup_col_groups:
        warnings.append(
            f"{len(dup_col_groups)} duplicated column group(s) detected"
        )
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
    return [str(s)[:60] for s in non_null.head(n).tolist()]
