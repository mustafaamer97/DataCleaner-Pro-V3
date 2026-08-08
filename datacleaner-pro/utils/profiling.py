"""
profiling.py — Smart Data Profiling Engine for DataCleaner Pro V3.

Detects column semantic types using BOTH column name heuristics
and content sampling. Date detection always runs before phone
detection to prevent date strings being misclassified as phone numbers.

Column name keyword matching is the PRIMARY signal.
A column named 'signup_date' is ALWAYS classified as 'date' — period.
Content sampling is only used when no keyword match exists.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Column Name Keyword Patterns
#  Checked FIRST — before any content sampling.
#  A matching keyword produces an immediate classification with no ambiguity.
# ─────────────────────────────────────────────────────────────────────────────

# Matches: date, signup_date, birth_date, created_at, updated_at,
#          timestamp, dob, registered, joined, modified, etc.
_NAME_DATE_KW = re.compile(
    r"(^date$"
    r"|[-_]date$"
    r"|^date[-_]"
    r"|datetime"
    r"|timestamp"
    r"|[-_]at$"
    r"|^created"
    r"|^updated"
    r"|^modified"
    r"|^signup"
    r"|^sign_up"
    r"|^registered"
    r"|^joined"
    r"|^birth"
    r"|^dob$"
    r"|^born"
    r"|[-_]dob$"
    r"|^expir"
    r"|[-_]time$"
    r"|^time[-_]"
    r")",
    re.IGNORECASE,
)

# Matches: email, e_mail, e-mail, mail_address, etc.
_NAME_EMAIL_KW = re.compile(
    r"(^email$|[-_]email$|^email[-_]|^e[-_]mail|^mail$)",
    re.IGNORECASE,
)

# Matches: phone, mobile, cell, tel, fax, phone_no, contact_no, etc.
# Does NOT match anything date-related.
_NAME_PHONE_KW = re.compile(
    r"(^phone$"
    r"|[-_]phone$"
    r"|^phone[-_]"
    r"|^mobile$"
    r"|[-_]mobile$"
    r"|^cell$"
    r"|[-_]cell$"
    r"|^tel$"
    r"|[-_]tel$"
    r"|^fax$"
    r"|contact[-_]?no"
    r"|phone[-_]?no"
    r"|^msisdn$"
    r")",
    re.IGNORECASE,
)

# Matches: id, user_id, customer_id, order_id, _id suffix, uuid, guid
_NAME_ID_KW = re.compile(
    r"(^id$"
    r"|[-_]id$"
    r"|^id[-_]"
    r"|uuid"
    r"|guid"
    r"|[-_]key$"
    r")",
    re.IGNORECASE,
)

# Matches: url, link, website, href, uri, site
_NAME_URL_KW = re.compile(
    r"(^url$|[-_]url$|^link$|[-_]link$|^website$|^site$|^href$|^uri$)",
    re.IGNORECASE,
)

# Matches: price, cost, salary, wage, amount, revenue, fee, charge, budget, total
_NAME_CURRENCY_KW = re.compile(
    r"(price|cost|salary|wage|revenue|fee|charge|"
    r"payment|budget|total|subtotal|amount)",
    re.IGNORECASE,
)

# Matches: percent, pct, rate, ratio
_NAME_PCT_KW = re.compile(
    r"(percent|pct|[-_]rate$|^rate[-_]|ratio|proportion)",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Content Regex Patterns (used ONLY when no name keyword matches)
# ─────────────────────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(
    r"^[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}$"
)

# Phone regex — used for content sampling only.
# Requires at least 7 digits, allows +, spaces, hyphens, dots, parens.
# Applied ONLY after confirming the value is NOT a date string.
_PHONE_CONTENT_RE = re.compile(
    r"^\+?[\d]{1,4}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,6}$"
)

_URL_RE = re.compile(
    r"^(https?://|www\.)",
    re.IGNORECASE,
)

_CURRENCY_RE = re.compile(
    r"^[\$€£¥₹]?\s?\d[\d,]*(\.\d{1,4})?$"
)

_PCT_RE = re.compile(
    r"^\d+\.?\d*\s?%$"
)

# ─────────────────────────────────────────────────────────────────────────────
#  Date Patterns — comprehensive list for content detection
#  All patterns use fullmatch via re.fullmatch or re.Pattern.fullmatch
# ─────────────────────────────────────────────────────────────────────────────

_DATE_FULLMATCH_PATTERNS: list[re.Pattern] = [
    # ISO 8601: 2024-01-15 or 2024-01-15 14:30
    re.compile(r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?"),
    # dd/mm/yyyy or mm/dd/yyyy
    re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}"),
    # dd-mm-yyyy
    re.compile(r"\d{1,2}-\d{1,2}-\d{4}"),
    # yyyy/mm/dd
    re.compile(r"\d{4}/\d{2}/\d{2}"),
    # Feb 3 2024 / Feb 3, 2024
    re.compile(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
        r"[\s,]+\d{1,2}[,\s]+\d{2,4}",
        re.IGNORECASE,
    ),
    # 3 Feb 2024
    re.compile(
        r"\d{1,2}\s+"
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
        r"\s+\d{4}",
        re.IGNORECASE,
    ),
    # Feb 2024 (month + year only)
    re.compile(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
        r"\s+\d{4}",
        re.IGNORECASE,
    ),
]


def _looks_like_date(val: str) -> bool:
    """
    Return True if the string value looks like a date.
    Uses fullmatch against every compiled date pattern,
    then falls back to pandas parser.
    """
    v = val.strip()
    for pat in _DATE_FULLMATCH_PATTERNS:
        if pat.fullmatch(v):
            return True
    # pandas fallback — strict=False allows flexible parsing
    try:
        pd.to_datetime(v, infer_datetime_format=True)
        # Extra guard: if it parses but is very short (e.g. "12") skip it
        if len(v) >= 6:
            return True
    except Exception:
        pass
    return False


def _looks_like_phone(val: str) -> bool:
    """
    Return True if the string looks like a phone number.

    Explicit exclusions:
    - Anything that looks like a date is NOT a phone.
    - Pure integers with fewer than 7 or more than 15 digits are NOT phones.
    """
    v = val.strip()

    # Exclude dates first — highest priority
    if _looks_like_date(v):
        return False

    digits_only = re.sub(r"[^\d]", "", v)
    if len(digits_only) < 7 or len(digits_only) > 15:
        return False

    return bool(_PHONE_CONTENT_RE.fullmatch(v))


# ─────────────────────────────────────────────────────────────────────────────
#  Sampling Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_col(series: pd.Series, n: int = 200) -> pd.Series:
    """Return a non-null sample of a Series."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return non_null
    return (
        non_null.sample(min(n, len(non_null)), random_state=42)
        if len(non_null) > n
        else non_null
    )


def _content_ratio(series: pd.Series, test_fn, n: int = 200) -> float:
    """
    Return the fraction of sampled non-null values for which test_fn returns True.
    test_fn receives a plain Python str.
    """
    sample = _sample_col(series, n).astype(str)
    if sample.empty:
        return 0.0
    return float(sample.apply(test_fn).mean())


def _regex_ratio(series: pd.Series, pattern: re.Pattern, n: int = 200) -> float:
    """Return the fraction of sampled non-null values matching pattern (fullmatch)."""
    sample = _sample_col(series, n).astype(str)
    if sample.empty:
        return 0.0
    return float(sample.apply(lambda v: bool(pattern.fullmatch(v.strip()))).mean())


# ─────────────────────────────────────────────────────────────────────────────
#  Main Type Detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_column_type(series: pd.Series, col_name: str = "") -> str:
    """
    Detect the semantic type of a column.

    Decision order (strict priority):
        1.  Pandas dtype shortcuts  (datetime64 → 'date', numeric → 'numeric'/'id')
        2.  Column name → date      (UNCONDITIONAL — no content check required)
        3.  Column name → email
        4.  Column name → phone
        5.  Column name → url
        6.  Column name → currency
        7.  Column name → percentage
        8.  Column name → id
        9.  Content → date          (must be checked before phone)
        10. Content → email
        11. Content → phone         (only reached if content is NOT date-like)
        12. Content → url
        13. Content → currency
        14. Content → percentage
        15. Cardinality fallback    (categorical / text / unknown)

    RULE: A column whose NAME contains a date keyword is ALWAYS 'date'.
          It never proceeds to phone content detection.
    """
    name = str(col_name).strip()

    # ── 1. Dtype shortcuts ─────────────────────────────────────────────────
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    if pd.api.types.is_numeric_dtype(series):
        if _NAME_ID_KW.search(name):
            return "id"
        if (
            pd.api.types.is_integer_dtype(series)
            and series.nunique() == len(series.dropna())
            and len(series.dropna()) > 1
        ):
            return "id"
        return "numeric"

    # ── 2. Column name → date (UNCONDITIONAL) ─────────────────────────────
    # This rule fires before ANY content sampling.
    # signup_date, birth_date, created_at, updated_at, timestamp, dob, etc.
    # all exit here immediately — they never reach phone detection.
    if _NAME_DATE_KW.search(name):
        return "date"

    # ── 3. Column name → email ────────────────────────────────────────────
    if _NAME_EMAIL_KW.search(name):
        return "email"

    # ── 4. Column name → phone ────────────────────────────────────────────
    if _NAME_PHONE_KW.search(name):
        return "phone"

    # ── 5. Column name → url ──────────────────────────────────────────────
    if _NAME_URL_KW.search(name):
        return "url"

    # ── 6. Column name → currency ─────────────────────────────────────────
    if _NAME_CURRENCY_KW.search(name):
        return "currency"

    # ── 7. Column name → percentage ───────────────────────────────────────
    if _NAME_PCT_KW.search(name):
        return "percentage"

    # ── 8. Column name → id ───────────────────────────────────────────────
    if _NAME_ID_KW.search(name):
        return "id"

    # ── 9. Content → date (checked before phone) ──────────────────────────
    if _content_ratio(series, _looks_like_date) >= 0.70:
        return "date"

    # ── 10. Content → email ───────────────────────────────────────────────
    if _regex_ratio(series, _EMAIL_RE) >= 0.70:
        return "email"

    # ── 11. Content → phone ───────────────────────────────────────────────
    # _looks_like_phone already excludes date strings internally,
    # but we are also guaranteed dates were caught in steps 2 and 9.
    if _content_ratio(series, _looks_like_phone) >= 0.60:
        return "phone"

    # ── 12. Content → url ─────────────────────────────────────────────────
    if _regex_ratio(series, _URL_RE) >= 0.50:
        return "url"

    # ── 13. Content → currency ────────────────────────────────────────────
    if _regex_ratio(series, _CURRENCY_RE) >= 0.60:
        return "currency"

    # ── 14. Content → percentage ──────────────────────────────────────────
    if _regex_ratio(series, _PCT_RE) >= 0.60:
        return "percentage"

    # ── 15. Cardinality fallback ──────────────────────────────────────────
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
#  Sample Value Helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_sample_values(series: pd.Series, n: int = 3) -> list[str]:
    """Return up to n non-null sample values as strings (max 60 chars each)."""
    non_null = series.dropna()
    return [str(s)[:60] for s in non_null.head(n).tolist()]


# ─────────────────────────────────────────────────────────────────────────────
#  Main Profiler
# ─────────────────────────────────────────────────────────────────────────────

def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run a full profile on a DataFrame.

    Returns a structured dict containing:
        - Basic stats (rows, cols, missing, duplicates)
        - Per-column profiles (dtype, semantic type, missing count, samples)
        - Type groups (lists of column names per semantic type)
        - Warnings and cleaning recommendations
    """
    rows, cols = df.shape

    total_cells   = rows * cols
    missing_count = int(df.isnull().sum().sum())
    missing_pct   = round(missing_count / max(total_cells, 1) * 100, 2)
    dup_rows      = int(df.duplicated().sum())
    dup_row_pct   = round(dup_rows / max(rows, 1) * 100, 2)

    col_profiles: dict[str, dict] = {}

    # Initialise all known semantic type buckets
    type_groups: dict[str, list[str]] = {
        "numeric":    [],
        "categorical":[],
        "date":       [],
        "email":      [],
        "phone":      [],
        "url":        [],
        "currency":   [],
        "percentage": [],
        "id":         [],
        "text":       [],
        "unknown":    [],
    }

    constant_cols:  list[str]       = []
    empty_cols:     list[str]       = []
    dup_col_groups: list[list[str]] = []

    for col in df.columns:
        series       = df[col]
        col_missing  = int(series.isnull().sum())
        col_miss_pct = round(col_missing / max(rows, 1) * 100, 2)
        nunique      = int(series.nunique())

        # Always pass col_name so keyword rules fire correctly
        sem_type = detect_column_type(series, col_name=col)

        col_profiles[col] = {
            "dtype":       str(series.dtype),
            "semantic":    sem_type,
            "missing":     col_missing,
            "missing_pct": col_miss_pct,
            "unique":      nunique,
            "sample":      _get_sample_values(series),
        }

        # Bucket the column into its type group
        if sem_type in type_groups:
            type_groups[sem_type].append(col)
        else:
            type_groups.setdefault(sem_type, []).append(col)

        if col_missing == rows:
            empty_cols.append(col)
        if nunique <= 1:
            constant_cols.append(col)

    # ── Duplicate column detection ────────────────────────────────────────
    seen_hashes: dict[int, str] = {}
    for col in df.columns:
        try:
            h = hash(df[col].to_json())
        except Exception:
            continue
        if h in seen_hashes:
            placed = False
            for grp in dup_col_groups:
                if seen_hashes[h] in grp:
                    grp.append(col)
                    placed = True
                    break
            if not placed:
                dup_col_groups.append([seen_hashes[h], col])
        else:
            seen_hashes[h] = col

    # ── Warnings ──────────────────────────────────────────────────────────
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
