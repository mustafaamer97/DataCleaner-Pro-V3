"""
profiling.py — Smart Data Profiling Engine for DataCleaner Pro V3.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


_NAME_DATE_KW = re.compile(
    r"(^date$|[-_]date$|^date[-_]|datetime|timestamp|[-_]at$|^created"
    r"|^updated|^modified|^signup|^sign_up|^registered|^joined|^birth"
    r"|^dob$|^born|[-_]dob$|^expir|[-_]time$|^time[-_])",
    re.IGNORECASE,
)
_NAME_EMAIL_KW = re.compile(
    r"(^email$|[-_]email$|^email[-_]|^e[-_]mail|^mail$)",
    re.IGNORECASE,
)
_NAME_PHONE_KW = re.compile(
    r"(^phone$|[-_]phone$|^phone[-_]|^mobile$|[-_]mobile$|^cell$"
    r"|[-_]cell$|^tel$|[-_]tel$|^fax$|contact[-_]?no|phone[-_]?no|^msisdn$)",
    re.IGNORECASE,
)
_NAME_ID_KW = re.compile(
    r"(^id$|[-_]id$|^id[-_]|uuid|guid|[-_]key$)",
    re.IGNORECASE,
)
_NAME_URL_KW = re.compile(
    r"(^url$|[-_]url$|^link$|[-_]link$|^website$|^site$|^href$|^uri$)",
    re.IGNORECASE,
)
_NAME_CURRENCY_KW = re.compile(
    r"(price|cost|salary|wage|revenue|fee|charge|payment|budget|total|subtotal|amount)",
    re.IGNORECASE,
)
_NAME_PCT_KW = re.compile(
    r"(percent|pct|[-_]rate$|^rate[-_]|ratio|proportion)",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r"^[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}$")
_PHONE_CONTENT_RE = re.compile(
    r"^\+?[\d]{1,4}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,6}$"
)
_URL_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"^[\$€£¥₹]?\s?\d[\d,]*(\.\d{1,4})?$")
_PCT_RE = re.compile(r"^\d+\.?\d*\s?%$")

_DATE_FULLMATCH_PATTERNS: list[re.Pattern] = [
    re.compile(r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?"),
    re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}"),
    re.compile(r"\d{1,2}-\d{1,2}-\d{4}"),
    re.compile(r"\d{4}/\d{2}/\d{2}"),
    re.compile(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
        r"[\s,]+\d{1,2}[,\s]+\d{2,4}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[a-z]*\s+\d{4}",
        re.IGNORECASE,
    ),
    re.compile(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}",
        re.IGNORECASE,
    ),
]


def _looks_like_date(val: str) -> bool:
    value = val.strip()
    for pattern in _DATE_FULLMATCH_PATTERNS:
        if pattern.fullmatch(value):
            return True
    try:
        pd.to_datetime(value, infer_datetime_format=True)
        return len(value) >= 6
    except Exception:
        return False


def _looks_like_phone(val: str) -> bool:
    value = val.strip()
    if _looks_like_date(value):
        return False
    digits_only = re.sub(r"[^\d]", "", value)
    if len(digits_only) < 7 or len(digits_only) > 15:
        return False
    return bool(_PHONE_CONTENT_RE.fullmatch(value))


def _sample_col(series: pd.Series, n: int = 200) -> pd.Series:
    non_null = series.dropna()
    if len(non_null) == 0:
        return non_null
    if len(non_null) > n:
        return non_null.sample(n, random_state=42)
    return non_null


def _content_ratio(series: pd.Series, test_fn, n: int = 200) -> float:
    sample = _sample_col(series, n).astype(str)
    if sample.empty:
        return 0.0
    return float(sample.apply(test_fn).mean())


def _regex_ratio(series: pd.Series, pattern: re.Pattern, n: int = 200) -> float:
    sample = _sample_col(series, n).astype(str)
    if sample.empty:
        return 0.0
    return float(
        sample.apply(lambda value: bool(pattern.fullmatch(value.strip()))).mean()
    )


def detect_column_type(series: pd.Series, col_name: str = "") -> str:
    """
    Detect the semantic type of a column.

    Explicit identifier names are classified as IDs. Numeric columns without
    an identifier name are classified as numeric, including unique numeric
    columns such as age and salary.
    """
    name = str(col_name).strip()

    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    if pd.api.types.is_numeric_dtype(series):
        if _NAME_ID_KW.search(name):
            return "id"
        return "numeric"

    if _NAME_DATE_KW.search(name):
        return "date"
    if _NAME_EMAIL_KW.search(name):
        return "email"
    if _NAME_PHONE_KW.search(name):
        return "phone"
    if _NAME_URL_KW.search(name):
        return "url"
    if _NAME_CURRENCY_KW.search(name):
        return "currency"
    if _NAME_PCT_KW.search(name):
        return "percentage"
    if _NAME_ID_KW.search(name):
        return "id"

    if _content_ratio(series, _looks_like_date) >= 0.70:
        return "date"
    if _regex_ratio(series, _EMAIL_RE) >= 0.70:
        return "email"
    if _content_ratio(series, _looks_like_phone) >= 0.60:
        return "phone"
    if _regex_ratio(series, _URL_RE) >= 0.50:
        return "url"
    if _regex_ratio(series, _CURRENCY_RE) >= 0.60:
        return "currency"
    if _regex_ratio(series, _PCT_RE) >= 0.60:
        return "percentage"

    nunique = series.nunique()
    nrows = len(series.dropna())

    if nrows == 0:
        return "unknown"

    ratio = nunique / nrows

    if ratio < 0.05 or nunique <= 20:
        return "categorical"
    if ratio > 0.90:
        return "text"
    return "categorical"


def _get_sample_values(series: pd.Series, n: int = 3) -> list[str]:
    non_null = series.dropna()
    return [str(value)[:60] for value in non_null.head(n).tolist()]


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    rows, cols = df.shape

    total_cells = rows * cols
    missing_count = int(df.isnull().sum().sum())
    missing_pct = round(missing_count / max(total_cells, 1) * 100, 2)
    dup_rows = int(df.duplicated().sum())
    dup_row_pct = round(dup_rows / max(rows, 1) * 100, 2)

    col_profiles: dict[str, dict] = {}
    type_groups: dict[str, list[str]] = {
        "numeric": [],
        "categorical": [],
        "date": [],
        "email": [],
        "phone": [],
        "url": [],
        "currency": [],
        "percentage": [],
        "id": [],
        "text": [],
        "unknown": [],
    }

    constant_cols: list[str] = []
    empty_cols: list[str] = []
    dup_col_groups: list[list[str]] = []

    for col in df.columns:
        series = df[col]
        col_missing = int(series.isnull().sum())
        col_miss_pct = round(col_missing / max(rows, 1) * 100, 2)
        nunique = int(series.nunique())
        sem_type = detect_column_type(series, col_name=col)

        col_profiles[col] = {
            "dtype": str(series.dtype),
            "semantic": sem_type,
            "missing": col_missing,
            "missing_pct": col_miss_pct,
            "unique": nunique,
            "sample": _get_sample_values(series),
        }

        type_groups.setdefault(sem_type, []).append(col)

        if col_missing == rows:
            empty_cols.append(col)
        if nunique <= 1:
            constant_cols.append(col)

    seen_hashes: dict[int, str] = {}

    for col in df.columns:
        try:
            value_hash = hash(df[col].to_json())
        except Exception:
            continue

        if value_hash in seen_hashes:
            placed = False
            for group in dup_col_groups:
                if seen_hashes[value_hash] in group:
                    group.append(col)
                    placed = True
                    break
            if not placed:
                dup_col_groups.append([seen_hashes[value_hash], col])
        else:
            seen_hashes[value_hash] = col

    warnings: list[str] = []
    recommendations: list[str] = []

    if missing_pct > 0:
        warnings.append(f"{missing_pct}% missing values across the dataset")
        recommendations.append("Review and fill or drop missing values")

    if dup_rows > 0:
        warnings.append(
            f"{dup_rows:,} exact duplicate rows detected ({dup_row_pct}%)"
        )
        recommendations.append("Remove exact duplicate rows")

    if constant_cols:
        warnings.append(
            f"{len(constant_cols)} constant column(s): "
            f"{', '.join(constant_cols)}"
        )
        recommendations.append("Remove constant/zero-variance columns")

    if empty_cols:
        warnings.append(
            f"{len(empty_cols)} completely empty column(s): "
            f"{', '.join(empty_cols)}"
        )
        recommendations.append("Remove empty columns")

    if dup_col_groups:
        warnings.append(
            f"{len(dup_col_groups)} duplicated column group(s) detected"
        )
        recommendations.append("Remove duplicated columns")

    if type_groups["email"]:
        recommendations.append(
            f"Normalize email addresses in: "
            f"{', '.join(type_groups['email'])}"
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
        "rows": rows,
        "columns": cols,
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "duplicate_rows": dup_rows,
        "duplicate_row_pct": dup_row_pct,
        "constant_cols": constant_cols,
        "empty_cols": empty_cols,
        "dup_col_groups": dup_col_groups,
        "col_profiles": col_profiles,
        "type_groups": type_groups,
        "warnings": warnings,
        "recommendations": recommendations,
    }
