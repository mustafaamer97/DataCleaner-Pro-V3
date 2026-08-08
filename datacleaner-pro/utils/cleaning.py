"""
cleaning.py — Core Cleaning Engine for DataCleaner Pro V3.

All cleaning functions are pure (input DataFrame → output DataFrame).
No Streamlit imports in this module.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Callable

import pandas as pd

from utils.helpers import to_snake_case


# ── Encoding Repair ───────────────────────────────────────────────────────────

def _fix_encoding_ftfy(val) -> str | float:
    """Repair mojibake using ftfy. Returns original if ftfy unavailable."""
    if pd.isna(val):
        return val
    try:
        import ftfy
        return ftfy.fix_text(str(val))
    except Exception:
        return val


def _trim_spaces(val) -> str | float:
    """Collapse multiple spaces and strip leading/trailing whitespace."""
    if pd.isna(val):
        return val
    try:
        result = re.sub(r" {2,}", " ", str(val)).strip()
        return pd.NA if result in ("", "nan", "None") else result
    except Exception:
        return val


# ── Email Cleaning ────────────────────────────────────────────────────────────

_VALID_EMAIL = re.compile(r"^[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}$")


def normalize_email(val) -> str | float:
    """Lowercase and strip an email address."""
    if pd.isna(val):
        return val
    cleaned = str(val).strip().lower()
    return cleaned if _VALID_EMAIL.match(cleaned) else val


def get_invalid_emails(series: pd.Series) -> pd.Series:
    """Return a boolean mask — True where email appears invalid."""
    def _is_invalid(val) -> bool:
        if pd.isna(val):
            return False
        return not bool(_VALID_EMAIL.match(str(val).strip().lower()))
    return series.apply(_is_invalid)


# ── Phone Cleaning ────────────────────────────────────────────────────────────

def normalize_phone(val) -> str | float:
    """
    Normalize a phone number:
    - Remove parentheses, hyphens, extra spaces
    - Preserve country codes (leading +)
    - Do NOT assume any country code
    """
    if pd.isna(val):
        return val
    s = str(val).strip()
    has_plus = s.startswith("+")
    digits   = re.sub(r"[^\d]", "", s)
    if not digits:
        return val
    return ("+" if has_plus else "") + digits


# ── Date Normalization ────────────────────────────────────────────────────────

_DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
    "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y",
]


def normalize_date(val, target_fmt: str = "%Y-%m-%d") -> str | float:
    """
    Parse a date string and return it in target_fmt.
    Returns original value if parsing fails.
    """
    if pd.isna(val):
        return val
    s = str(val).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime(target_fmt)
        except ValueError:
            continue
    # Last resort: pandas parser
    try:
        return pd.to_datetime(s, infer_datetime_format=True).strftime(target_fmt)
    except Exception:
        return val


# ── Missing Value Handling ────────────────────────────────────────────────────

def fill_missing(
    df: pd.DataFrame,
    strategy: str,
) -> tuple[pd.DataFrame, int]:
    """
    Fill or drop missing values according to the chosen strategy.

    Strategies:
        "Auto (Median/Mode)"            — median for numeric, mode for categorical
        "Fill with 0"                   — fill all with 0
        "Fill with 'Unknown'"           — fill all with "Unknown"
        "Drop rows with missing values" — drop any row containing NaN

    Returns:
        (modified_df, cells_affected)
    """
    missing_before = int(df.isnull().sum().sum())

    if strategy == "Drop rows with missing values":
        df = df.dropna()
        return df, missing_before

    if strategy == "Fill with 0":
        df = df.fillna(0)
        return df, missing_before

    if strategy == "Fill with 'Unknown'":
        df = df.fillna("Unknown")
        return df, missing_before

    # Auto (Median/Mode)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median()
            df[col] = df[col].fillna(med)
        else:
            mode_s = df[col].mode()
            fill_v = mode_s.iloc[0] if not mode_s.empty else "Unknown"
            df[col] = df[col].fillna(fill_v)

    return df, missing_before


# ── Column Operations ─────────────────────────────────────────────────────────

def remove_empty_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop columns that are entirely NaN."""
    before = len(df.columns)
    df = df.dropna(axis=1, how="all")
    return df, before - len(df.columns)


def remove_duplicate_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove columns with identical content (keep the first occurrence)."""
    before      = len(df.columns)
    seen        : set[int] = set()
    keep        : list[str] = []
    for col in df.columns:
        try:
            h = hash(df[col].to_json())
        except Exception:
            keep.append(col)
            continue
        if h not in seen:
            seen.add(h)
            keep.append(col)
    df = df[keep]
    return df, before - len(df.columns)


def remove_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop columns where all non-null values are the same."""
    before       = len(df.columns)
    const_cols   = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    df           = df.drop(columns=const_cols)
    return df, before - len(df.columns)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_cleaning_pipeline(
    df: pd.DataFrame,
    options: dict,
    progress_cb: Callable[[float, str], None] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Execute the full cleaning pipeline.

    Args:
        df:          Input DataFrame (will be copied internally).
        options:     Dict of cleaning options (see defaults below).
        progress_cb: Optional callback(fraction, message) for progress reporting.

    Returns:
        (cleaned_df, report_dict)
    """
    df = df.copy()

    # ── Defaults ──────────────────────────────────────────
    opt = {
        "fill_strategy":     "Auto (Median/Mode)",
        "use_ftfy":          True,
        "remove_empty_cols": True,
        "remove_dup_cols":   True,
        "remove_const_cols": False,
        "snake_case":        False,
        "trim_spaces":       True,
        "remove_empty_rows": True,
        "normalize_emails":  False,
        "normalize_phones":  False,
        "normalize_dates":   False,
        "date_target_fmt":   "%Y-%m-%d",
        "email_columns":     [],
        "phone_columns":     [],
        "date_columns":      [],
        **options,
    }

    report = {
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows_before":        len(df),
        "cols_before":        len(df.columns),
        "rows_after":         0,
        "cols_after":         0,
        "empty_rows_removed": 0,
        "duplicates_removed": 0,
        "empty_cols_removed": 0,
        "dup_cols_removed":   0,
        "const_cols_removed": 0,
        "missing_filled":     0,
        "missing_dropped_rows": 0,
        "headers_stripped":   0,
        "headers_snake_cased": 0,
        "encoding_repaired":  0,
        "spaces_trimmed":     0,
        "emails_normalized":  0,
        "phones_normalized":  0,
        "dates_normalized":   0,
    }

    steps        = 11
    current_step = [0]

    def _progress(msg: str) -> None:
        current_step[0] += 1
        if progress_cb:
            progress_cb(current_step[0] / steps, msg)

    # ── 1. Strip header whitespace ───────────────────────
    orig_cols    = list(df.columns)
    df.columns   = [str(c).strip() for c in df.columns]
    report["headers_stripped"] = sum(
        1 for a, b in zip(df.columns, orig_cols) if a != str(b).strip()
    )
    _progress("Stripping column header whitespace…")

    # ── 2. Remove empty rows ─────────────────────────────
    if opt["remove_empty_rows"]:
        before                     = len(df)
        df                         = df.dropna(how="all")
        report["empty_rows_removed"] = before - len(df)
    _progress("Removing empty rows…")

    # ── 3. Remove empty columns ──────────────────────────
    if opt["remove_empty_cols"]:
        df, n                      = remove_empty_columns(df)
        report["empty_cols_removed"] = n
    _progress("Removing empty columns…")

    # ── 4. Remove duplicate columns ──────────────────────
    if opt["remove_dup_cols"]:
        df, n                    = remove_duplicate_columns(df)
        report["dup_cols_removed"] = n
    _progress("Removing duplicate columns…")

    # ── 5. Remove constant columns ───────────────────────
    if opt["remove_const_cols"]:
        df, n                     = remove_constant_columns(df)
        report["const_cols_removed"] = n
    _progress("Removing constant columns…")

    # ── 6. Remove exact duplicate rows ───────────────────
    before                       = len(df)
    df                           = df.drop_duplicates(keep="first")
    report["duplicates_removed"] = before - len(df)
    _progress("Removing duplicate rows…")

    # ── 7. Repair encoding with ftfy ─────────────────────
    if opt["use_ftfy"]:
        repaired  = 0
        str_cols  = df.select_dtypes(include="object").columns
        for col in str_cols:
            original   = df[col].copy()
            df[col]    = df[col].apply(_fix_encoding_ftfy)
            repaired  += int((df[col].astype(str) != original.astype(str)).sum())
        report["encoding_repaired"] = repaired
    _progress("Repairing text encoding…")

    # ── 8. Trim extra whitespace ─────────────────────────
    if opt["trim_spaces"]:
        trimmed  = 0
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            original  = df[col].copy()
            df[col]   = df[col].apply(_trim_spaces)
            trimmed  += int((df[col].astype(str) != original.astype(str)).sum())
        report["spaces_trimmed"] = trimmed
    _progress("Trimming whitespace…")

    # ── 9. Normalize emails / phones / dates ─────────────
    if opt["normalize_emails"]:
        for col in opt["email_columns"]:
            if col in df.columns:
                before_col        = df[col].copy()
                df[col]           = df[col].apply(normalize_email)
                report["emails_normalized"] += int(
                    (df[col].astype(str) != before_col.astype(str)).sum()
                )

    if opt["normalize_phones"]:
        for col in opt["phone_columns"]:
            if col in df.columns:
                before_col         = df[col].copy()
                df[col]            = df[col].apply(normalize_phone)
                report["phones_normalized"] += int(
                    (df[col].astype(str) != before_col.astype(str)).sum()
                )

    if opt["normalize_dates"]:
        for col in opt["date_columns"]:
            if col in df.columns:
                before_col        = df[col].copy()
                df[col]           = df[col].apply(
                    lambda v: normalize_date(v, opt["date_target_fmt"])
                )
                report["dates_normalized"] += int(
                    (df[col].astype(str) != before_col.astype(str)).sum()
                )
    _progress("Normalizing email / phone / date columns…")

    # ── 10. Fill / drop missing values ───────────────────
    missing_before = int(df.isnull().sum().sum())
    df, _          = fill_missing(df, opt["fill_strategy"])
    missing_after  = int(df.isnull().sum().sum())

    if opt["fill_strategy"] == "Drop rows with missing values":
        report["missing_dropped_rows"] = missing_before  # approximate
    else:
        report["missing_filled"] = missing_before - missing_after
    _progress("Handling missing values…")

    # ── 11. snake_case headers ───────────────────────────
    if opt["snake_case"]:
        orig_cols2               = list(df.columns)
        df.columns               = [to_snake_case(c) for c in df.columns]
        report["headers_snake_cased"] = sum(
            1 for a, b in zip(df.columns, orig_cols2) if a != b
        )
    _progress("Finalizing…")

    report["rows_after"] = len(df)
    report["cols_after"] = len(df.columns)
    return df, report
