"""
cleaning.py — Core Cleaning Engine for DataCleaner Pro V3.

All cleaning functions are pure (input DataFrame → output DataFrame).
No Streamlit imports anywhere in this module.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Callable

import pandas as pd

from utils.helpers import to_snake_case


# ── Encoding Repair ───────────────────────────────────────────────────────────

def _fix_encoding_ftfy(val):
    """
    Repair mojibake using ftfy.

    Returns the fixed string only when it actually differs from the original.
    Returns the original value unchanged when:
      - ftfy makes no difference
      - ftfy is not installed
      - val is NaN / None
    """
    if pd.isna(val):
        return val
    original = str(val)
    try:
        import ftfy  # optional dependency
        fixed = ftfy.fix_text(original)
        return fixed if fixed != original else val
    except Exception:
        return val


def _count_encoding_repairs(
    original_series: pd.Series,
    fixed_series:    pd.Series,
) -> int:
    """
    Count the number of cells that were actually changed by encoding repair.

    Rules:
    - NaN cells on either side are never counted.
    - Only increments when str(original) != str(fixed).
    """
    count = 0
    for orig, fixed in zip(original_series, fixed_series):
        orig_na  = orig  is None or (isinstance(orig,  float) and pd.isna(orig))
        fixed_na = fixed is None or (isinstance(fixed, float) and pd.isna(fixed))
        if orig_na or fixed_na:
            continue
        if str(orig) != str(fixed):
            count += 1
    return count


# ── Whitespace Trimming ───────────────────────────────────────────────────────

def _trim_spaces(val):
    """
    Collapse multiple consecutive spaces and strip leading/trailing whitespace.
    Returns pd.NA for empty / null-like strings.
    """
    if pd.isna(val):
        return val
    try:
        result = re.sub(r" {2,}", " ", str(val)).strip()
        return pd.NA if result in ("", "nan", "None") else result
    except Exception:
        return val


# ── Email Cleaning ────────────────────────────────────────────────────────────

_VALID_EMAIL = re.compile(r"^[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}$")


def normalize_email(val):
    """
    Lowercase and strip a valid email address.
    Returns the original value unchanged if the address does not look valid.
    """
    if pd.isna(val):
        return val
    cleaned = str(val).strip().lower()
    return cleaned if _VALID_EMAIL.match(cleaned) else val


def get_invalid_emails(series: pd.Series) -> pd.Series:
    """Return a boolean mask — True where the email value appears invalid."""
    def _is_invalid(val) -> bool:
        if pd.isna(val):
            return False
        return not bool(_VALID_EMAIL.match(str(val).strip().lower()))
    return series.apply(_is_invalid)


# ── Phone Cleaning ────────────────────────────────────────────────────────────

def normalize_phone(val):
    """
    Normalize a phone number by removing non-digit characters while
    preserving a leading '+' (country code indicator).

    Does NOT assume or inject any country code.
    Returns the original value if no digits are found.
    """
    if pd.isna(val):
        return val
    s        = str(val).strip()
    has_plus = s.startswith("+")
    digits   = re.sub(r"[^\d]", "", s)
    if not digits:
        return val
    return ("+" if has_plus else "") + digits


# ── Date Normalization ────────────────────────────────────────────────────────

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%b %d %Y",
    "%B %d %Y",
]


def normalize_date(val, target_fmt: str = "%Y-%m-%d"):
    """
    Parse a date string using common formats and return it in target_fmt.
    Returns the original value unchanged if no format matches.
    """
    if pd.isna(val):
        return val
    s = str(val).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime(target_fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(s, infer_datetime_format=True).strftime(target_fmt)
    except Exception:
        return val


# ── Missing Value Handling ────────────────────────────────────────────────────

def fill_missing(
    df:       pd.DataFrame,
    strategy: str,
) -> tuple[pd.DataFrame, int]:
    """
    Fill or drop missing values according to the chosen strategy.

    Strategies
    ----------
    "Auto (Median/Mode)"
        Numeric columns → median; object/string columns → mode.
    "Fill with 0"
        Replace all NaN with scalar 0.
    "Fill with 'Unknown'"
        Replace all NaN with the string "Unknown".
    "Drop rows with missing values"
        Drop every row that contains at least one NaN.

    Returns
    -------
    (modified_df, cells_or_rows_affected)
        For drop strategy: rows dropped.
        For fill strategies: cells filled (missing_before − missing_after).
    """
    missing_before = int(df.isnull().sum().sum())

    if missing_before == 0:
        return df, 0

    if strategy == "Drop rows with missing values":
        rows_before = len(df)
        df          = df.dropna()
        return df, rows_before - len(df)

    if strategy == "Fill with 0":
        df = df.fillna(0)
        return df, missing_before

    if strategy == "Fill with 'Unknown'":
        df = df.fillna("Unknown")
        return df, missing_before

    # ── Auto (Median/Mode) ────────────────────────────────────────────────────
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median()
            # If entire column is NaN, median is NaN — fill with 0 to preserve dtype
            df[col] = df[col].fillna(0 if pd.isna(med) else med)
        else:
            mode_s = df[col].mode()
            fill_v = mode_s.iloc[0] if not mode_s.empty else "Unknown"
            df[col] = df[col].fillna(fill_v)

    missing_after = int(df.isnull().sum().sum())
    return df, missing_before - missing_after


# ── Column Operations ─────────────────────────────────────────────────────────

def remove_empty_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop columns that contain only NaN values."""
    before = len(df.columns)
    df     = df.dropna(axis=1, how="all")
    return df, before - len(df.columns)


def remove_duplicate_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove columns whose content is identical to an earlier column.
    Uses a fast tuple-hash — much faster than to_json() on large DataFrames.
    The first occurrence of each unique column is kept.
    """
    before = len(df.columns)
    seen:  set[int]  = set()
    keep:  list[str] = []
    for col in df.columns:
        try:
            h = hash(tuple(df[col].values))
        except Exception:
            # Unhashable types — always keep
            keep.append(col)
            continue
        if h not in seen:
            seen.add(h)
            keep.append(col)
    df = df[keep]
    return df, before - len(df.columns)


def remove_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop columns where all non-null values are the same (zero variance)."""
    before     = len(df.columns)
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    df         = df.drop(columns=const_cols)
    return df, before - len(df.columns)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_cleaning_pipeline(
    df:          pd.DataFrame,
    options:     dict,
    progress_cb: Callable[[float, str], None] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Execute the full cleaning pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Copied internally — original is never modified.
    options : dict
        Cleaning options merged with safe defaults (see ``_DEFAULTS`` below).
    progress_cb : callable, optional
        Called as ``progress_cb(fraction: float, message: str)`` after each step.

    Returns
    -------
    (cleaned_df, report_dict)
        report_dict contains one key per cleaning action plus before/after stats.
    """
    df = df.copy()

    # ── Defaults — every key guaranteed present ───────────────────────────────
    opt: dict = {
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

    # ── Initial snapshot for accurate reporting ───────────────────────────────
    _initial_missing = int(df.isnull().sum().sum())

    report: dict = {
        # ── Timestamps & identity ─────────────────────────────────────────────
        "timestamp":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fill_strategy_used":   opt["fill_strategy"],
        # ── Before stats ─────────────────────────────────────────────────────
        "rows_before":          len(df),
        "cols_before":          len(df.columns),
        "missing_count_before": _initial_missing,   # ← NEW: actual initial count
        # ── After stats (filled at end) ───────────────────────────────────────
        "rows_after":           0,
        "cols_after":           0,
        "missing_count_after":  0,                  # ← NEW: filled at end
        # ── Action counters ───────────────────────────────────────────────────
        "empty_rows_removed":   0,
        "duplicates_removed":   0,
        "empty_cols_removed":   0,
        "dup_cols_removed":     0,
        "const_cols_removed":   0,
        "missing_filled":       0,
        "missing_dropped_rows": 0,
        "headers_stripped":     0,
        "headers_snake_cased":  0,
        "encoding_repaired":    0,
        "spaces_trimmed":       0,
        "emails_normalized":    0,
        "phones_normalized":    0,
        "dates_normalized":     0,
    }

    steps        = 11
    current_step = [0]

    def _progress(msg: str) -> None:
        current_step[0] += 1
        if progress_cb:
            progress_cb(min(current_step[0] / steps, 1.0), msg)

    # ── Step 1: Strip header whitespace ──────────────────────────────────────
    orig_cols    = list(df.columns)
    df.columns   = [str(c).strip() for c in df.columns]
    report["headers_stripped"] = sum(
        1 for a, b in zip(df.columns, orig_cols) if a != str(b).strip()
    )
    _progress("Stripping column header whitespace…")

    # ── Step 2: Remove empty rows ─────────────────────────────────────────────
    if opt["remove_empty_rows"]:
        _before                      = len(df)
        df                           = df.dropna(how="all")
        report["empty_rows_removed"] = _before - len(df)
    _progress("Removing empty rows…")

    # ── Step 3: Remove empty columns ──────────────────────────────────────────
    if opt["remove_empty_cols"]:
        df, n                        = remove_empty_columns(df)
        report["empty_cols_removed"] = n
    _progress("Removing empty columns…")

    # ── Step 4: Remove duplicate columns ──────────────────────────────────────
    if opt["remove_dup_cols"]:
        df, n                      = remove_duplicate_columns(df)
        report["dup_cols_removed"] = n
    _progress("Removing duplicate columns…")

    # ── Step 5: Remove constant columns ───────────────────────────────────────
    if opt["remove_const_cols"]:
        df, n                       = remove_constant_columns(df)
        report["const_cols_removed"] = n
    _progress("Removing constant columns…")

    # ── Step 6: Remove exact duplicate rows ───────────────────────────────────
    _before                      = len(df)
    df                           = df.drop_duplicates(keep="first")
    report["duplicates_removed"] = _before - len(df)
    _progress("Removing duplicate rows…")

    # ── Step 7: Repair encoding with ftfy ─────────────────────────────────────
    if opt["use_ftfy"]:
        _total_repaired = 0
        _str_cols       = df.select_dtypes(include="object").columns
        for col in _str_cols:
            _orig_col       = df[col].copy()
            df[col]         = df[col].apply(_fix_encoding_ftfy)
            _total_repaired += _count_encoding_repairs(_orig_col, df[col])
        report["encoding_repaired"] = _total_repaired
    _progress("Repairing text encoding…")

    # ── Step 8: Trim extra whitespace ─────────────────────────────────────────
    if opt["trim_spaces"]:
        _trimmed  = 0
        _str_cols = df.select_dtypes(include="object").columns
        for col in _str_cols:
            _orig_col = df[col].copy()
            df[col]   = df[col].apply(_trim_spaces)
            for _o, _f in zip(_orig_col, df[col]):
                _o_na = _o is None or (isinstance(_o, float) and pd.isna(_o))
                _f_na = _f is None or (isinstance(_f, float) and pd.isna(_f))
                if not _o_na and not _f_na and str(_o) != str(_f):
                    _trimmed += 1
        report["spaces_trimmed"] = _trimmed
    _progress("Trimming whitespace…")

    # ── Step 9: Normalize emails / phones / dates ──────────────────────────────
    if opt["normalize_emails"]:
        for col in opt["email_columns"]:
            if col in df.columns:
                _bc      = df[col].copy()
                df[col]  = df[col].apply(normalize_email)
                report["emails_normalized"] += sum(
                    1 for o, f in zip(_bc, df[col]) if str(o) != str(f)
                )

    if opt["normalize_phones"]:
        for col in opt["phone_columns"]:
            if col in df.columns:
                _bc      = df[col].copy()
                df[col]  = df[col].apply(normalize_phone)
                report["phones_normalized"] += sum(
                    1 for o, f in zip(_bc, df[col]) if str(o) != str(f)
                )

    if opt["normalize_dates"]:
        for col in opt["date_columns"]:
            if col in df.columns:
                _bc      = df[col].copy()
                df[col]  = df[col].apply(
                    lambda v: normalize_date(v, opt["date_target_fmt"])
                )
                report["dates_normalized"] += sum(
                    1 for o, f in zip(_bc, df[col]) if str(o) != str(f)
                )
    _progress("Normalizing email / phone / date columns…")

    # ── Step 10: Fill / drop missing values ───────────────────────────────────
    _missing_before_fill = int(df.isnull().sum().sum())
    df, _affected        = fill_missing(df, opt["fill_strategy"])
    _missing_after_fill  = int(df.isnull().sum().sum())

    if opt["fill_strategy"] == "Drop rows with missing values":
        report["missing_dropped_rows"] = _affected
        report["missing_filled"]       = 0
    else:
        report["missing_filled"]       = max(0, _missing_before_fill - _missing_after_fill)
        report["missing_dropped_rows"] = 0
    _progress("Handling missing values…")

    # ── Step 11: snake_case headers ───────────────────────────────────────────
    if opt["snake_case"]:
        _orig_cols2             = list(df.columns)
        df.columns              = [to_snake_case(c) for c in df.columns]
        report["headers_snake_cased"] = sum(
            1 for a, b in zip(df.columns, _orig_cols2) if a != b
        )
    _progress("Finalizing…")

    # ── Final stats ───────────────────────────────────────────────────────────
    report["rows_after"]          = len(df)
    report["cols_after"]          = len(df.columns)
    report["missing_count_after"] = int(df.isnull().sum().sum())

    return df, report
