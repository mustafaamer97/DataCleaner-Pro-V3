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


# ═══════════════════════════════════════════════════════════════════════════════
# ENCODING REPAIR
# ═══════════════════════════════════════════════════════════════════════════════

def _fix_encoding_ftfy(val):
    """
    Repair mojibake using ftfy when available.

    Returns the original value when:
    - value is null
    - ftfy is unavailable
    - ftfy does not change the value
    """
    if pd.isna(val):
        return val

    original = str(val)

    try:
        import ftfy

        fixed = ftfy.fix_text(original)

        return fixed if fixed != original else val

    except Exception:
        return val


def _is_null(value) -> bool:
    """Safely determine whether a scalar value is null."""
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _count_changed_values(
    original_series: pd.Series,
    fixed_series: pd.Series,
) -> int:
    """Count non-null cells whose value actually changed."""
    count = 0

    for original, fixed in zip(original_series, fixed_series):
        if _is_null(original) and _is_null(fixed):
            continue

        if str(original) != str(fixed):
            count += 1

    return count


# ═══════════════════════════════════════════════════════════════════════════════
# WHITESPACE
# ═══════════════════════════════════════════════════════════════════════════════

def _trim_spaces(val):
    """
    Normalize whitespace inside text values.

    - Leading/trailing whitespace is removed.
    - Consecutive whitespace is collapsed.
    - Truly empty strings become pd.NA.
    - Literal text such as 'nan' or 'None' is preserved.
    """
    if pd.isna(val):
        return val

    try:
        result = re.sub(r"\s+", " ", str(val)).strip()

        if result == "":
            return pd.NA

        return result

    except Exception:
        return val


# ═══════════════════════════════════════════════════════════════════════════════
# EMAIL
# ═══════════════════════════════════════════════════════════════════════════════

_VALID_EMAIL = re.compile(
    r"^[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}$"
)


def normalize_email(val):
    """
    Lowercase and trim a valid email address.

    Invalid email-looking values are left unchanged so the cleaner
    does not silently destroy potentially useful data.
    """
    if pd.isna(val):
        return val

    cleaned = str(val).strip().lower()

    if _VALID_EMAIL.fullmatch(cleaned):
        return cleaned

    return val


def get_invalid_emails(series: pd.Series) -> pd.Series:
    """Return True for values that appear to be invalid email addresses."""

    def _is_invalid(val) -> bool:
        if pd.isna(val):
            return False

        return not bool(
            _VALID_EMAIL.fullmatch(
                str(val).strip().lower()
            )
        )

    return series.apply(_is_invalid)


# ═══════════════════════════════════════════════════════════════════════════════
# PHONE
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_phone(val):
    """
    Normalize a phone number.

    Non-digit characters are removed while preserving a leading '+'.
    No country code is invented or assumed.
    """
    if pd.isna(val):
        return val

    value = str(val).strip()

    has_plus = value.startswith("+")
    digits = re.sub(r"[^\d]", "", value)

    if not digits:
        return val

    return ("+" if has_plus else "") + digits


# ═══════════════════════════════════════════════════════════════════════════════
# DATE NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d %Y",
    "%b %d %Y",
]


def normalize_date(
    val,
    target_fmt: str = "%Y-%m-%d",
):
    """
    Parse common date representations and return target_fmt.

    Ambiguous formats are handled using the explicit formats above first.
    If those fail, pandas is used as a fallback.
    """
    if pd.isna(val):
        return val

    value = str(val).strip()

    if not value:
        return pd.NA

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).strftime(target_fmt)
        except ValueError:
            continue

    # Modern pandas supports format='mixed'.
    try:
        parsed = pd.to_datetime(
            value,
            format="mixed",
            errors="coerce",
        )

        if not pd.isna(parsed):
            return parsed.strftime(target_fmt)

    except (TypeError, ValueError):
        pass

    # Compatibility fallback for older pandas versions.
    try:
        parsed = pd.to_datetime(
            value,
            errors="coerce",
        )

        if not pd.isna(parsed):
            return parsed.strftime(target_fmt)

    except Exception:
        pass

    return val


# ═══════════════════════════════════════════════════════════════════════════════
# MISSING VALUES
# ═══════════════════════════════════════════════════════════════════════════════

def fill_missing(
    df: pd.DataFrame,
    strategy: str,
) -> tuple[pd.DataFrame, int]:
    """
    Fill or drop missing values.

    Strategies:
        Auto (Median/Mode)
        Fill with 0
        Fill with 'Unknown'
        Drop rows with missing values

    Returns:
        (modified_dataframe, affected_count)
    """
    missing_before = int(df.isnull().sum().sum())

    if missing_before == 0:
        return df, 0

    if strategy == "Drop rows with missing values":
        rows_before = len(df)

        df = df.dropna()

        return df, rows_before - len(df)

    if strategy == "Fill with 0":
        df = df.fillna(0)

        return df, missing_before

    if strategy == "Fill with 'Unknown'":
        df = df.fillna("Unknown")

        return df, missing_before

    # Default: Auto (Median/Mode)
    for column in df.columns:

        if not df[column].isnull().any():
            continue

        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            median = series.median()

            fill_value = 0 if pd.isna(median) else median

            df[column] = series.fillna(fill_value)

        else:
            mode = series.mode(dropna=True)

            fill_value = (
                mode.iloc[0]
                if not mode.empty
                else "Unknown"
            )

            df[column] = series.fillna(fill_value)

    missing_after = int(df.isnull().sum().sum())

    return df, missing_before - missing_after


# ═══════════════════════════════════════════════════════════════════════════════
# EMPTY ROWS / COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════

def remove_empty_rows(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Remove rows that contain no meaningful values.

    This works correctly after whitespace normalization because blank
    strings have already been converted to pd.NA.
    """
    before = len(df)

    df = df.dropna(how="all")

    return df, before - len(df)


def remove_empty_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Remove columns containing no meaningful values.
    """
    before = len(df.columns)

    df = df.dropna(axis=1, how="all")

    return df, before - len(df.columns)


# ═══════════════════════════════════════════════════════════════════════════════
# DUPLICATE COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════

def remove_duplicate_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Remove columns whose values are identical to an earlier column.

    Uses pandas' equality comparison instead of relying only on Python
    hash values, avoiding false duplicate detection caused by hash collisions.
    """
    before = len(df.columns)

    if len(df.columns) <= 1:
        return df, 0

    keep_positions: list[int] = []

    for position in range(len(df.columns)):

        current = df.iloc[:, position]

        duplicate = False

        for previous_position in keep_positions:
            previous = df.iloc[:, previous_position]

            try:
                if current.equals(previous):
                    duplicate = True
                    break

            except Exception:
                continue

        if not duplicate:
            keep_positions.append(position)

    df = df.iloc[:, keep_positions]

    return df, before - len(df.columns)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANT COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════

def remove_constant_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Remove columns where all non-null values are identical.

    Empty columns should already have been removed before this function.
    """
    before = len(df.columns)

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=True) <= 1
    ]

    df = df.drop(columns=constant_columns)

    return df, before - len(df.columns)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def _make_unique_columns(
    columns: list[str],
) -> list[str]:
    """
    Guarantee unique column names.

    Example:
        name, name, name
    becomes:
        name, name_2, name_3
    """
    used: dict[str, int] = {}
    result: list[str] = []

    for column in columns:
        base = str(column).strip()

        if not base:
            base = "column"

        count = used.get(base, 0) + 1
        used[base] = count

        if count == 1:
            result.append(base)
        else:
            result.append(f"{base}_{count}")

    return result


def normalize_headers(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Convert column names to snake_case and guarantee uniqueness.

    Returns:
        (modified_dataframe, number_of_changed_headers)
    """
    original = [str(column) for column in df.columns]

    normalized = [
        to_snake_case(column)
        for column in original
    ]

    normalized = _make_unique_columns(normalized)

    changed = sum(
        1
        for old, new in zip(original, normalized)
        if old != new
    )

    df = df.copy()
    df.columns = normalized

    return df, changed


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLEANING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_cleaning_pipeline(
    df: pd.DataFrame,
    options: dict,
    progress_cb: Callable[[float, str], None] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Execute the complete DataCleaner Pro cleaning pipeline.

    The input DataFrame is never modified directly.

    Returns:
        cleaned_df, report_dict
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    df = df.copy()

    # ═════════════════════════════════════════════════════════════════════════
    # SAFE DEFAULTS
    # ═════════════════════════════════════════════════════════════════════════

    opt: dict = {
        "fill_strategy": "Auto (Median/Mode)",
        "use_ftfy": True,
        "remove_empty_cols": True,
        "remove_dup_cols": True,
        "remove_const_cols": False,
        "snake_case": False,
        "trim_spaces": True,
        "remove_empty_rows": True,
        "normalize_emails": False,
        "normalize_phones": False,
        "normalize_dates": False,
        "date_target_fmt": "%Y-%m-%d",
        "email_columns": [],
        "phone_columns": [],
        "date_columns": [],
        **(options or {}),
    }

    # ═════════════════════════════════════════════════════════════════════════
    # INITIAL STATS
    # ═════════════════════════════════════════════════════════════════════════

    initial_missing = int(
        df.isnull().sum().sum()
    )

    report: dict = {
        # Identity
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "fill_strategy_used": opt["fill_strategy"],

        # Before
        "rows_before": len(df),
        "cols_before": len(df.columns),
        "missing_count_before": initial_missing,

        # After
        "rows_after": 0,
        "cols_after": 0,
        "missing_count_after": 0,

        # Actions
        "empty_rows_removed": 0,
        "duplicates_removed": 0,
        "empty_cols_removed": 0,
        "dup_cols_removed": 0,
        "const_cols_removed": 0,
        "missing_filled": 0,
        "missing_dropped_rows": 0,
        "headers_stripped": 0,
        "headers_snake_cased": 0,
        "encoding_repaired": 0,
        "spaces_trimmed": 0,
        "emails_normalized": 0,
        "phones_normalized": 0,
        "dates_normalized": 0,
    }

    # We keep 11 progress stages for compatibility with app.py.
    steps = 11
    current_step = 0

    def _progress(message: str) -> None:
        nonlocal current_step

        current_step += 1

        if progress_cb:
            progress_cb(
                min(current_step / steps, 1.0),
                message,
            )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1 — HEADER WHITESPACE
    # ═════════════════════════════════════════════════════════════════════════

    original_columns = [
        str(column)
        for column in df.columns
    ]

    stripped_columns = [
        column.strip()
        for column in original_columns
    ]

    report["headers_stripped"] = sum(
        1
        for old, new in zip(
            original_columns,
            stripped_columns,
        )
        if old != new
    )

    df.columns = stripped_columns

    _progress(
        "Cleaning column headers…"
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2 — ENCODING REPAIR
    # ═════════════════════════════════════════════════════════════════════════

    if opt["use_ftfy"]:

        total_repaired = 0

        string_columns = df.select_dtypes(
            include=["object", "string"]
        ).columns

        for column in string_columns:

            original_series = df[column].copy()

            df[column] = df[column].apply(
                _fix_encoding_ftfy
            )

            total_repaired += _count_changed_values(
                original_series,
                df[column],
            )

        report["encoding_repaired"] = total_repaired

    _progress(
        "Repairing text encoding…"
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3 — WHITESPACE NORMALIZATION
    # ═════════════════════════════════════════════════════════════════════════

    if opt["trim_spaces"]:

        total_trimmed = 0

        string_columns = df.select_dtypes(
            include=["object", "string"]
        ).columns

        for column in string_columns:

            original_series = df[column].copy()

            df[column] = df[column].apply(
                _trim_spaces
            )

            total_trimmed += _count_changed_values(
                original_series,
                df[column],
            )

        report["spaces_trimmed"] = total_trimmed

    _progress(
        "Trimming and normalizing whitespace…"
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 4 — EMAIL / PHONE / DATE NORMALIZATION
    # ═════════════════════════════════════════════════════════════════════════

    if opt["normalize_emails"]:

        for column in opt["email_columns"]:

            if column not in df.columns:
                continue

            original_series = df[column].copy()

            df[column] = df[column].apply(
                normalize_email
            )

            report["emails_normalized"] += (
                _count_changed_values(
                    original_series,
                    df[column],
                )
            )

    if opt["normalize_phones"]:

        for column in opt["phone_columns"]:

            if column not in df.columns:
                continue

            original_series = df[column].copy()

            df[column] = df[column].apply(
                normalize_phone
            )

            report["phones_normalized"] += (
                _count_changed_values(
                    original_series,
                    df[column],
                )
            )

    if opt["normalize_dates"]:

        for column in opt["date_columns"]:

            if column not in df.columns:
                continue

            original_series = df[column].copy()

            df[column] = df[column].apply(
                lambda value: normalize_date(
                    value,
                    opt["date_target_fmt"],
                )
            )

            report["dates_normalized"] += (
                _count_changed_values(
                    original_series,
                    df[column],
                )
            )

    _progress(
        "Normalizing email, phone and date data…"
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 5 — EMPTY ROWS
    # ═════════════════════════════════════════════════════════════════════════

    if opt["remove_empty_rows"]:

        df, removed = remove_empty_rows(df)

        report["empty_rows_removed"] = removed

    _progress(
        "Removing empty rows…"
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 6 — EMPTY COLUMNS
    # ═════════════════════════════════════════════════════════════════════════

    if opt["remove_empty_cols"]:

        df, removed = remove_empty_columns(df)

        report["empty_cols_removed"] = removed

    _progress(
        "Removing empty columns…"
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 7 — DUPLICATE COLUMNS
    # ═════════════════════════════════════════════════════════════════════════

    if opt["remove_dup_cols"]:

        df, removed = remove_duplicate_columns(df)

        report["dup_cols_removed"] = removed

    _progress(
        "Removing duplicate columns…"
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 8 — DUPLICATE ROWS
    # ═════════════════════════════════════════════════════════════════════════

    rows_before = len(df)

    df = df.drop_duplicates(
        keep="first"
    ).reset_index(drop=True)

    report["duplicates_removed"] = (
        rows_before - len(df)
    )

    _progress(
        "Removing duplicate rows…"
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 9 — CONSTANT COLUMNS
    # ═════════════════════════════════════════════════════════════════════════

    if opt["remove_const_cols"]:

        df, removed = remove_constant_columns(df)

        report["const_cols_removed"] = removed

    _progress(
        "Removing constant columns…"
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 10 — MISSING VALUES
    # ═════════════════════════════════════════════════════════════════════════

    missing_before_fill = int(
        df.isnull().sum().sum()
    )

    df, affected = fill_missing(
        df,
        opt["fill_strategy"],
    )

    missing_after_fill = int(
        df.isnull().sum().sum()
    )

    if opt["fill_strategy"] == (
        "Drop rows with missing values"
    ):

        report["missing_dropped_rows"] = affected
        report["missing_filled"] = 0

    else:

        report["missing_filled"] = max(
            0,
            missing_before_fill
            - missing_after_fill,
        )

        report["missing_dropped_rows"] = 0

    _progress(
        "Handling missing values…"
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 11 — SNAKE CASE HEADERS
    # ═════════════════════════════════════════════════════════════════════════

    if opt["snake_case"]:

        original_columns = [
            str(column)
            for column in df.columns
        ]

        df, changed = normalize_headers(df)

        report["headers_snake_cased"] = changed

    _progress(
        "Finalizing cleaned dataset…"
    )

    # ═════════════════════════════════════════════════════════════════════════
    # FINAL STATS
    # ═════════════════════════════════════════════════════════════════════════

    report["rows_after"] = len(df)

    report["cols_after"] = len(df.columns)

    report["missing_count_after"] = int(
        df.isnull().sum().sum()
    )

    return df, report
