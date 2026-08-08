"""
helpers.py — Shared utility functions for DataCleaner Pro V3.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MAX_FILE_SIZE_MB = 50
MAX_ROWS_FUZZY = 50_000
MAX_ROWS_PROFILING = 500_000

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".pdf"}

# Only DataCleaner-owned session keys should be cleared when files change.
DATA_STATE_KEYS = {
    "demo_mode",
    "df_clean",
    "df_raw_snap",
    "clean_report",
    "active_file",
    "batch_results",
    "batch_before",
}


# ═══════════════════════════════════════════════════════════════════════════════
# STRING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def to_snake_case(name: str) -> str:
    """Convert a column/header name to normalized snake_case."""
    name = str(name).strip()

    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^\w]", "_", name)
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    name = re.sub(r"_+", "_", name)

    return name.lower().strip("_")


def human_readable_size(num_bytes: int) -> str:
    """Return a human-readable byte size."""
    num_bytes = max(0, int(num_bytes))

    if num_bytes < 1_024:
        return f"{num_bytes} B"

    if num_bytes < 1_024 ** 2:
        return f"{num_bytes / 1_024:.1f} KB"

    if num_bytes < 1_024 ** 3:
        return f"{num_bytes / 1_024 ** 2:.2f} MB"

    return f"{num_bytes / 1_024 ** 3:.2f} GB"


def get_df_memory(df: pd.DataFrame) -> str:
    """Return human-readable DataFrame memory usage."""
    if df is None:
        return "0 B"

    return human_readable_size(
        int(df.memory_usage(deep=True).sum())
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FILE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_uploaded_file(uploaded_file) -> tuple[bool, str]:
    """
    Validate an uploaded file before processing.

    Returns:
        (is_valid, error_message)
    """
    if uploaded_file is None:
        return False, "No file provided."

    filename = str(getattr(uploaded_file, "name", ""))
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))

        return False, (
            f"Unsupported file type: **{ext or 'unknown'}**. "
            f"Supported: {supported}"
        )

    try:
        size_bytes = int(uploaded_file.size)
    except (AttributeError, TypeError, ValueError):
        return False, "Could not determine the uploaded file size."

    max_bytes = MAX_FILE_SIZE_MB * 1_024 * 1_024

    if size_bytes > max_bytes:
        size_mb = size_bytes / 1_024 / 1_024

        return False, (
            f"File **{filename}** is {size_mb:.1f} MB. "
            f"Maximum allowed: {MAX_FILE_SIZE_MB} MB."
        )

    return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
# FILE SIGNATURE
# ═══════════════════════════════════════════════════════════════════════════════

def file_signature(files: list) -> str:
    """
    Generate a stable signature from uploaded filenames and sizes.

    Sorting makes the signature independent of upload ordering.
    """
    if not files:
        return ""

    parts = []

    for file in files:
        name = str(getattr(file, "name", ""))
        size = int(getattr(file, "size", 0))

        parts.append(f"{name}_{size}")

    return "|".join(sorted(parts))


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def reset_state_if_new_files(files: list) -> None:
    """
    Clear DataCleaner-specific state when the uploaded file set changes.

    Widget state and unrelated session state are preserved.
    """
    sig = file_signature(files)

    if st.session_state.get("_file_sig") == sig:
        return

    for key in list(DATA_STATE_KEYS):
        st.session_state.pop(key, None)

    # Remove per-file cached DataCleaner state.
    for key in list(st.session_state.keys()):
        if (
            key.startswith("profile_")
            or key.startswith("pdf_")
            or key.startswith("_bytes_")
        ):
            st.session_state.pop(key, None)

    st.session_state["_file_sig"] = sig


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEL SHEETS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_excel_sheet_names(
    file_bytes: bytes,
    filename: str,
) -> list[str]:
    """
    Return Excel worksheet names.

    Returns an empty list when the file cannot be inspected.
    """
    ext = Path(filename).suffix.lower()

    if ext not in {".xlsx", ".xls"}:
        return []

    try:
        buffer = io.BytesIO(file_bytes)

        engine = "openpyxl" if ext == ".xlsx" else "xlrd"

        with pd.ExcelFile(buffer, engine=engine) as excel:
            return list(excel.sheet_names)

    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# DATAFRAME LOADER
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_dataframe(
    file_bytes: bytes,
    filename: str,
    sheet_name: str | None = None,
) -> pd.DataFrame | None:
    """
    Load CSV or Excel data from raw bytes.

    Args:
        file_bytes: Raw uploaded file bytes.
        filename: Original filename.
        sheet_name: Optional Excel worksheet name.

    Returns:
        DataFrame or None if loading fails.
    """
    ext = Path(filename).suffix.lower()

    if ext not in {".csv", ".xlsx", ".xls"}:
        return None

    buffer = io.BytesIO(file_bytes)

    try:

        # ── CSV ───────────────────────────────────────────────────────────────
        if ext == ".csv":

            encodings = [
                "utf-8-sig",
                "utf-8",
                "cp1252",
                "latin-1",
            ]

            for encoding in encodings:
                try:
                    buffer.seek(0)

                    df = pd.read_csv(
                        buffer,
                        encoding=encoding,
                        low_memory=False,
                    )

                    if df.empty:
                        return None

                    return df

                except (
                    UnicodeDecodeError,
                    pd.errors.ParserError,
                ):
                    continue

            # Final fallback.
            buffer.seek(0)

            df = pd.read_csv(
                buffer,
                encoding="latin-1",
                encoding_errors="replace",
                low_memory=False,
            )

            return df if not df.empty else None

        # ── Excel ─────────────────────────────────────────────────────────────
        if ext in {".xlsx", ".xls"}:

            engine = (
                "openpyxl"
                if ext == ".xlsx"
                else "xlrd"
            )

            buffer.seek(0)

            df = pd.read_excel(
                buffer,
                engine=engine,
                sheet_name=sheet_name,
            )

            if df.empty:
                return None

            return df

    except Exception:
        return None

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# DATAFRAME SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def df_to_bytes(
    df: pd.DataFrame,
    fmt: str,
) -> bytes:
    """
    Serialize a DataFrame into downloadable bytes.

    Supported formats:
        - xlsx
        - csv
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    fmt = str(fmt).lower().strip()

    if fmt not in {"xlsx", "csv"}:
        raise ValueError(
            f"Unsupported format '{fmt}'. "
            "Expected 'xlsx' or 'csv'."
        )

    buffer = io.BytesIO()

    if fmt == "xlsx":
        with pd.ExcelWriter(
            buffer,
            engine="openpyxl",
        ) as writer:
            df.to_excel(
                writer,
                index=False,
                sheet_name="Cleaned Data",
            )

    else:
        df.to_csv(
            buffer,
            index=False,
            encoding="utf-8-sig",
        )

    return buffer.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def info_box(text: str) -> None:
    """Render an informational HTML box."""
    st.markdown(
        f'<div class="info-box">{text}</div>',
        unsafe_allow_html=True,
    )


def success_box(text: str) -> None:
    """Render a success HTML box."""
    st.markdown(
        f'<div class="success-box">{text}</div>',
        unsafe_allow_html=True,
    )


def warning_box(text: str) -> None:
    """Render a warning HTML box."""
    st.markdown(
        f'<div class="warning-box">{text}</div>',
        unsafe_allow_html=True,
    )


def error_box(text: str) -> None:
    """Render an error HTML box."""
    st.markdown(
        f'<div class="error-box">{text}</div>',
        unsafe_allow_html=True,
    )


def section_header(text: str) -> None:
    """Render a styled section header."""
    st.markdown(
        f'<div class="section-header">{text}</div>',
        unsafe_allow_html=True,
    )


def metric_card(
    icon: str,
    value: str,
    label: str,
) -> str:
    """Return HTML for a metric card."""
    return f"""
    <div class="metric-card">
        <div class="icon">{icon}</div>
        <div class="value">{value}</div>
        <div class="label">{label}</div>
    </div>
    """
