"""
helpers.py — Shared utility functions for DataCleaner Pro V3.
"""

from __future__ import annotations

import re
import io
from pathlib import Path

import pandas as pd
import streamlit as st


# ── Constants ────────────────────────────────────────────────────────────────

MAX_FILE_SIZE_MB   = 50
MAX_ROWS_FUZZY     = 50_000
MAX_ROWS_PROFILING = 500_000

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".pdf"}


# ── String Utilities ─────────────────────────────────────────────────────────

def to_snake_case(name: str) -> str:
    """Convert any string to snake_case."""
    name = str(name).strip()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^\w]", "_", name)
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.lower().strip("_")


def human_readable_size(num_bytes: int) -> str:
    """Return a human-readable file/memory size string."""
    if num_bytes < 1_024:
        return f"{num_bytes} B"
    if num_bytes < 1_024 ** 2:
        return f"{num_bytes / 1_024:.1f} KB"
    if num_bytes < 1_024 ** 3:
        return f"{num_bytes / 1_024 ** 2:.2f} MB"
    return f"{num_bytes / 1_024 ** 3:.2f} GB"


def get_df_memory(df: pd.DataFrame) -> str:
    """Return human-readable memory usage of a DataFrame."""
    return human_readable_size(int(df.memory_usage(deep=True).sum()))


# ── File Validation ───────────────────────────────────────────────────────────

def validate_uploaded_file(uploaded_file) -> tuple[bool, str]:
    """
    Validate an uploaded file before processing.

    Returns:
        (is_valid, error_message)
    """
    if uploaded_file is None:
        return False, "No file provided."

    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, (
            f"Unsupported file type: **{ext}**. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    size_mb = uploaded_file.size / 1_024 / 1_024
    if size_mb > MAX_FILE_SIZE_MB:
        return False, (
            f"File **{uploaded_file.name}** is {size_mb:.1f} MB. "
            f"Maximum allowed: {MAX_FILE_SIZE_MB} MB."
        )

    return True, ""


# ── File Signature ────────────────────────────────────────────────────────────

def file_signature(files: list) -> str:
    """Generate a stable signature string from a list of uploaded files."""
    if not files:
        return ""
    return "|".join(sorted(f"{f.name}_{f.size}" for f in files))


# ── Session State ─────────────────────────────────────────────────────────────

def reset_state_if_new_files(files: list) -> None:
    """
    Clear all session_state keys (except internal ones)
    when a different set of files is detected.
    """
    sig = file_signature(files)
    if st.session_state.get("_file_sig") != sig:
        protected = {"_file_sig"}
        for key in list(st.session_state.keys()):
            if key not in protected:
                del st.session_state[key]
        st.session_state["_file_sig"] = sig


# ── DataFrame Loader ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame | None:
    """
    Load a CSV or Excel file from raw bytes into a DataFrame.

    Tries multiple encodings for CSV files.
    Returns None on failure (never raises).
    """
    ext = Path(filename).suffix.lower()
    buffer = io.BytesIO(file_bytes)

    try:
        if ext == ".csv":
            for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]:
                try:
                    buffer.seek(0)
                    df = pd.read_csv(buffer, encoding=enc)
                    if df.empty:
                        return None
                    return df
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            buffer.seek(0)
            return pd.read_csv(buffer, encoding="latin-1", errors="replace")

        if ext in (".xlsx", ".xls"):
            buffer.seek(0)
            engine = "openpyxl" if ext == ".xlsx" else "xlrd"
            df = pd.read_excel(buffer, engine=engine)
            return df if not df.empty else None

    except Exception:
        return None

    return None


# ── DataFrame Serializer ──────────────────────────────────────────────────────

def df_to_bytes(df: pd.DataFrame, fmt: str) -> bytes:
    """
    Serialize a DataFrame to bytes.

    Args:
        df:  The DataFrame to serialize.
        fmt: 'xlsx' or 'csv'.

    Returns:
        Raw bytes ready for st.download_button.
    """
    buffer = io.BytesIO()
    if fmt == "xlsx":
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Cleaned Data")
    else:
        df.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue()


# ── HTML Helpers ──────────────────────────────────────────────────────────────

def info_box(text: str) -> None:
    st.markdown(
        f'<div class="info-box">{text}</div>', unsafe_allow_html=True
    )


def success_box(text: str) -> None:
    st.markdown(
        f'<div class="success-box">{text}</div>', unsafe_allow_html=True
    )


def warning_box(text: str) -> None:
    st.markdown(
        f'<div class="warning-box">{text}</div>', unsafe_allow_html=True
    )


def error_box(text: str) -> None:
    st.markdown(
        f'<div class="error-box">{text}</div>', unsafe_allow_html=True
    )


def section_header(text: str) -> None:
    st.markdown(
        f'<div class="section-header">{text}</div>', unsafe_allow_html=True
    )


def metric_card(icon: str, value: str, label: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="icon">{icon}</div>
        <div class="value">{value}</div>
        <div class="label">{label}</div>
    </div>"""
