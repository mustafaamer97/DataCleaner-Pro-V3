"""
exporters.py — Export Engine for DataCleaner Pro V3.

Handles Excel, CSV, and ZIP bundle exports.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_sheet_name(name: str) -> str:
    """
    Return a valid Excel worksheet name.

    Excel limitations:
    - Maximum 31 characters
    - Cannot contain: : \\ / ? * [ ]
    """
    name = str(name).strip() or "Cleaned Data"

    # Remove characters forbidden by Excel.
    name = re.sub(r'[:\\/?*\[\]]', "_", name)

    # Excel worksheet names cannot exceed 31 characters.
    name = name[:31].strip()

    return name or "Cleaned Data"


def _safe_filename_stem(name: str) -> str:
    """
    Return a safe filename stem for ZIP exports.

    Prevents path separators and problematic filesystem characters
    from leaking into archive paths.
    """
    stem = Path(str(name)).stem

    # Replace unsafe characters with underscores.
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem)

    # Collapse repeated underscores.
    stem = re.sub(r"_+", "_", stem).strip(" ._")

    return stem or "data"


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEL EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def df_to_excel_bytes(
    df: pd.DataFrame,
    sheet_name: str = "Cleaned Data",
) -> bytes:
    """Serialize a DataFrame to Excel (.xlsx) bytes."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    safe_sheet_name = _safe_sheet_name(sheet_name)

    buf = io.BytesIO()

    with pd.ExcelWriter(
        buf,
        engine="openpyxl",
    ) as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name=safe_sheet_name,
        )

    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Serialize a DataFrame to CSV bytes.

    UTF-8 BOM is included for reliable Excel compatibility.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    return df.to_csv(
        index=False,
        encoding="utf-8-sig",
    ).encode("utf-8-sig")


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH ZIP EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def build_batch_zip(
    results: list[tuple[str, pd.DataFrame, dict]],
    report_fn,
) -> bytes:
    """
    Build a ZIP archive from batch cleaning results.

    Args:
        results:
            List of:
                (original_filename, cleaned_df, report_dict)

        report_fn:
            Callable(report_dict, filename) -> str

    Returns:
        Raw ZIP bytes.
    """
    if not isinstance(results, list):
        raise TypeError("results must be a list.")

    if not callable(report_fn):
        raise TypeError("report_fn must be callable.")

    buf = io.BytesIO()

    used_stems: set[str] = set()

    with zipfile.ZipFile(
        buf,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:

        for original_name, df_clean, report in results:

            if not isinstance(df_clean, pd.DataFrame):
                raise TypeError(
                    f"Cleaned data for '{original_name}' "
                    "must be a pandas DataFrame."
                )

            stem = _safe_filename_stem(original_name)

            # Prevent duplicate archive filenames.
            base_stem = stem
            counter = 2

            while stem.lower() in used_stems:
                stem = f"{base_stem}_{counter}"
                counter += 1

            used_stems.add(stem.lower())

            # ── Cleaned Excel ────────────────────────────────────────────────
            zf.writestr(
                f"cleaned_{stem}.xlsx",
                df_to_excel_bytes(df_clean),
            )

            # ── Cleaned CSV ──────────────────────────────────────────────────
            zf.writestr(
                f"cleaned_{stem}.csv",
                df_to_csv_bytes(df_clean),
            )

            # ── Text Report ──────────────────────────────────────────────────
            report_text = report_fn(
                report,
                original_name,
            )

            if not isinstance(report_text, str):
                report_text = str(report_text)

            zf.writestr(
                f"reports/report_{stem}.txt",
                report_text.encode("utf-8"),
            )

    return buf.getvalue()
