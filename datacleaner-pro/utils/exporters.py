"""
exporters.py — Export Engine for DataCleaner Pro V3.

Handles Excel, CSV, and ZIP bundle exports.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Cleaned Data") -> bytes:
    """Serialize a DataFrame to Excel (.xlsx) bytes."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to CSV bytes (UTF-8 with BOM for Excel compat)."""
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def build_batch_zip(
    results: list[tuple[str, pd.DataFrame, dict]],
    report_fn,
) -> bytes:
    """
    Build a ZIP archive from batch cleaning results.

    Args:
        results:   List of (original_filename, cleaned_df, report_dict).
        report_fn: Callable(report_dict, filename) → str (the report text).

    Returns:
        Raw ZIP bytes.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for original_name, df_clean, report in results:
            stem = Path(original_name).stem

            # Cleaned Excel
            zf.writestr(
                f"cleaned_{stem}.xlsx",
                df_to_excel_bytes(df_clean),
            )

            # Cleaned CSV
            zf.writestr(
                f"cleaned_{stem}.csv",
                df_to_csv_bytes(df_clean),
            )

            # Text report
            report_text = report_fn(report, original_name)
            zf.writestr(
                f"reports/report_{stem}.txt",
                report_text.encode("utf-8"),
            )

    return buf.getvalue()
