"""
reports.py — Professional Report Generation for DataCleaner Pro V3.
"""

from __future__ import annotations

from utils.helpers import get_df_memory
import pandas as pd


_LINE = "─" * 54
_DLINE = "═" * 54


def build_text_report(
    report:   dict,
    filename: str,
    df_before: pd.DataFrame | None = None,
    df_after:  pd.DataFrame | None = None,
) -> str:
    """
    Build a professional plain-text cleaning report.

    Args:
        report:    Report dict returned by run_cleaning_pipeline().
        filename:  Original filename.
        df_before: Original DataFrame (optional, for memory stats).
        df_after:  Cleaned DataFrame (optional, for memory stats).

    Returns:
        Formatted multi-line string.
    """
    mem_before = get_df_memory(df_before) if df_before is not None else "—"
    mem_after  = get_df_memory(df_after)  if df_after  is not None else "—"

    rows_delta = report["rows_before"] - report["rows_after"]
    cols_delta = report["cols_before"] - report["cols_after"]

    lines = [
        _DLINE,
        "   DataCleaner Pro V3 — Cleaning Report",
        "   Commercial Edition",
        _DLINE,
        f"  File       : {filename}",
        f"  Generated  : {report['timestamp']}",
        _LINE,
        "  DATASET OVERVIEW",
        f"    Rows (before)    : {report['rows_before']:>10,}",
        f"    Rows (after)     : {report['rows_after']:>10,}   Δ {rows_delta:+,}",
        f"    Columns (before) : {report['cols_before']:>10}",
        f"    Columns (after)  : {report['cols_after']:>10}   Δ {cols_delta:+}",
        f"    Memory (before)  : {mem_before:>10}",
        f"    Memory (after)   : {mem_after:>10}",
        _LINE,
        "  CLEANING ACTIONS",
        f"    Duplicate rows removed   : {report['duplicates_removed']:>8,}",
        f"    Empty rows removed       : {report['empty_rows_removed']:>8,}",
        f"    Rows dropped (missing)   : {report['missing_dropped_rows']:>8,}",
        f"    Empty columns removed    : {report['empty_cols_removed']:>8}",
        f"    Duplicate cols removed   : {report['dup_cols_removed']:>8}",
        f"    Constant cols removed    : {report['const_cols_removed']:>8}",
        f"    Missing values filled    : {report['missing_filled']:>8,}",
        f"    Encoding repaired (ftfy) : {report['encoding_repaired']:>8,}",
        f"    Spaces trimmed           : {report['spaces_trimmed']:>8,}",
        f"    Headers stripped         : {report['headers_stripped']:>8}",
        f"    Headers → snake_case     : {report['headers_snake_cased']:>8}",
        f"    Emails normalized        : {report['emails_normalized']:>8,}",
        f"    Phones normalized        : {report['phones_normalized']:>8,}",
        f"    Dates normalized         : {report['dates_normalized']:>8,}",
        _LINE,
        "  BEFORE vs AFTER",
        f"    {'Metric':<25} {'Before':>10}  {'After':>10}",
        f"    {'─'*25} {'─'*10}  {'─'*10}",
        f"    {'Rows':<25} {report['rows_before']:>10,}  {report['rows_after']:>10,}",
        f"    {'Columns':<25} {report['cols_before']:>10}  {report['cols_after']:>10}",
        f"    {'Memory':<25} {mem_before:>10}  {mem_after:>10}",
        _DLINE,
        "  DataCleaner Pro V3 — Commercial Edition",
        "  Clean. Analyze. Export.",
        _DLINE,
    ]
    return "\n".join(lines)


def build_comparison_df(report: dict) -> pd.DataFrame:
    """Return a before/after comparison as a tidy DataFrame."""
    return pd.DataFrame(
        {
            "Metric": [
                "Rows", "Columns",
                "Duplicate Rows Removed", "Empty Rows Removed",
                "Missing Values Filled", "Encoding Repairs",
            ],
            "Before / Count": [
                report["rows_before"], report["cols_before"],
                report["duplicates_removed"], report["empty_rows_removed"],
                report["missing_filled"], report["encoding_repaired"],
            ],
            "After / Result": [
                report["rows_after"], report["cols_after"],
                0, 0, 0, 0,
            ],
        }
    )
