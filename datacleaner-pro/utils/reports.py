"""
reports.py — Professional Report Generation for DataCleaner Pro V3.
"""

from __future__ import annotations

import pandas as pd

from utils.helpers import get_df_memory


_LINE  = "─" * 54
_DLINE = "═" * 54


def build_text_report(
    report:    dict,
    filename:  str,
    df_before: pd.DataFrame | None = None,
    df_after:  pd.DataFrame | None = None,
) -> str:
    """
    Build a professional plain-text cleaning report.

    Parameters
    ----------
    report : dict
        Report dict returned by ``run_cleaning_pipeline()``.
        Expected keys: rows_before, rows_after, cols_before, cols_after,
        missing_count_before, missing_count_after, duplicates_removed,
        empty_rows_removed, missing_dropped_rows, empty_cols_removed,
        dup_cols_removed, const_cols_removed, missing_filled,
        encoding_repaired, spaces_trimmed, headers_stripped,
        headers_snake_cased, emails_normalized, phones_normalized,
        dates_normalized, timestamp, fill_strategy_used.
    filename : str
        Original filename for display.
    df_before : pd.DataFrame, optional
        Original DataFrame — used for memory stats.
    df_after : pd.DataFrame, optional
        Cleaned DataFrame — used for memory stats.

    Returns
    -------
    str
        Formatted multi-line plain-text report.
    """
    mem_before = get_df_memory(df_before) if df_before is not None else "—"
    mem_after  = get_df_memory(df_after)  if df_after  is not None else "—"

    rows_delta = report["rows_after"]  - report["rows_before"]
    cols_delta = report["cols_after"]  - report["cols_before"]

    # Use the actual tracked missing counts (not a derived formula)
    missing_before = report.get("missing_count_before", "—")
    missing_after  = report.get("missing_count_after",  "—")

    fill_strategy = report.get("fill_strategy_used", "—")

    # Format missing counts safely
    def _fmt(v) -> str:
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)

    lines = [
        _DLINE,
        "   DataCleaner Pro V3 — Cleaning Report",
        "   Commercial Edition",
        _DLINE,
        f"  File            : {filename}",
        f"  Generated       : {report['timestamp']}",
        f"  Missing strategy: {fill_strategy}",
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
        f"    {'Metric':<26} {'Before':>10}   {'After':>10}",
        f"    {'─' * 26} {'─' * 10}   {'─' * 10}",
        f"    {'Rows':<26} {report['rows_before']:>10,}   {report['rows_after']:>10,}",
        f"    {'Columns':<26} {report['cols_before']:>10}   {report['cols_after']:>10}",
        f"    {'Missing values':<26} {_fmt(missing_before):>10}   {_fmt(missing_after):>10}",
        f"    {'Memory':<26} {mem_before:>10}   {mem_after:>10}",
        _DLINE,
        "  DataCleaner Pro V3 — Commercial Edition",
        "  Clean. Analyze. Export.",
        _DLINE,
    ]
    return "\n".join(lines)


def build_comparison_df(report: dict) -> pd.DataFrame:
    """
    Return a tidy before/after comparison as a DataFrame.
    Suitable for display with st.dataframe().
    """
    missing_before = report.get("missing_count_before", 0)
    missing_after  = report.get("missing_count_after",  0)

    return pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Missing Values",
            "Duplicate Rows Removed",
            "Empty Rows Removed",
            "Encoding Repairs",
        ],
        "Before": [
            report["rows_before"],
            report["cols_before"],
            missing_before,
            report["duplicates_removed"],
            report["empty_rows_removed"],
            report["encoding_repaired"],
        ],
        "After": [
            report["rows_after"],
            report["cols_after"],
            missing_after,
            0,
            0,
            0,
        ],
    })
