"""
pdf_processor.py — PDF Table Extraction for DataCleaner Pro V3.

Uses a 3-tier extraction strategy:
    Tier 1: extract_tables()
    Tier 2: extract_table()
    Tier 3: extract_text() → plain text fallback
"""

from __future__ import annotations

import io
from typing import Callable

import pandas as pd
import pdfplumber
import streamlit as st


def _safe_list_to_df(table: list[list]) -> pd.DataFrame | None:
    """
    Convert a raw list-of-lists table to a DataFrame.
    De-duplicates column headers. Returns None on failure.
    """
    try:
        if not table or len(table) < 2:
            return None

        raw_headers = table[0]
        headers = [
            str(h).strip() if h else f"Col_{i}"
            for i, h in enumerate(raw_headers)
        ]

        # De-duplicate headers
        seen: dict[str, int] = {}
        clean_headers: list[str] = []
        for h in headers:
            if h in seen:
                seen[h] += 1
                clean_headers.append(f"{h}_{seen[h]}")
            else:
                seen[h] = 0
                clean_headers.append(h)

        return pd.DataFrame(table[1:], columns=clean_headers)
    except Exception:
        return None


def extract_pdf_tables(
    file_bytes: bytes,
    filename:   str,
    page_selection:  str = "all",
    specific_pages:  list[int] | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> list[dict]:
    """
    Extract tables from a PDF file.

    Args:
        file_bytes:      Raw bytes of the PDF file.
        filename:        Original filename (for display).
        page_selection:  'all' or 'specific'.
        specific_pages:  1-based page numbers when page_selection == 'specific'.
        progress_cb:     Optional progress callback(fraction, message).

    Returns:
        List of result dicts:
            {
                page:        int (1-based),
                table_index: int | None,
                method:      str,
                dataframe:   pd.DataFrame | None,
                rows:        int,
                cols:        int,
            }
    """
    results: list[dict] = []

    try:
        buffer = io.BytesIO(file_bytes)
        with pdfplumber.open(buffer) as pdf:
            total_pages = len(pdf.pages)

            if page_selection == "specific" and specific_pages:
                pages_idx = [p - 1 for p in specific_pages if 1 <= p <= total_pages]
            else:
                pages_idx = list(range(total_pages))

            total = len(pages_idx)

            for pos, page_num in enumerate(pages_idx):
                frac = (pos + 1) / max(total, 1)
                msg  = f"Reading page {page_num + 1} / {total_pages}…"
                if progress_cb:
                    progress_cb(frac, msg)

                try:
                    page = pdf.pages[page_num]
                except Exception:
                    results.append(_empty_result(page_num + 1))
                    continue

                # ── Tier 1: extract_tables() ─────────────────
                try:
                    raw_tables = page.extract_tables() or []
                except Exception:
                    raw_tables = []

                if raw_tables:
                    added = False
                    for t_idx, table in enumerate(raw_tables):
                        df_t = _safe_list_to_df(table)
                        if df_t is not None and not df_t.empty:
                            results.append({
                                "page":        page_num + 1,
                                "table_index": t_idx + 1,
                                "method":      "extract_tables()",
                                "dataframe":   df_t,
                                "rows":        len(df_t),
                                "cols":        len(df_t.columns),
                            })
                            added = True
                    if added:
                        continue

                # ── Tier 2: extract_table() ──────────────────
                try:
                    single = page.extract_table()
                except Exception:
                    single = None

                if single and len(single) > 1:
                    df_t = _safe_list_to_df(single)
                    if df_t is not None and not df_t.empty:
                        results.append({
                            "page":        page_num + 1,
                            "table_index": 1,
                            "method":      "extract_table()",
                            "dataframe":   df_t,
                            "rows":        len(df_t),
                            "cols":        len(df_t.columns),
                        })
                        continue

                # ── Tier 3: extract_text() ───────────────────
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

                if text.strip():
                    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                    df_t  = pd.DataFrame({"Text Content": lines})
                    results.append({
                        "page":        page_num + 1,
                        "table_index": 1,
                        "method":      "extract_text() ← plain text fallback",
                        "dataframe":   df_t,
                        "rows":        len(df_t),
                        "cols":        1,
                    })
                else:
                    results.append(_empty_result(page_num + 1))

    except Exception as e:
        st.error(f"❌ Could not read PDF **{filename}**: {e}")

    return results


def _empty_result(page_num: int) -> dict:
    return {
        "page":        page_num,
        "table_index": None,
        "method":      "none",
        "dataframe":   None,
        "rows":        0,
        "cols":        0,
    }
