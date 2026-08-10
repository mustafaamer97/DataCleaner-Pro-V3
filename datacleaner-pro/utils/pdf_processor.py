"""
pdf_processor.py — PDF Table Extraction for DataCleaner Pro V3.

Uses a 3-tier extraction strategy:
    Tier 1: extract_tables()
    Tier 2: extract_table()
    Tier 3: extract_text() → plain text fallback

This module contains PDF processing logic only.
No Streamlit/UI dependencies.
"""

from __future__ import annotations

import io
from typing import Callable

import pandas as pd
import pdfplumber


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum number of pages processed when page_selection == "all".
# This limit does NOT apply to specific-page selection, where the user
# has already explicitly bounded the range.
MAX_PDF_PAGES = 100


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_list_to_df(
    table: list[list],
) -> pd.DataFrame | None:
    """
    Convert a raw PDF table into a DataFrame.

    Handles:
    - Empty tables
    - Missing headers
    - Duplicate headers
    - Rows with different lengths
    - Completely empty rows

    Returns None when the table cannot be converted safely.
    """
    try:
        if not table or len(table) < 2:
            return None

        raw_headers = table[0] or []

        if not raw_headers:
            return None

        # ── Normalize headers ────────────────────────────────────────────────
        headers: list[str] = []

        for i, value in enumerate(raw_headers):
            if value is None or not str(value).strip():
                headers.append(f"Col_{i + 1}")
            else:
                headers.append(str(value).strip())

        # ── De-duplicate headers ────────────────────────────────────────────
        seen: dict[str, int] = {}
        clean_headers: list[str] = []

        for header in headers:
            if header in seen:
                seen[header] += 1
                clean_headers.append(
                    f"{header}_{seen[header]}"
                )
            else:
                seen[header] = 0
                clean_headers.append(header)

        column_count = len(clean_headers)

        # ── Normalize row lengths ───────────────────────────────────────────
        normalized_rows: list[list] = []

        for row in table[1:]:
            if row is None:
                continue

            row = list(row)

            if len(row) < column_count:
                row = row + [None] * (column_count - len(row))
            elif len(row) > column_count:
                row = row[:column_count]

            # Ignore completely empty rows.
            if all(
                value is None or not str(value).strip()
                for value in row
            ):
                continue

            normalized_rows.append(row)

        if not normalized_rows:
            return None

        df = pd.DataFrame(
            normalized_rows,
            columns=clean_headers,
        )

        return df if not df.empty else None

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PDF EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pdf_tables(
    file_bytes: bytes,
    filename: str,
    page_selection: str = "all",
    specific_pages: list[int] | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> list[dict]:
    """
    Extract tables/content from a PDF.

    Extraction priority:

        1. extract_tables()
        2. extract_table()
        3. extract_text()

    Args:
        file_bytes:
            Raw PDF bytes.

        filename:
            Original filename. Used for error reporting metadata.

        page_selection:
            Either "all" or "specific".
            When "all", processing is limited to MAX_PDF_PAGES pages.
            When the PDF exceeds this limit, extraction is rejected and
            a single result dict with method "page_limit_exceeded" is
            returned so the caller can display a clear message.

        specific_pages:
            1-based page numbers when page_selection == "specific".
            This path is NOT subject to the MAX_PDF_PAGES limit.

        progress_cb:
            Optional callback:
                progress_cb(fraction, message)

    Returns:
        List of result dictionaries:

        Normal result:
        {
            "page": int,
            "table_index": int | None,
            "method": str,
            "dataframe": pd.DataFrame | None,
            "rows": int,
            "cols": int,
        }

        Page-limit rejection (method == "page_limit_exceeded"):
        {
            "page": None,
            "table_index": None,
            "method": "page_limit_exceeded",
            "dataframe": None,
            "rows": 0,
            "cols": 0,
            "total_pages": int,
            "max_pages": int,
            "filename": str,
        }

    Notes:
        This function does not display UI errors.
        The caller is responsible for presenting errors to users.
    """

    results: list[dict] = []

    # ── Basic input validation ───────────────────────────────────────────────
    if not isinstance(file_bytes, (bytes, bytearray)):
        return results

    if not file_bytes:
        return results

    try:
        buffer = io.BytesIO(file_bytes)

        with pdfplumber.open(buffer) as pdf:

            total_pages = len(pdf.pages)

            if total_pages == 0:
                return results

            # ── Determine pages to process ───────────────────────────────────
            if page_selection == "specific":

                # Specific-page path: NOT subject to MAX_PDF_PAGES.
                # Existing bounds validation is preserved exactly.
                requested_pages = specific_pages or []

                pages_idx = [
                    page - 1
                    for page in requested_pages
                    if isinstance(page, int)
                    and 1 <= page <= total_pages
                ]

                # If specific selection was requested but no valid
                # pages were supplied, return nothing rather than
                # unexpectedly processing the entire PDF.
                if not pages_idx:
                    return results

            else:
                # "all" pages mode — enforce MAX_PDF_PAGES.
                if total_pages > MAX_PDF_PAGES:
                    # Reject cleanly. Return a structured result that the
                    # caller can inspect to display a specific message.
                    # Do NOT process any pages.
                    results.append(
                        {
                            "page":        None,
                            "table_index": None,
                            "method":      "page_limit_exceeded",
                            "dataframe":   None,
                            "rows":        0,
                            "cols":        0,
                            "total_pages": total_pages,
                            "max_pages":   MAX_PDF_PAGES,
                            "filename":    str(filename),
                        }
                    )
                    return results

                pages_idx = list(range(total_pages))

            total = len(pages_idx)

            # ── Process selected pages ───────────────────────────────────────
            for position, page_num in enumerate(pages_idx):

                fraction = (position + 1) / max(total, 1)

                if progress_cb:
                    progress_cb(
                        fraction,
                        f"Reading page {page_num + 1} / {total_pages}…",
                    )

                try:
                    page = pdf.pages[page_num]
                except Exception:
                    results.append(
                        _empty_result(page_num + 1)
                    )
                    continue

                # ═════════════════════════════════════════════════════════════
                # TIER 1 — Multiple table extraction
                # ═════════════════════════════════════════════════════════════
                try:
                    raw_tables = page.extract_tables() or []
                except Exception:
                    raw_tables = []

                added_tables = False

                for table_index, table in enumerate(raw_tables, start=1):

                    df_table = _safe_list_to_df(table)

                    if df_table is None or df_table.empty:
                        continue

                    results.append(
                        {
                            "page":        page_num + 1,
                            "table_index": table_index,
                            "method":      "extract_tables()",
                            "dataframe":   df_table,
                            "rows":        len(df_table),
                            "cols":        len(df_table.columns),
                        }
                    )

                    added_tables = True

                if added_tables:
                    continue

                # ═════════════════════════════════════════════════════════════
                # TIER 2 — Single table extraction
                # ═════════════════════════════════════════════════════════════
                try:
                    single_table = page.extract_table()
                except Exception:
                    single_table = None

                if single_table:
                    df_table = _safe_list_to_df(single_table)

                    if df_table is not None and not df_table.empty:
                        results.append(
                            {
                                "page":        page_num + 1,
                                "table_index": 1,
                                "method":      "extract_table()",
                                "dataframe":   df_table,
                                "rows":        len(df_table),
                                "cols":        len(df_table.columns),
                            }
                        )
                        continue

                # ═════════════════════════════════════════════════════════════
                # TIER 3 — Plain text fallback
                # ═════════════════════════════════════════════════════════════
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

                if text.strip():

                    lines = [
                        line.strip()
                        for line in text.splitlines()
                        if line.strip()
                    ]

                    df_text = pd.DataFrame(
                        {"Text Content": lines}
                    )

                    results.append(
                        {
                            "page":        page_num + 1,
                            "table_index": 1,
                            "method":      (
                                "extract_text() "
                                "← plain text fallback"
                            ),
                            "dataframe":   df_text,
                            "rows":        len(df_text),
                            "cols":        1,
                        }
                    )

                else:
                    results.append(
                        _empty_result(page_num + 1)
                    )

    except Exception as exc:
        # Do not use Streamlit here.
        # Return structured error information to the caller.
        results.append(
            {
                "page":        None,
                "table_index": None,
                "method":      "error",
                "dataframe":   None,
                "rows":        0,
                "cols":        0,
                "error":       str(exc),
                "filename":    str(filename),
            }
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EMPTY RESULT
# ═══════════════════════════════════════════════════════════════════════════════

def _empty_result(page_num: int) -> dict:
    """Return a standardized empty-page result."""
    return {
        "page":        page_num,
        "table_index": None,
        "method":      "none",
        "dataframe":   None,
        "rows":        0,
        "cols":        0,
    }
