"""
tests/test_p0_fixes.py
======================
Regression tests for DataCleaner Pro V3 — P0 fixes.

Covers:
    F01 — build_batch_zip() accepts before_map parameter
    F02 — load_dataframe() enforces MAX_ROWS / raises RowLimitExceeded
    F03 — extract_pdf_tables() enforces MAX_PDF_PAGES in "all" mode
    F05 — _count_changed_values importable; _count_encoding_repairs absent

Run from repository root:
    pytest tests/ -v --tb=short

All tests are isolated and deterministic.
No external files, no network access, no Streamlit context required.
"""

from __future__ import annotations

import io
import sys
import os
import zipfile
import importlib
import importlib.util

# Ensure repository root is on sys.path so utils.* imports resolve
# regardless of how pytest is invoked.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import pytest

from utils.helpers import MAX_ROWS, RowLimitExceeded, load_dataframe
from utils.exporters import build_batch_zip
from utils.pdf_processor import MAX_PDF_PAGES, extract_pdf_tables


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_csv_bytes(nrows: int, ncols: int = 3) -> bytes:
    """
    Return UTF-8 CSV bytes with exactly nrows data rows (plus one header row).

    Uses minimal integer values to keep memory low.
    """
    header = ",".join(f"col{c}" for c in range(ncols))
    lines  = [header]
    for r in range(nrows):
        lines.append(",".join(str(r * ncols + c) for c in range(ncols)))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _make_excel_bytes(nrows: int, ncols: int = 2) -> bytes:
    """
    Return .xlsx bytes with exactly nrows data rows.
    Uses openpyxl directly — no file system required.
    """
    buf = io.BytesIO()
    df  = pd.DataFrame(
        {f"col{c}": range(nrows) for c in range(ncols)}
    )
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


def _make_report_fn_spy():
    """
    Return (spy_fn, calls_list).

    spy_fn accepts any positional arguments, records them in calls_list,
    and returns a plain string.  This lets tests inspect exactly which
    arguments build_batch_zip() passed to report_fn.
    """
    calls: list[dict] = []

    def _spy(*args):
        calls.append({"args": args})
        return "REPORT TEXT"

    return _spy, calls


def _minimal_report() -> dict:
    """
    Return the minimal report dict expected by build_batch_zip / build_text_report.

    All keys are taken from the report dict produced by run_cleaning_pipeline()
    in utils/cleaning.py.
    """
    return {
        "timestamp":            "2024-01-01 00:00:00",
        "fill_strategy_used":   "Auto (Median/Mode)",
        "rows_before":          5,
        "rows_after":           5,
        "cols_before":          2,
        "cols_after":           2,
        "missing_count_before": 0,
        "missing_count_after":  0,
        "duplicates_removed":   0,
        "empty_rows_removed":   0,
        "missing_dropped_rows": 0,
        "empty_cols_removed":   0,
        "dup_cols_removed":     0,
        "const_cols_removed":   0,
        "missing_filled":       0,
        "encoding_repaired":    0,
        "spaces_trimmed":       0,
        "headers_stripped":     0,
        "headers_snake_cased":  0,
        "emails_normalized":    0,
        "phones_normalized":    0,
        "dates_normalized":     0,
    }


def _sample_results(n: int = 2) -> list[tuple[str, pd.DataFrame, dict]]:
    """Return n valid batch result tuples."""
    return [
        (
            f"file{i}.csv",
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            _minimal_report(),
        )
        for i in range(n)
    ]


def _sample_before_map(n: int = 2) -> dict[str, pd.DataFrame]:
    """Return a before_map matching the filenames in _sample_results(n)."""
    return {
        f"file{i}.csv": pd.DataFrame({"A": [10, 20], "B": [30, 40]})
        for i in range(n)
    }


def _make_minimal_pdf_bytes(n_pages: int) -> bytes:
    """
    Build a syntactically valid minimal PDF with n_pages pages in pure Python.

    Each page has a MediaBox but no content stream — pdfplumber can open it
    without error; extract_tables() and extract_text() return nothing (empty
    page results).  This avoids any dependency on reportlab or other PDF
    libraries in the test suite.

    Structure:
        Object 1  — Catalog
        Object 2  — Pages node  (Kids list of n_pages Page objects)
        Objects 3 … 2+n — individual Page objects
    """
    if n_pages < 1:
        raise ValueError("n_pages must be >= 1")

    # Build object strings ─────────────────────────────────────────────────────
    obj_strings: list[str] = []

    page_refs = " ".join(f"{i} 0 R" for i in range(3, 3 + n_pages))

    # Object 1 — Catalog
    obj_strings.append(
        "1 0 obj\n"
        "<< /Type /Catalog /Pages 2 0 R >>\n"
        "endobj\n"
    )

    # Object 2 — Pages node
    obj_strings.append(
        f"2 0 obj\n"
        f"<< /Type /Pages /Kids [{page_refs}] /Count {n_pages} >>\n"
        f"endobj\n"
    )

    # Objects 3 … 2+n — individual pages
    for page_idx in range(n_pages):
        obj_num = 3 + page_idx
        obj_strings.append(
            f"{obj_num} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 612 792] >>\n"
            f"endobj\n"
        )

    # Assemble body, recording byte offsets for the xref table ─────────────────
    header: bytes = b"%PDF-1.4\n"
    body:   bytes = b""
    offsets: list[int] = []

    for obj_str in obj_strings:
        offsets.append(len(header) + len(body))
        body += obj_str.encode("latin-1")

    n_objects   = len(obj_strings)
    xref_offset = len(header) + len(body)

    # Cross-reference table ────────────────────────────────────────────────────
    xref_lines = [f"xref\n0 {n_objects + 1}"]
    xref_lines.append("0000000000 65535 f ")
    for off in offsets:
        xref_lines.append(f"{off:010d} 00000 n ")
    xref = "\n".join(xref_lines) + "\n"

    trailer = (
        f"trailer\n"
        f"<< /Size {n_objects + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )

    return header + body + xref.encode("latin-1") + trailer.encode("latin-1")


# ═══════════════════════════════════════════════════════════════════════════════
# F01 — build_batch_zip() before_map parameter
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildBatchZipBeforeMap:
    """
    F01 — build_batch_zip() must accept before_map without crashing.

    The original bug: app.py called
        build_batch_zip(_br, build_text_report, before_map=_bm)
    but exporters.py defined only
        def build_batch_zip(results, report_fn)
    causing TypeError on every batch ZIP download.
    """

    # ── Backward compatibility ────────────────────────────────────────────────

    def test_two_arg_call_still_works(self):
        """Original two-argument call must continue to work unchanged."""
        spy, _ = _make_report_fn_spy()
        result = build_batch_zip(_sample_results(), spy)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_explicit_none_before_map_works(self):
        """Passing before_map=None explicitly must behave like omitting it."""
        spy, _ = _make_report_fn_spy()
        result = build_batch_zip(_sample_results(), spy, before_map=None)
        assert isinstance(result, bytes)
        assert len(result) > 0

    # ── New before_map parameter ──────────────────────────────────────────────

    def test_before_map_kwarg_does_not_raise(self):
        """
        The exact app.py call pattern must not raise TypeError.

        This is the direct regression test for F01.
        """
        spy, _ = _make_report_fn_spy()
        # Must not raise TypeError.
        result = build_batch_zip(
            _sample_results(),
            spy,
            before_map=_sample_before_map(),
        )
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_before_map_causes_report_fn_to_receive_four_args(self):
        """
        When before_map is supplied, report_fn must be called with 4 args:
            (report_dict, filename, df_before, df_after)
        so that build_text_report() can include memory statistics.
        """
        spy, calls = _make_report_fn_spy()
        build_batch_zip(
            _sample_results(n=2),
            spy,
            before_map=_sample_before_map(n=2),
        )
        assert len(calls) == 2
        for call in calls:
            args = call["args"]
            assert len(args) == 4, (
                f"Expected 4 args to report_fn, got {len(args)}: {args}"
            )
            report_arg, filename_arg, df_before_arg, df_after_arg = args
            assert isinstance(report_arg,    dict)
            assert isinstance(filename_arg,  str)
            assert isinstance(df_before_arg, pd.DataFrame)
            assert isinstance(df_after_arg,  pd.DataFrame)

    def test_without_before_map_report_fn_receives_two_args(self):
        """
        Without before_map, report_fn must receive exactly 2 args:
            (report_dict, filename)
        preserving the original behavior.
        """
        spy, calls = _make_report_fn_spy()
        build_batch_zip(_sample_results(n=2), spy)
        assert len(calls) == 2
        for call in calls:
            assert len(call["args"]) == 2, (
                f"Expected 2 args, got {len(call['args'])}"
            )

    def test_missing_key_in_before_map_does_not_crash(self):
        """
        If before_map does not contain a key for a filename,
        report_fn must still be called (with df_before=None) and not crash.
        """
        spy, calls = _make_report_fn_spy()
        build_batch_zip(
            _sample_results(n=1),
            spy,
            before_map={},   # empty — no matching key
        )
        assert len(calls) == 1
        args = calls[0]["args"]
        # 4 args expected; third arg (df_before) must be None
        assert len(args) == 4
        assert args[2] is None

    # ── ZIP contents ─────────────────────────────────────────────────────────

    def test_zip_contains_xlsx_csv_and_report_for_each_file(self):
        """ZIP must contain .xlsx, .csv, and report .txt for every result."""
        spy, _ = _make_report_fn_spy()
        zip_bytes = build_batch_zip(
            _sample_results(n=2),
            spy,
            before_map=_sample_before_map(n=2),
        )
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = set(zf.namelist())

        assert "cleaned_file0.xlsx"       in names
        assert "cleaned_file0.csv"        in names
        assert "reports/report_file0.txt" in names
        assert "cleaned_file1.xlsx"       in names
        assert "cleaned_file1.csv"        in names
        assert "reports/report_file1.txt" in names

    def test_zip_no_path_traversal_in_entry_names(self):
        """ZIP archive entries must not contain path-traversal sequences."""
        spy, _ = _make_report_fn_spy()
        evil_results = [
            (
                "../../../etc/passwd.csv",
                pd.DataFrame({"A": [1]}),
                _minimal_report(),
            )
        ]
        zip_bytes = build_batch_zip(evil_results, spy)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                assert ".." not in name, (
                    f"Path traversal sequence in ZIP entry: '{name}'"
                )
                assert not name.startswith("/"), (
                    f"Absolute path in ZIP entry: '{name}'"
                )

    def test_zip_is_valid_zip_file(self):
        """The returned bytes must be a valid ZIP archive."""
        spy, _ = _make_report_fn_spy()
        zip_bytes = build_batch_zip(_sample_results(), spy)
        assert zipfile.is_zipfile(io.BytesIO(zip_bytes))

    # ── Exact app.py call pattern ─────────────────────────────────────────────

    def test_exact_app_py_call_pattern_with_build_text_report(self):
        """
        Reproduce the exact call that was crashing in app.py:

            build_batch_zip(_br, build_text_report, before_map=_bm)

        This test uses the real build_text_report from utils/reports.py.
        """
        from utils.reports import build_text_report

        df_before = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [30, 25]})
        df_after  = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [30, 25]})
        report    = _minimal_report()

        _br = [("customers.csv", df_after, report)]
        _bm = {"customers.csv": df_before}

        # Must not raise TypeError — this is the F01 regression.
        result = build_batch_zip(_br, build_text_report, before_map=_bm)
        assert isinstance(result, bytes)
        assert len(result) > 0

        # Verify the ZIP is valid and contains the expected files.
        with zipfile.ZipFile(io.BytesIO(result)) as zf:
            names = set(zf.namelist())
        assert "cleaned_customers.xlsx"       in names
        assert "cleaned_customers.csv"        in names
        assert "reports/report_customers.txt" in names

    # ── Input validation ──────────────────────────────────────────────────────

    def test_invalid_results_type_raises_type_error(self):
        """Non-list results must raise TypeError."""
        spy, _ = _make_report_fn_spy()
        with pytest.raises(TypeError):
            build_batch_zip("not a list", spy)

    def test_non_callable_report_fn_raises_type_error(self):
        """Non-callable report_fn must raise TypeError."""
        with pytest.raises(TypeError):
            build_batch_zip(_sample_results(), "not callable")

    def test_invalid_before_map_type_raises_type_error(self):
        """before_map that is neither None nor dict must raise TypeError."""
        spy, _ = _make_report_fn_spy()
        with pytest.raises(TypeError):
            build_batch_zip(_sample_results(), spy, before_map="invalid")


# ═══════════════════════════════════════════════════════════════════════════════
# F02 — load_dataframe() row limit
# ═══════════════════════════════════════════════════════════════════════════════

class TestRowLimit:
    """
    F02 — load_dataframe() must enforce MAX_ROWS.

    The original bug: MAX_ROWS_PROFILING existed as a constant but was never
    used in load_dataframe() — a 50 MB file with dense rows could load
    millions of rows into memory unchecked.
    """

    # ── CSV — boundary conditions ─────────────────────────────────────────────

    def test_csv_exactly_max_rows_is_accepted(self):
        """
        A CSV with exactly MAX_ROWS data rows must load without error.
        The row limit is strict greater-than, so MAX_ROWS itself is allowed.
        """
        csv_bytes = _make_csv_bytes(MAX_ROWS)
        df = load_dataframe(csv_bytes, "boundary.csv")
        assert df is not None
        assert len(df) == MAX_ROWS

    def test_csv_max_rows_plus_one_raises_row_limit_exceeded(self):
        """A CSV with MAX_ROWS + 1 rows must raise RowLimitExceeded."""
        csv_bytes = _make_csv_bytes(MAX_ROWS + 1)
        with pytest.raises(RowLimitExceeded):
            load_dataframe(csv_bytes, "toolarge.csv")

    def test_csv_well_within_limit_loads_normally(self):
        """A small CSV must load normally — row limit must not affect it."""
        csv_bytes = _make_csv_bytes(100)
        df = load_dataframe(csv_bytes, "small.csv")
        assert df is not None
        assert len(df) == 100

    # ── RowLimitExceeded attributes ───────────────────────────────────────────

    def test_exception_carries_correct_row_count(self):
        """RowLimitExceeded.row_count must equal the actual parsed row count."""
        nrows     = MAX_ROWS + 1
        csv_bytes = _make_csv_bytes(nrows)
        with pytest.raises(RowLimitExceeded) as exc_info:
            load_dataframe(csv_bytes, "big.csv")
        assert exc_info.value.row_count == nrows

    def test_exception_carries_correct_max_rows(self):
        """RowLimitExceeded.max_rows must equal MAX_ROWS."""
        csv_bytes = _make_csv_bytes(MAX_ROWS + 1)
        with pytest.raises(RowLimitExceeded) as exc_info:
            load_dataframe(csv_bytes, "big.csv")
        assert exc_info.value.max_rows == MAX_ROWS

    def test_exception_carries_filename(self):
        """RowLimitExceeded.filename must contain the filename passed in."""
        csv_bytes = _make_csv_bytes(MAX_ROWS + 1)
        with pytest.raises(RowLimitExceeded) as exc_info:
            load_dataframe(csv_bytes, "mydata.csv")
        assert "mydata.csv" in exc_info.value.filename

    def test_exception_message_is_informative(self):
        """
        str(RowLimitExceeded) must mention both the actual row count
        and the maximum allowed, so callers can display a useful message.
        """
        csv_bytes = _make_csv_bytes(MAX_ROWS + 1)
        with pytest.raises(RowLimitExceeded) as exc_info:
            load_dataframe(csv_bytes, "toobig.csv")
        msg = str(exc_info.value)
        # Must mention actual count in some numeric form.
        assert str(MAX_ROWS + 1) in msg or f"{MAX_ROWS + 1:,}" in msg
        # Must mention the limit in some numeric form.
        assert str(MAX_ROWS) in msg or f"{MAX_ROWS:,}" in msg

    # ── RowLimitExceeded class hierarchy ──────────────────────────────────────

    def test_row_limit_exceeded_is_subclass_of_value_error(self):
        """RowLimitExceeded must inherit from ValueError."""
        assert issubclass(RowLimitExceeded, ValueError)

    def test_row_limit_exceeded_is_subclass_of_exception(self):
        """RowLimitExceeded must be catchable as a generic Exception."""
        csv_bytes = _make_csv_bytes(MAX_ROWS + 1)
        caught = False
        try:
            load_dataframe(csv_bytes, "x.csv")
        except Exception:
            caught = True
        assert caught, "RowLimitExceeded was not caught by 'except Exception'"

    def test_row_limit_exceeded_not_swallowed_by_load_dataframe(self):
        """
        load_dataframe() must NOT return None for an over-limit file.
        It must propagate RowLimitExceeded so callers can distinguish
        a row-limit rejection from a genuine parse failure.
        """
        csv_bytes = _make_csv_bytes(MAX_ROWS + 1)
        result_was_none = False
        try:
            result = load_dataframe(csv_bytes, "big.csv")
            if result is None:
                result_was_none = True
        except RowLimitExceeded:
            pass  # correct behavior

        assert not result_was_none, (
            "load_dataframe() returned None for an over-limit file. "
            "It must raise RowLimitExceeded instead so callers can show "
            "a specific error message."
        )

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_empty_csv_returns_none_not_exception(self):
        """
        A CSV with a header row but no data rows must return None,
        not raise RowLimitExceeded.  0 rows is always within the limit.
        """
        csv_bytes = b"col1,col2,col3\n"
        result = load_dataframe(csv_bytes, "empty.csv")
        assert result is None

    def test_parse_failure_returns_none_not_row_limit_exception(self):
        """
        Completely invalid bytes must return None (parse failure),
        not RowLimitExceeded.  The two failure modes must remain distinct.
        """
        result = load_dataframe(b"this is not csv or excel data !@#$", "bad.csv")
        # Should be None (parse failure) or raise some non-RowLimitExceeded exception.
        # The important thing is it must NOT raise RowLimitExceeded.
        # We check this by ensuring RowLimitExceeded is not raised.
        # (If it raises another exception, the test fails — which is also useful.)
        assert result is None

    # ── Excel ─────────────────────────────────────────────────────────────────

    def test_excel_within_limit_loads_normally(self):
        """A small .xlsx file must load normally."""
        excel_bytes = _make_excel_bytes(50)
        df = load_dataframe(excel_bytes, "small.xlsx")
        assert df is not None
        assert len(df) == 50

    def test_excel_over_limit_raises_row_limit_exceeded(self):
        """
        An .xlsx file exceeding MAX_ROWS must raise RowLimitExceeded.

        To keep the test fast, we patch MAX_ROWS to a small value
        (10 rows) so we do not need to generate a 500,000-row Excel file.
        """
        import unittest.mock as mock

        small_limit  = 10
        excel_bytes  = _make_excel_bytes(small_limit + 1)

        with mock.patch("utils.helpers.MAX_ROWS", small_limit):
            with pytest.raises(RowLimitExceeded) as exc_info:
                load_dataframe(excel_bytes, "toolarge.xlsx")

        assert exc_info.value.row_count == small_limit + 1


# ═══════════════════════════════════════════════════════════════════════════════
# F03 — extract_pdf_tables() page limit
# ═══════════════════════════════════════════════════════════════════════════════

class TestPdfPageLimit:
    """
    F03 — extract_pdf_tables() must enforce MAX_PDF_PAGES in "all" mode.

    The original bug: when page_selection == "all", the function did
        pages_idx = list(range(total_pages))
    with no upper bound, allowing a 10,000-page PDF to process every page.
    """

    # ── "all" mode — within limit ─────────────────────────────────────────────

    def test_single_page_pdf_processes_normally(self):
        """A 1-page PDF must not be rejected."""
        pdf_bytes = _make_minimal_pdf_bytes(1)
        results   = extract_pdf_tables(pdf_bytes, "one.pdf", page_selection="all")
        limit_hits = [r for r in results if r.get("method") == "page_limit_exceeded"]
        assert len(limit_hits) == 0

    def test_pdf_exactly_at_max_pages_processes_normally(self):
        """
        A PDF with exactly MAX_PDF_PAGES pages must not be rejected.
        The limit fires only when total_pages > MAX_PDF_PAGES.
        """
        pdf_bytes = _make_minimal_pdf_bytes(MAX_PDF_PAGES)
        results   = extract_pdf_tables(pdf_bytes, "atmax.pdf", page_selection="all")
        limit_hits = [r for r in results if r.get("method") == "page_limit_exceeded"]
        assert len(limit_hits) == 0, (
            f"A {MAX_PDF_PAGES}-page PDF must not trigger page_limit_exceeded"
        )

    # ── "all" mode — over limit ───────────────────────────────────────────────

    def test_pdf_one_over_max_pages_is_rejected(self):
        """
        A PDF with MAX_PDF_PAGES + 1 pages must be rejected in "all" mode.
        Results must contain exactly one page_limit_exceeded entry.
        """
        pdf_bytes = _make_minimal_pdf_bytes(MAX_PDF_PAGES + 1)
        results   = extract_pdf_tables(pdf_bytes, "toolarge.pdf", page_selection="all")
        limit_hits = [r for r in results if r.get("method") == "page_limit_exceeded"]
        assert len(limit_hits) == 1, (
            f"Expected 1 page_limit_exceeded result, got {len(limit_hits)}"
        )

    def test_page_limit_result_carries_total_pages(self):
        """
        The page_limit_exceeded result must carry total_pages so the caller
        can display a message like 'This PDF has 347 pages'.
        """
        n_pages   = MAX_PDF_PAGES + 50
        pdf_bytes = _make_minimal_pdf_bytes(n_pages)
        results   = extract_pdf_tables(pdf_bytes, "big.pdf", page_selection="all")
        limit_r   = next(
            (r for r in results if r.get("method") == "page_limit_exceeded"),
            None,
        )
        assert limit_r is not None
        assert limit_r["total_pages"] == n_pages

    def test_page_limit_result_carries_max_pages(self):
        """
        The page_limit_exceeded result must carry max_pages (== MAX_PDF_PAGES)
        so the caller can display a message like 'maximum is 100 pages'.
        """
        pdf_bytes = _make_minimal_pdf_bytes(MAX_PDF_PAGES + 1)
        results   = extract_pdf_tables(pdf_bytes, "big.pdf", page_selection="all")
        limit_r   = next(
            (r for r in results if r.get("method") == "page_limit_exceeded"),
            None,
        )
        assert limit_r is not None
        assert limit_r["max_pages"] == MAX_PDF_PAGES

    def test_page_limit_result_has_none_dataframe(self):
        """page_limit_exceeded result must have dataframe == None."""
        pdf_bytes = _make_minimal_pdf_bytes(MAX_PDF_PAGES + 1)
        results   = extract_pdf_tables(pdf_bytes, "big.pdf", page_selection="all")
        limit_r   = next(
            r for r in results if r.get("method") == "page_limit_exceeded"
        )
        assert limit_r["dataframe"] is None

    def test_no_pages_processed_when_limit_exceeded(self):
        """
        When the limit is exceeded, zero page-level results must be returned.
        The function must reject immediately — not partially process.
        """
        pdf_bytes    = _make_minimal_pdf_bytes(MAX_PDF_PAGES + 1)
        results      = extract_pdf_tables(pdf_bytes, "big.pdf", page_selection="all")
        page_results = [r for r in results if r.get("page") is not None]
        assert len(page_results) == 0, (
            f"Expected 0 page results after limit rejection, "
            f"got {len(page_results)}: {page_results}"
        )

    # ── "specific" mode — not subject to MAX_PDF_PAGES ───────────────────────

    def test_specific_mode_not_limited_by_max_pdf_pages(self):
        """
        Specific-page mode must NOT be subject to MAX_PDF_PAGES.
        Requesting page 1 of a (MAX_PDF_PAGES + 100)-page PDF must work.
        """
        n_pages   = MAX_PDF_PAGES + 100
        pdf_bytes = _make_minimal_pdf_bytes(n_pages)
        results   = extract_pdf_tables(
            pdf_bytes,
            "large.pdf",
            page_selection="specific",
            specific_pages=[1],
        )
        limit_hits = [r for r in results if r.get("method") == "page_limit_exceeded"]
        assert len(limit_hits) == 0, (
            "page_limit_exceeded must not appear in specific-page mode"
        )

    def test_specific_mode_out_of_range_pages_silently_ignored(self):
        """
        Out-of-range page numbers in specific mode must be silently ignored.
        No IndexError, no crash, no page_limit_exceeded.
        """
        pdf_bytes = _make_minimal_pdf_bytes(5)
        results   = extract_pdf_tables(
            pdf_bytes,
            "small.pdf",
            page_selection="specific",
            specific_pages=[999, 10000],
        )
        # All out-of-range → empty results list.
        assert results == []

    def test_specific_mode_mixed_valid_and_invalid_pages(self):
        """
        A mix of valid and out-of-range specific pages must not crash
        and must not produce a page_limit_exceeded entry.
        """
        pdf_bytes = _make_minimal_pdf_bytes(5)
        results   = extract_pdf_tables(
            pdf_bytes,
            "small.pdf",
            page_selection="specific",
            specific_pages=[1, 999],   # page 1 valid, 999 invalid
        )
        limit_hits = [r for r in results if r.get("method") == "page_limit_exceeded"]
        assert len(limit_hits) == 0

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_empty_bytes_returns_empty_list(self):
        """Empty bytes must return an empty list without raising."""
        results = extract_pdf_tables(b"", "empty.pdf")
        assert results == []

    def test_non_pdf_bytes_returns_list_without_raising(self):
        """
        Completely invalid bytes must return a list (possibly containing an
        error entry) without raising an unhandled exception.
        """
        results = extract_pdf_tables(b"this is not a pdf", "fake.pdf")
        assert isinstance(results, list)

    def test_non_bytes_input_returns_empty_list(self):
        """Non-bytes input must return an empty list without raising."""
        results = extract_pdf_tables("not bytes", "bad.pdf")  # type: ignore[arg-type]
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════════
# F05 — _count_changed_values import correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestF05ImportCorrectness:
    """
    F05 — Verifies the corrected import in test_cleaning.py.

    The original bug: test_cleaning.py imported _count_encoding_repairs,
    which does not exist in cleaning.py.  The correct name is
    _count_changed_values.  This caused an ImportError at pytest collection
    time, preventing the entire test_cleaning.py file from loading.

    These tests verify:
    1. _count_changed_values exists and is callable.
    2. _count_encoding_repairs does NOT exist (no accidental alias added).
    3. test_cleaning.py can be imported without ImportError.
    """

    def test_count_changed_values_is_importable(self):
        """_count_changed_values must be importable from utils.cleaning."""
        from utils.cleaning import _count_changed_values
        assert callable(_count_changed_values)

    def test_count_changed_values_returns_int(self):
        """_count_changed_values must return an integer."""
        from utils.cleaning import _count_changed_values
        s_orig  = pd.Series(["a", "b", "c"])
        s_fixed = pd.Series(["a", "B", "c"])   # one value changed
        result  = _count_changed_values(s_orig, s_fixed)
        assert isinstance(result, int)

    def test_count_changed_values_counts_correctly(self):
        """_count_changed_values must return the number of changed cells."""
        from utils.cleaning import _count_changed_values
        s_orig  = pd.Series(["hello", "world", "foo"])
        s_fixed = pd.Series(["hello", "WORLD", "FOO"])   # 2 changed
        result  = _count_changed_values(s_orig, s_fixed)
        assert result == 2

    def test_count_changed_values_zero_for_identical_series(self):
        """_count_changed_values must return 0 when both series are identical."""
        from utils.cleaning import _count_changed_values
        s = pd.Series(["x", "y", "z"])
        assert _count_changed_values(s, s.copy()) == 0

    def test_count_changed_values_handles_null_cells(self):
        """Null-to-null transitions must not be counted as changes."""
        from utils.cleaning import _count_changed_values
        s_orig  = pd.Series([None, float("nan"), pd.NA])
        s_fixed = pd.Series([None, float("nan"), pd.NA])
        result  = _count_changed_values(s_orig, s_fixed)
        assert result == 0

    def test_count_encoding_repairs_does_not_exist_in_cleaning(self):
        """
        _count_encoding_repairs must NOT exist in utils.cleaning.
        If it does, someone added a compatibility alias — which would
        hide the F05 bug rather than fixing it.
        """
        import utils.cleaning as cleaning_module
        assert not hasattr(cleaning_module, "_count_encoding_repairs"), (
            "_count_encoding_repairs should not exist in utils/cleaning.py. "
            "The correct name is _count_changed_values. "
            "Do not add a compatibility alias — fix the import in test_cleaning.py."
        )

    def test_test_cleaning_module_is_importable_without_error(self):
        """
        tests/test_cleaning.py must be importable without raising ImportError.

        If _count_encoding_repairs has not been replaced with
        _count_changed_values, this test will fail with:
            ImportError: cannot import name '_count_encoding_repairs'
            from 'utils.cleaning'

        That failure is the exact symptom of F05.
        """
        test_cleaning_path = os.path.join(
            _REPO_ROOT, "tests", "test_cleaning.py"
        )

        if not os.path.exists(test_cleaning_path):
            pytest.skip(
                "tests/test_cleaning.py not found — "
                "skipping importability check."
            )

        spec   = importlib.util.spec_from_file_location(
            "test_cleaning_import_check",
            test_cleaning_path,
        )
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
        except ImportError as exc:
            pytest.fail(
                f"tests/test_cleaning.py raised ImportError: {exc}\n"
                f"This is the F05 regression. "
                f"Replace '_count_encoding_repairs' with '_count_changed_values' "
                f"in the import block of tests/test_cleaning.py."
            )
