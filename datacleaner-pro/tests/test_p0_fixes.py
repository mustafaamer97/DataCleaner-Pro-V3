"""
tests/test_p0_fixes.py
======================
Regression tests for DataCleaner Pro V3 — P0 fixes.

Covers:
    F01 — build_batch_zip() accepts before_map parameter
    F02 — load_dataframe() enforces MAX_ROWS / raises RowLimitExceeded
    F03 — extract_pdf_tables() enforces MAX_PDF_PAGES in "all" mode
    F05 — _count_encoding_repairs and _count_changed_values both importable

Run from repository root:
    pytest tests/ -v --tb=short
"""

from __future__ import annotations

import io
import sys
import os
import zipfile
import importlib
import importlib.util

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import pytest

from utils.helpers      import MAX_ROWS, RowLimitExceeded, load_dataframe
from utils.exporters    import build_batch_zip
from utils.pdf_processor import MAX_PDF_PAGES, extract_pdf_tables


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_csv_bytes(nrows: int, ncols: int = 3) -> bytes:
    """Return UTF-8 CSV bytes with exactly nrows data rows."""
    header = ",".join(f"col{c}" for c in range(ncols))
    lines  = [header]
    for r in range(nrows):
        lines.append(",".join(str(r * ncols + c) for c in range(ncols)))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _make_excel_bytes(nrows: int, ncols: int = 2) -> bytes:
    """Return .xlsx bytes with exactly nrows data rows."""
    buf = io.BytesIO()
    df  = pd.DataFrame({f"col{c}": range(nrows) for c in range(ncols)})
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


def _make_report_fn_spy():
    """Return (spy_fn, calls_list)."""
    calls: list[dict] = []

    def _spy(*args):
        calls.append({"args": args})
        return "REPORT TEXT"

    return _spy, calls


def _minimal_report() -> dict:
    """Minimal report dict matching run_cleaning_pipeline() output."""
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
    return [
        (
            f"file{i}.csv",
            pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            _minimal_report(),
        )
        for i in range(n)
    ]


def _sample_before_map(n: int = 2) -> dict[str, pd.DataFrame]:
    return {
        f"file{i}.csv": pd.DataFrame({"A": [10, 20], "B": [30, 40]})
        for i in range(n)
    }


def _make_minimal_pdf_bytes(n_pages: int) -> bytes:
    """Build a syntactically valid minimal PDF with n_pages pages."""
    if n_pages < 1:
        raise ValueError("n_pages must be >= 1")

    obj_strings: list[str] = []
    page_refs = " ".join(f"{i} 0 R" for i in range(3, 3 + n_pages))

    obj_strings.append(
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    )
    obj_strings.append(
        f"2 0 obj\n"
        f"<< /Type /Pages /Kids [{page_refs}] /Count {n_pages} >>\n"
        f"endobj\n"
    )
    for page_idx in range(n_pages):
        obj_num = 3 + page_idx
        obj_strings.append(
            f"{obj_num} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\n"
            f"endobj\n"
        )

    header:  bytes = b"%PDF-1.4\n"
    body:    bytes = b""
    offsets: list[int] = []
    for obj_str in obj_strings:
        offsets.append(len(header) + len(body))
        body += obj_str.encode("latin-1")

    n_objects   = len(obj_strings)
    xref_offset = len(header) + len(body)

    xref_lines = [f"xref\n0 {n_objects + 1}"]
    xref_lines.append("0000000000 65535 f ")
    for off in offsets:
        xref_lines.append(f"{off:010d} 00000 n ")
    xref = "\n".join(xref_lines) + "\n"

    trailer = (
        f"trailer\n<< /Size {n_objects + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    return header + body + xref.encode("latin-1") + trailer.encode("latin-1")


# ═══════════════════════════════════════════════════════════════════════════════
# F01 — build_batch_zip() before_map parameter
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildBatchZipBeforeMap:
    """F01 — build_batch_zip() must accept before_map without crashing."""

    def test_two_arg_call_still_works(self):
        spy, _ = _make_report_fn_spy()
        result = build_batch_zip(_sample_results(), spy)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_explicit_none_before_map_works(self):
        spy, _ = _make_report_fn_spy()
        result = build_batch_zip(_sample_results(), spy, before_map=None)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_before_map_kwarg_does_not_raise(self):
        """Direct regression test for F01 — must not raise TypeError."""
        spy, _ = _make_report_fn_spy()
        result = build_batch_zip(
            _sample_results(),
            spy,
            before_map=_sample_before_map(),
        )
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_before_map_causes_report_fn_to_receive_four_args(self):
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
        spy, calls = _make_report_fn_spy()
        build_batch_zip(_sample_results(n=2), spy)
        assert len(calls) == 2
        for call in calls:
            assert len(call["args"]) == 2

    def test_missing_key_in_before_map_does_not_crash(self):
        spy, calls = _make_report_fn_spy()
        build_batch_zip(
            _sample_results(n=1),
            spy,
            before_map={},
        )
        assert len(calls) == 1
        args = calls[0]["args"]
        assert len(args) == 4
        assert args[2] is None

    def test_zip_contains_xlsx_csv_and_report_for_each_file(self):
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
                assert ".." not in name
                assert not name.startswith("/")

    def test_zip_is_valid_zip_file(self):
        spy, _ = _make_report_fn_spy()
        zip_bytes = build_batch_zip(_sample_results(), spy)
        assert zipfile.is_zipfile(io.BytesIO(zip_bytes))

    def test_exact_app_py_call_pattern_with_build_text_report(self):
        """Reproduces the exact app.py call that was crashing (F01)."""
        from utils.reports import build_text_report

        df_before = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [30, 25]})
        df_after  = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [30, 25]})
        report    = _minimal_report()

        _br = [("customers.csv", df_after, report)]
        _bm = {"customers.csv": df_before}

        result = build_batch_zip(_br, build_text_report, before_map=_bm)
        assert isinstance(result, bytes)
        assert len(result) > 0

        with zipfile.ZipFile(io.BytesIO(result)) as zf:
            names = set(zf.namelist())
        assert "cleaned_customers.xlsx"       in names
        assert "cleaned_customers.csv"        in names
        assert "reports/report_customers.txt" in names

    def test_invalid_results_type_raises_type_error(self):
        spy, _ = _make_report_fn_spy()
        with pytest.raises(TypeError):
            build_batch_zip("not a list", spy)

    def test_non_callable_report_fn_raises_type_error(self):
        with pytest.raises(TypeError):
            build_batch_zip(_sample_results(), "not callable")

    def test_invalid_before_map_type_raises_type_error(self):
        spy, _ = _make_report_fn_spy()
        with pytest.raises(TypeError):
            build_batch_zip(_sample_results(), spy, before_map="invalid")


# ═══════════════════════════════════════════════════════════════════════════════
# F02 — load_dataframe() row limit
# ═══════════════════════════════════════════════════════════════════════════════

class TestRowLimit:
    """F02 — load_dataframe() must enforce MAX_ROWS."""

    def test_csv_exactly_max_rows_is_accepted(self):
        csv_bytes = _make_csv_bytes(MAX_ROWS)
        df = load_dataframe(csv_bytes, "boundary.csv")
        assert df is not None
        assert len(df) == MAX_ROWS

    def test_csv_max_rows_plus_one_raises_row_limit_exceeded(self):
        csv_bytes = _make_csv_bytes(MAX_ROWS + 1)
        with pytest.raises(RowLimitExceeded):
            load_dataframe(csv_bytes, "toolarge.csv")

    def test_csv_well_within_limit_loads_normally(self):
        csv_bytes = _make_csv_bytes(100)
        df = load_dataframe(csv_bytes, "small.csv")
        assert df is not None
        assert len(df) == 100

    def test_exception_carries_correct_row_count(self):
        nrows     = MAX_ROWS + 1
        csv_bytes = _make_csv_bytes(nrows)
        with pytest.raises(RowLimitExceeded) as exc_info:
            load_dataframe(csv_bytes, "big.csv")
        assert exc_info.value.row_count == nrows

    def test_exception_carries_correct_max_rows(self):
        csv_bytes = _make_csv_bytes(MAX_ROWS + 1)
        with pytest.raises(RowLimitExceeded) as exc_info:
            load_dataframe(csv_bytes, "big.csv")
        assert exc_info.value.max_rows == MAX_ROWS

    def test_exception_carries_filename(self):
        csv_bytes = _make_csv_bytes(MAX_ROWS + 1)
        with pytest.raises(RowLimitExceeded) as exc_info:
            load_dataframe(csv_bytes, "mydata.csv")
        assert "mydata.csv" in exc_info.value.filename

    def test_exception_message_is_informative(self):
        csv_bytes = _make_csv_bytes(MAX_ROWS + 1)
        with pytest.raises(RowLimitExceeded) as exc_info:
            load_dataframe(csv_bytes, "toobig.csv")
        msg = str(exc_info.value)
        assert str(MAX_ROWS + 1) in msg or f"{MAX_ROWS + 1:,}" in msg
        assert str(MAX_ROWS) in msg or f"{MAX_ROWS:,}" in msg

    def test_row_limit_exceeded_is_subclass_of_value_error(self):
        assert issubclass(RowLimitExceeded, ValueError)

    def test_row_limit_exceeded_is_subclass_of_exception(self):
        csv_bytes = _make_csv_bytes(MAX_ROWS + 1)
        caught = False
        try:
            load_dataframe(csv_bytes, "x.csv")
        except Exception:
            caught = True
        assert caught

    def test_row_limit_exceeded_not_swallowed_by_load_dataframe(self):
        csv_bytes = _make_csv_bytes(MAX_ROWS + 1)
        result_was_none = False
        try:
            result = load_dataframe(csv_bytes, "big.csv")
            if result is None:
                result_was_none = True
        except RowLimitExceeded:
            pass
        assert not result_was_none

    def test_empty_csv_returns_none_not_exception(self):
        csv_bytes = b"col1,col2,col3\n"
        result = load_dataframe(csv_bytes, "empty.csv")
        assert result is None

    def test_parse_failure_returns_none_not_row_limit_exception(self):
        result = load_dataframe(b"this is not csv or excel data !@#$", "bad.csv")
        assert result is None

    def test_excel_within_limit_loads_normally(self):
        excel_bytes = _make_excel_bytes(50)
        df = load_dataframe(excel_bytes, "small.xlsx")
        assert df is not None
        assert len(df) == 50

    def test_excel_over_limit_raises_row_limit_exceeded(self):
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
    """F03 — extract_pdf_tables() must enforce MAX_PDF_PAGES in "all" mode."""

    def test_single_page_pdf_processes_normally(self):
        pdf_bytes = _make_minimal_pdf_bytes(1)
        results   = extract_pdf_tables(pdf_bytes, "one.pdf", page_selection="all")
        limit_hits = [r for r in results if r.get("method") == "page_limit_exceeded"]
        assert len(limit_hits) == 0

    def test_pdf_exactly_at_max_pages_processes_normally(self):
        pdf_bytes = _make_minimal_pdf_bytes(MAX_PDF_PAGES)
        results   = extract_pdf_tables(pdf_bytes, "atmax.pdf", page_selection="all")
        limit_hits = [r for r in results if r.get("method") == "page_limit_exceeded"]
        assert len(limit_hits) == 0

    def test_pdf_one_over_max_pages_is_rejected(self):
        pdf_bytes = _make_minimal_pdf_bytes(MAX_PDF_PAGES + 1)
        results   = extract_pdf_tables(pdf_bytes, "toolarge.pdf", page_selection="all")
        limit_hits = [r for r in results if r.get("method") == "page_limit_exceeded"]
        assert len(limit_hits) == 1

    def test_page_limit_result_carries_total_pages(self):
        n_pages   = MAX_PDF_PAGES + 50
        pdf_bytes = _make_minimal_pdf_bytes(n_pages)
        results   = extract_pdf_tables(pdf_bytes, "big.pdf", page_selection="all")
        limit_r   = next(
            (r for r in results if r.get("method") == "page_limit_exceeded"), None
        )
        assert limit_r is not None
        assert limit_r["total_pages"] == n_pages

    def test_page_limit_result_carries_max_pages(self):
        pdf_bytes = _make_minimal_pdf_bytes(MAX_PDF_PAGES + 1)
        results   = extract_pdf_tables(pdf_bytes, "big.pdf", page_selection="all")
        limit_r   = next(
            r for r in results if r.get("method") == "page_limit_exceeded"
        )
        assert limit_r["max_pages"] == MAX_PDF_PAGES

    def test_page_limit_result_has_none_dataframe(self):
        pdf_bytes = _make_minimal_pdf_bytes(MAX_PDF_PAGES + 1)
        results   = extract_pdf_tables(pdf_bytes, "big.pdf", page_selection="all")
        limit_r   = next(
            r for r in results if r.get("method") == "page_limit_exceeded"
        )
        assert limit_r["dataframe"] is None

    def test_no_pages_processed_when_limit_exceeded(self):
        pdf_bytes    = _make_minimal_pdf_bytes(MAX_PDF_PAGES + 1)
        results      = extract_pdf_tables(pdf_bytes, "big.pdf", page_selection="all")
        page_results = [r for r in results if r.get("page") is not None]
        assert len(page_results) == 0

    def test_specific_mode_not_limited_by_max_pdf_pages(self):
        n_pages   = MAX_PDF_PAGES + 100
        pdf_bytes = _make_minimal_pdf_bytes(n_pages)
        results   = extract_pdf_tables(
            pdf_bytes, "large.pdf",
            page_selection="specific",
            specific_pages=[1],
        )
        limit_hits = [r for r in results if r.get("method") == "page_limit_exceeded"]
        assert len(limit_hits) == 0

    def test_specific_mode_out_of_range_pages_silently_ignored(self):
        pdf_bytes = _make_minimal_pdf_bytes(5)
        results   = extract_pdf_tables(
            pdf_bytes, "small.pdf",
            page_selection="specific",
            specific_pages=[999, 10000],
        )
        assert results == []

    def test_specific_mode_mixed_valid_and_invalid_pages(self):
        pdf_bytes = _make_minimal_pdf_bytes(5)
        results   = extract_pdf_tables(
            pdf_bytes, "small.pdf",
            page_selection="specific",
            specific_pages=[1, 999],
        )
        limit_hits = [r for r in results if r.get("method") == "page_limit_exceeded"]
        assert len(limit_hits) == 0

    def test_empty_bytes_returns_empty_list(self):
        results = extract_pdf_tables(b"", "empty.pdf")
        assert results == []

    def test_non_pdf_bytes_returns_list_without_raising(self):
        results = extract_pdf_tables(b"this is not a pdf", "fake.pdf")
        assert isinstance(results, list)

    def test_non_bytes_input_returns_empty_list(self):
        results = extract_pdf_tables("not bytes", "bad.pdf")  # type: ignore[arg-type]
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════════
# F05 — _count_encoding_repairs / _count_changed_values
# ═══════════════════════════════════════════════════════════════════════════════

class TestF05ImportCorrectness:
    """
    F05 — Both _count_encoding_repairs and _count_changed_values must exist
    in utils.cleaning.

    test_cleaning.py imports _count_encoding_repairs.
    The implementation function is _count_changed_values.
    Both must exist; _count_encoding_repairs must be an alias.
    """

    def test_count_changed_values_is_importable(self):
        from utils.cleaning import _count_changed_values
        assert callable(_count_changed_values)

    def test_count_encoding_repairs_is_importable(self):
        """test_cleaning.py imports this name — it must exist."""
        from utils.cleaning import _count_encoding_repairs
        assert callable(_count_encoding_repairs)

    def test_count_encoding_repairs_is_alias_for_count_changed_values(self):
        """Both names must refer to the same underlying function."""
        import utils.cleaning as cleaning_module
        assert hasattr(cleaning_module, "_count_encoding_repairs")
        assert hasattr(cleaning_module, "_count_changed_values")
        assert (
            cleaning_module._count_encoding_repairs
            is cleaning_module._count_changed_values
        ), (
            "_count_encoding_repairs must be an alias for _count_changed_values, "
            "not a separate function."
        )

    def test_count_changed_values_returns_int(self):
        from utils.cleaning import _count_changed_values
        s_orig  = pd.Series(["a", "b", "c"])
        s_fixed = pd.Series(["a", "B", "c"])
        result  = _count_changed_values(s_orig, s_fixed)
        assert isinstance(result, int)

    def test_count_changed_values_counts_correctly(self):
        from utils.cleaning import _count_changed_values
        s_orig  = pd.Series(["hello", "world", "foo"])
        s_fixed = pd.Series(["hello", "WORLD", "FOO"])
        assert _count_changed_values(s_orig, s_fixed) == 2

    def test_count_changed_values_zero_for_identical_series(self):
        from utils.cleaning import _count_changed_values
        s = pd.Series(["x", "y", "z"])
        assert _count_changed_values(s, s.copy()) == 0

    def test_count_changed_values_handles_null_cells(self):
        from utils.cleaning import _count_changed_values
        s_orig  = pd.Series([None, float("nan"), pd.NA])
        s_fixed = pd.Series([None, float("nan"), pd.NA])
        assert _count_changed_values(s_orig, s_fixed) == 0

    def test_test_cleaning_module_is_importable_without_error(self):
        """tests/test_cleaning.py must import without ImportError."""
        test_cleaning_path = os.path.join(_REPO_ROOT, "tests", "test_cleaning.py")

        if not os.path.exists(test_cleaning_path):
            pytest.skip("tests/test_cleaning.py not found.")

        spec   = importlib.util.spec_from_file_location(
            "test_cleaning_import_check", test_cleaning_path
        )
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ImportError as exc:
            pytest.fail(
                f"tests/test_cleaning.py raised ImportError: {exc}\n"
                "Ensure _count_encoding_repairs exists in utils/cleaning.py."
            )
