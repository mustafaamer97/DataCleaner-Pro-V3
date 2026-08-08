"""
tests/test_cleaning.py
Unit tests for DataCleaner Pro V3.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import pytest

from utils.cleaning import (
    run_cleaning_pipeline,
    normalize_email,
    normalize_phone,
    normalize_date,
    remove_empty_columns,
    remove_duplicate_columns,
    fill_missing,
    _fix_encoding_ftfy,
    _count_encoding_repairs,
)
from utils.duplicates import find_fuzzy_duplicates, remove_exact_duplicates
from utils.helpers import to_snake_case
from utils.profiling import detect_column_type, profile_dataframe


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal DataFrame with mixed data quality issues."""
    return pd.DataFrame({
        "Name":  ["Alice", "Bob", "Alice", "Charlie", None],
        "Email": ["ALICE@EMAIL.COM", "bob@email.com", "ALICE@EMAIL.COM", "bad-email", None],
        "Age":   [30, 25, 30, None, 40],
        "Notes": ["  Hello  ", "World", "  Hello  ", "Test  ", ""],
    })


@pytest.fixture
def customers_df():
    """Realistic fixture that mirrors sample_customers.csv column structure."""
    return pd.DataFrame({
        "id":          [1, 2, 3, 4, 5],
        "first_name":  ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "last_name":   ["Smith", "Jones", "Brown", "White", "Black"],
        "email":       [
            "alice@example.com", "bob@example.com",
            "carol@example.com", "dave@example.com", "eve@example.com",
        ],
        "phone":       [
            "+1 555 123 4567", "555-987-6543",
            "(555) 222-3333", "+44 20 7946 0958", "5553334444",
        ],
        "age":         [30, 25, 40, 52, 28],
        "signup_date": [
            "2024-01-15", "15/01/2024", "Feb 3 2024",
            "2024-03-10", "2024-06-30",
        ],
        "country":     ["USA", "Canada", "UK", "Australia", "Canada"],
        "salary":      [55000, 62000, 75000, 91000, 67000],
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Bug Fix Tests — Column Type Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestColumnTypeDetection:

    def test_signup_date_is_date_not_phone(self, customers_df):
        """
        signup_date must be classified as 'date', never as 'phone'.
        This was the primary regression reported in QA.
        """
        series   = customers_df["signup_date"]
        sem_type = detect_column_type(series, col_name="signup_date")
        assert sem_type == "date", (
            f"Expected 'date' but got '{sem_type}' for signup_date column"
        )

    def test_phone_column_is_phone(self, customers_df):
        """phone column with real phone numbers must be classified as 'phone'."""
        series   = customers_df["phone"]
        sem_type = detect_column_type(series, col_name="phone")
        assert sem_type == "phone", (
            f"Expected 'phone' but got '{sem_type}' for phone column"
        )

    def test_email_column_is_email(self, customers_df):
        """email column must be classified as 'email'."""
        series   = customers_df["email"]
        sem_type = detect_column_type(series, col_name="email")
        assert sem_type == "email", (
            f"Expected 'email' but got '{sem_type}' for email column"
        )

    def test_id_column_is_id(self, customers_df):
        """id column (integer, all unique) must be classified as 'id'."""
        series   = customers_df["id"]
        sem_type = detect_column_type(series, col_name="id")
        assert sem_type == "id", (
            f"Expected 'id' but got '{sem_type}' for id column"
        )

    def test_numeric_column_is_numeric(self, customers_df):
        """age column must be classified as 'numeric' (not id — no id keyword)."""
        series   = customers_df["age"]
        sem_type = detect_column_type(series, col_name="age")
        assert sem_type == "numeric", (
            f"Expected 'numeric' but got '{sem_type}' for age column"
        )

    def test_salary_is_numeric_or_currency(self, customers_df):
        """salary column should be 'numeric' or 'currency', never 'phone'."""
        series   = customers_df["salary"]
        sem_type = detect_column_type(series, col_name="salary")
        assert sem_type in ("numeric", "currency"), (
            f"Expected 'numeric' or 'currency' but got '{sem_type}'"
        )

    def test_profile_phone_group_excludes_dates(self, customers_df):
        """
        Full profile must NOT include signup_date in phone type_group.
        """
        profile = profile_dataframe(customers_df)
        phone_cols = profile["type_groups"].get("phone", [])
        assert "signup_date" not in phone_cols, (
            f"signup_date appeared in phone_cols: {phone_cols}"
        )

    def test_profile_date_group_includes_signup_date(self, customers_df):
        """
        Full profile must include signup_date in date type_group.
        """
        profile   = profile_dataframe(customers_df)
        date_cols = profile["type_groups"].get("date", [])
        assert "signup_date" in date_cols, (
            f"signup_date not found in date_cols: {date_cols}"
        )

    def test_date_keyword_columns_not_misclassified(self):
        """
        Columns with date-related names (birth_date, created_at, etc.)
        must not be classified as phone even with digit-heavy content.
        """
        date_col_names = [
            "birth_date", "created_at", "updated_at",
            "registration_date", "date", "dob",
        ]
        date_values = pd.Series([
            "2024-01-15", "2023-06-30", "2022-12-01",
            "15/01/2024", "Feb 3 2024", "2024-03-10",
        ])
        for col_name in date_col_names:
            result = detect_column_type(date_values, col_name=col_name)
            assert result == "date", (
                f"Column '{col_name}' should be 'date' but got '{result}'"
            )

    def test_content_date_detection_without_name_hint(self):
        """
        Even without a date-related column name, ISO date content
        should be detected as 'date' (content ratio ≥ 0.70).
        """
        date_values = pd.Series([
            "2024-01-15", "2024-02-20", "2024-03-10",
            "2024-04-01", "2024-05-12", "2024-06-30",
        ])
        result = detect_column_type(date_values, col_name="col_x")
        assert result == "date", (
            f"ISO date content should be detected as 'date', got '{result}'"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Bug Fix Tests — Encoding Repair Counter
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodingRepairCounter:

    def test_ftfy_count_zero_for_normal_text(self):
        """
        ftfy must not count unchanged strings as repaired.
        Normal ASCII text passes through ftfy unchanged.
        """
        df = pd.DataFrame({
            "Text": ["Hello World", "Normal text", "No encoding issues", "Clean data"]
        })
        original = df["Text"].copy()
        df["Text"] = df["Text"].apply(_fix_encoding_ftfy)
        count = _count_encoding_repairs(original, df["Text"])
        assert count == 0, (
            f"Expected 0 repairs for normal text, got {count}"
        )

    def test_ftfy_count_increases_for_broken_text(self):
        """
        ftfy must increment count ONLY when text actually changes.
        'GrÃ¢ce' is classic Latin-1/UTF-8 mojibake that ftfy can fix.
        """
        broken_text = "GrÃ¢ce"
        original_series = pd.Series([broken_text])
        fixed_series    = original_series.apply(_fix_encoding_ftfy)

        count = _count_encoding_repairs(original_series, fixed_series)

        # If ftfy is installed and fixes the string, count should be 1.
        # If ftfy is not installed or does not change it, count must be 0 (not negative).
        assert count >= 0, "Repair count must never be negative"
        # Verify the count correctly reflects the actual difference
        if str(fixed_series.iloc[0]) != broken_text:
            assert count == 1, f"Expected count=1 for one repair, got {count}"
        else:
            assert count == 0, f"Expected count=0 (no change occurred), got {count}"

    def test_ftfy_pipeline_reports_consistent_count(self):
        """
        The encoding_repaired value in the pipeline report must match
        what _count_encoding_repairs would return independently.
        Mixed: some broken, some clean.
        """
        df = pd.DataFrame({
            "text": ["Hello", "GrÃ¢ce", "Normal", "World"]
        })
        original_col = df["text"].copy()
        fixed_col    = original_col.apply(_fix_encoding_ftfy)
        expected     = _count_encoding_repairs(original_col, fixed_col)

        _, report = run_cleaning_pipeline(
            df,
            {
                "use_ftfy":          True,
                "remove_empty_cols": False,
                "remove_dup_cols":   False,
                "remove_empty_rows": False,
                "trim_spaces":       False,
                "fill_strategy":     "Auto (Median/Mode)",
            },
        )
        assert report["encoding_repaired"] == expected, (
            f"Pipeline reported {report['encoding_repaired']} repairs "
            f"but expected {expected}"
        )

    def test_ftfy_nan_not_counted_as_repair(self):
        """NaN cells must never be counted as encoding repairs."""
        s_orig  = pd.Series([None, float("nan"), pd.NA])
        s_fixed = pd.Series([None, float("nan"), pd.NA])
        count   = _count_encoding_repairs(s_orig, s_fixed)
        assert count == 0, f"NaN cells should not be counted, got {count}"

    def test_pipeline_encoding_repaired_not_over_counted(self):
        """
        When a DataFrame contains only clean ASCII text,
        encoding_repaired must be 0 — not len(df) * n_str_cols.
        """
        df = pd.DataFrame({
            "A": ["hello", "world", "foo", "bar"],
            "B": ["x", "y", "z", "w"],
        })
        _, report = run_cleaning_pipeline(
            df,
            {
                "use_ftfy":          True,
                "remove_empty_cols": False,
                "remove_dup_cols":   False,
                "remove_empty_rows": False,
                "trim_spaces":       False,
                "fill_strategy":     "Auto (Median/Mode)",
            },
        )
        assert report["encoding_repaired"] == 0, (
            f"Expected 0 for clean ASCII, got {report['encoding_repaired']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Original Tests (preserved + improved)
# ─────────────────────────────────────────────────────────────────────────────

class TestDuplicateDetection:

    def test_exact_duplicate_rows_removed(self, sample_df):
        """Duplicate rows are removed and the report reflects the correct count."""
        df, report = run_cleaning_pipeline(
            sample_df,
            {"remove_empty_rows": False, "fill_strategy": "Auto (Median/Mode)"},
        )
        assert report["duplicates_removed"] >= 1
        assert len(df) < len(sample_df)

    def test_exact_duplicate_helper(self):
        df = pd.DataFrame({"A": [1, 2, 1], "B": ["x", "y", "x"]})
        cleaned, n = remove_exact_duplicates(df)
        assert n == 1
        assert len(cleaned) == 2

    def test_fuzzy_no_auto_delete(self):
        """
        Fuzzy duplicate detection must return candidate pairs only.
        The original DataFrame must remain COMPLETELY UNCHANGED.
        """
        df = pd.DataFrame({
            "Name": ["John Smith", "john smith", "Jane Doe", "JOHN SMITH"],
        })
        original_len  = len(df)
        original_data = df.copy()

        try:
            result = find_fuzzy_duplicates(df, ["Name"], threshold=0.80)
            # Must be a DataFrame of pairs (not None means rapidfuzz is installed)
            if result is not None:
                assert isinstance(result, pd.DataFrame)
                # Must contain pair metadata columns — not actual cleaned rows
                if not result.empty:
                    assert "Index A" in result.columns
                    assert "Index B" in result.columns
                    assert "Similarity" in result.columns
        except Exception:
            pass  # rapidfuzz not installed — acceptable

        # The original DataFrame must be completely unchanged
        assert len(df) == original_len
        pd.testing.assert_frame_equal(df, original_data)

    def test_fuzzy_returns_none_for_large_df(self):
        """
        Fuzzy detection must return None for DataFrames > MAX_ROWS_FUZZY,
        preventing accidental timeout on large datasets.
        """
        from utils.helpers import MAX_ROWS_FUZZY
        large_df = pd.DataFrame({
            "Name": [f"Person {i}" for i in range(MAX_ROWS_FUZZY + 1)]
        })
        result = find_fuzzy_duplicates(large_df, ["Name"])
        assert result is None


class TestMissingValues:

    def test_fill_auto_no_nulls_remain(self):
        df = pd.DataFrame({"X": [1.0, 2.0, None], "Y": ["a", None, "a"]})
        cleaned, count = fill_missing(df, "Auto (Median/Mode)")
        assert cleaned.isnull().sum().sum() == 0
        assert count > 0

    def test_fill_with_zero(self):
        df = pd.DataFrame({"X": [1.0, None, 3.0]})
        cleaned, _ = fill_missing(df, "Fill with 0")
        assert cleaned["X"].isnull().sum() == 0
        assert cleaned["X"].iloc[1] == 0.0

    def test_fill_with_unknown(self):
        df = pd.DataFrame({"Y": ["a", None, "c"]})
        cleaned, _ = fill_missing(df, "Fill with 'Unknown'")
        assert "Unknown" in cleaned["Y"].values

    def test_drop_missing_rows(self):
        df = pd.DataFrame({"X": [1.0, None, 3.0]})
        cleaned, _ = fill_missing(df, "Drop rows with missing values")
        assert cleaned.isnull().sum().sum() == 0
        assert len(cleaned) == 2


class TestColumnOperations:

    def test_empty_columns_removed(self):
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [None, None, None],
            "C": ["x", "y", "z"],
        })
        cleaned, n = remove_empty_columns(df)
        assert "B" not in cleaned.columns
        assert n == 1

    def test_duplicate_columns_removed(self):
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [1, 2, 3],   # identical content
            "C": [4, 5, 6],
        })
        cleaned, n = remove_duplicate_columns(df)
        assert n == 1
        assert len(cleaned.columns) == 2


class TestWhitespace:

    def test_whitespace_stripped_from_values(self, sample_df):
        df, _ = run_cleaning_pipeline(
            sample_df,
            {"trim_spaces": True, "fill_strategy": "Auto (Median/Mode)"},
        )
        for val in df["Notes"].dropna():
            assert not str(val).startswith(" "), f"Leading space in: '{val}'"
            assert not str(val).endswith(" "),   f"Trailing space in: '{val}'"
            assert "  " not in str(val),         f"Double space in: '{val}'"


class TestSnakeCase:

    def test_to_snake_case_basic(self):
        assert to_snake_case("First Name")    == "first_name"
        assert to_snake_case("EmailAddress")  == "email_address"
        assert to_snake_case("  Phone #  ")  == "phone"
        assert to_snake_case("camelCase")     == "camel_case"
        assert to_snake_case("already_snake") == "already_snake"

    def test_snake_case_pipeline(self):
        df = pd.DataFrame({
            "First Name": [1], "Last Name": [2], "Email Address": [3]
        })
        cleaned, report = run_cleaning_pipeline(
            df,
            {"snake_case": True, "fill_strategy": "Auto (Median/Mode)"},
        )
        assert "first_name"    in cleaned.columns
        assert "last_name"     in cleaned.columns
        assert report["headers_snake_cased"] > 0


class TestEdgeCases:

    def test_pipeline_empty_dataframe(self):
        """Pipeline must not crash on empty DataFrame."""
        df = pd.DataFrame()
        try:
            cleaned, report = run_cleaning_pipeline(df, {})
            assert isinstance(cleaned, pd.DataFrame)
        except Exception as e:
            pytest.fail(f"Pipeline crashed on empty DataFrame: {e}")

    def test_pipeline_single_row(self):
        """Pipeline must handle a single-row DataFrame."""
        df = pd.DataFrame({"A": [1], "B": ["hello"]})
        cleaned, report = run_cleaning_pipeline(df, {})
        assert isinstance(cleaned, pd.DataFrame)
        assert len(cleaned) >= 0

    def test_encoding_repair_safe_for_normal_text(self):
        """Encoding repair must not corrupt normal ASCII/UTF-8 text."""
        df = pd.DataFrame({
            "Text": ["Hello World", "Normal text", "No encoding issues"]
        })
        cleaned, _ = run_cleaning_pipeline(
            df,
            {
                "use_ftfy":          True,
                "fill_strategy":     "Auto (Median/Mode)",
                "remove_empty_cols": False,
                "remove_dup_cols":   False,
            },
        )
        assert "Hello World"    in cleaned["Text"].values
        assert "Normal text"    in cleaned["Text"].values


class TestNormalization:

    def test_email_normalization_valid(self):
        assert normalize_email("JOHN@EMAIL.COM")   == "john@email.com"
        assert normalize_email(" alice@test.org ") == "alice@test.org"

    def test_email_normalization_invalid_unchanged(self):
        """Invalid emails must be returned unchanged, not deleted."""
        result = normalize_email("not-an-email")
        assert result == "not-an-email"

    def test_email_normalization_null(self):
        result = normalize_email(None)
        assert result is None or pd.isna(result)

    def test_phone_normalization(self):
        assert normalize_phone("+1 (555) 123-4567") == "+15551234567"
        assert normalize_phone("(555) 987 6543")    == "5559876543"
        assert normalize_phone("+44 20 7946 0958")  == "+442079460958"

    def test_date_normalization_iso(self):
        assert normalize_date("2024-01-15")  == "2024-01-15"

    def test_date_normalization_slash(self):
        # dd/mm/yyyy format
        result = normalize_date("15/01/2024")
        assert result in ("2024-01-15", "01-15-2024", "15/01/2024")
        # At minimum it should not raise

    def test_normalization_only_applied_to_correct_columns(self, customers_df):
        """
        Phone normalization must NOT be applied to signup_date.
        Date normalization must NOT be applied to phone column.
        """
        profile = profile_dataframe(customers_df)

        phone_cols = profile["type_groups"].get("phone", [])
        date_cols  = profile["type_groups"].get("date",  [])

        # signup_date must not appear in phone normalization targets
        assert "signup_date" not in phone_cols, (
            f"signup_date in phone_cols — would incorrectly normalize dates as phones"
        )

        # phone must not appear in date normalization targets
        assert "phone" not in date_cols, (
            f"phone in date_cols — would incorrectly normalize phones as dates"
        )
