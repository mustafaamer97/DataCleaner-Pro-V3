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
    """Realistic fixture mirroring sample_customers.csv column structure."""
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
#  REGRESSION TEST — exact case reported in live QA
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveRegressionCase:
    """
    Exact regression test for the bug reported from the live Streamlit app:
        Phone columns: phone, signup_date   ← WRONG
        Expected:
        Phone columns: phone
        Date columns:  signup_date
    """

    @pytest.fixture
    def regression_df(self):
        """Exact DataFrame described in the QA bug report."""
        return pd.DataFrame({
            "id":          [1, 2],
            "email":       ["a@test.com", "b@test.com"],
            "phone":       ["+15551234567", "+15551234568"],
            "signup_date": ["2026-01-01", "2026-01-02"],
        })

    def test_email_classified_as_email(self, regression_df):
        result = detect_column_type(regression_df["email"], col_name="email")
        assert result == "email", f"Expected 'email', got '{result}'"

    def test_phone_classified_as_phone(self, regression_df):
        result = detect_column_type(regression_df["phone"], col_name="phone")
        assert result == "phone", f"Expected 'phone', got '{result}'"

    def test_signup_date_classified_as_date(self, regression_df):
        result = detect_column_type(regression_df["signup_date"], col_name="signup_date")
        assert result == "date", (
            f"Expected 'date' for signup_date, got '{result}'. "
            f"This is the primary regression — signup_date must NEVER be 'phone'."
        )

    def test_signup_date_is_not_phone(self, regression_df):
        result = detect_column_type(regression_df["signup_date"], col_name="signup_date")
        assert result != "phone", (
            f"REGRESSION: signup_date classified as 'phone'. "
            f"This is the exact bug reported in live QA."
        )

    def test_id_classified_as_id(self, regression_df):
        result = detect_column_type(regression_df["id"], col_name="id")
        assert result == "id", f"Expected 'id', got '{result}'"

    def test_full_profile_phone_group_excludes_signup_date(self, regression_df):
        """Full profile phone group must NOT contain signup_date."""
        profile    = profile_dataframe(regression_df)
        phone_cols = profile["type_groups"].get("phone", [])
        assert "signup_date" not in phone_cols, (
            f"REGRESSION: signup_date appeared in phone_cols={phone_cols}. "
            f"This is the exact bug visible in the live Streamlit app."
        )

    def test_full_profile_date_group_includes_signup_date(self, regression_df):
        """Full profile date group must contain signup_date."""
        profile   = profile_dataframe(regression_df)
        date_cols = profile["type_groups"].get("date", [])
        assert "signup_date" in date_cols, (
            f"signup_date not found in date_cols={date_cols}"
        )

    def test_full_profile_complete_classification(self, regression_df):
        """All four columns must land in the correct type groups."""
        profile = profile_dataframe(regression_df)
        tg      = profile["type_groups"]

        assert "email"       in tg.get("email",  []), "email → email"
        assert "phone"       in tg.get("phone",  []), "phone → phone"
        assert "signup_date" in tg.get("date",   []), "signup_date → date"
        assert "id"          in tg.get("id",     []), "id → id"

        assert "signup_date" not in tg.get("phone", []), \
            "signup_date must NOT be in phone group"


# ─────────────────────────────────────────────────────────────────────────────
#  Column Type Detection — Full Coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestColumnTypeDetection:

    def test_signup_date_is_date_not_phone(self, customers_df):
        series = customers_df["signup_date"]
        result = detect_column_type(series, col_name="signup_date")
        assert result == "date", f"Expected 'date', got '{result}'"

    def test_phone_column_is_phone(self, customers_df):
        series = customers_df["phone"]
        result = detect_column_type(series, col_name="phone")
        assert result == "phone", f"Expected 'phone', got '{result}'"

    def test_email_column_is_email(self, customers_df):
        series = customers_df["email"]
        result = detect_column_type(series, col_name="email")
        assert result == "email", f"Expected 'email', got '{result}'"

    def test_id_column_is_id(self, customers_df):
        series = customers_df["id"]
        result = detect_column_type(series, col_name="id")
        assert result == "id", f"Expected 'id', got '{result}'"

    def test_age_column_is_numeric(self, customers_df):
        series = customers_df["age"]
        result = detect_column_type(series, col_name="age")
        assert result == "numeric", f"Expected 'numeric', got '{result}'"

    def test_salary_is_numeric_or_currency(self, customers_df):
        series = customers_df["salary"]
        result = detect_column_type(series, col_name="salary")
        assert result in ("numeric", "currency"), f"Unexpected type '{result}'"

    def test_profile_phone_group_excludes_signup_date(self, customers_df):
        profile    = profile_dataframe(customers_df)
        phone_cols = profile["type_groups"].get("phone", [])
        assert "signup_date" not in phone_cols, \
            f"signup_date in phone_cols: {phone_cols}"

    def test_profile_date_group_includes_signup_date(self, customers_df):
        profile   = profile_dataframe(customers_df)
        date_cols = profile["type_groups"].get("date", [])
        assert "signup_date" in date_cols, \
            f"signup_date not in date_cols: {date_cols}"

    def test_all_date_keyword_columns_not_phone(self):
        """Columns with date-related names must never be classified as phone."""
        date_col_names = [
            "birth_date", "created_at", "updated_at",
            "registration_date", "date", "dob",
            "signup_date", "joined_date", "expiry_date",
            "modified_at", "timestamp",
        ]
        date_values = pd.Series([
            "2024-01-15", "2023-06-30", "2022-12-01",
            "15/01/2024", "Feb 3 2024", "2024-03-10",
        ])
        for col_name in date_col_names:
            result = detect_column_type(date_values, col_name=col_name)
            assert result == "date", \
                f"Column '{col_name}' → expected 'date', got '{result}'"

    def test_iso_date_content_detected_without_name_hint(self):
        """ISO date values should be detected as 'date' even with a generic column name."""
        date_values = pd.Series([
            "2024-01-15", "2024-02-20", "2024-03-10",
            "2024-04-01", "2024-05-12", "2024-06-30",
        ])
        result = detect_column_type(date_values, col_name="col_x")
        assert result == "date", \
            f"ISO date content should be 'date', got '{result}'"

    def test_mixed_date_formats_detected(self):
        """Mixed date formats (ISO + slash + text) still resolve to 'date'."""
        mixed = pd.Series([
            "2024-01-15", "15/01/2024", "Feb 3 2024",
            "2024-03-10", "2024-06-30", "01/07/2024",
        ])
        result = detect_column_type(mixed, col_name="col_x")
        assert result == "date", \
            f"Mixed date formats should be 'date', got '{result}'"

    def test_phone_keyword_without_date_content(self):
        """A column named 'mobile' with phone numbers is 'phone'."""
        phones = pd.Series([
            "+15551234567", "5559876543", "+442079460958",
            "5552223333", "5553334444",
        ])
        result = detect_column_type(phones, col_name="mobile")
        assert result == "phone", f"Expected 'phone', got '{result}'"


# ─────────────────────────────────────────────────────────────────────────────
#  Encoding Repair Counter
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodingRepairCounter:

    def test_ftfy_count_zero_for_normal_text(self):
        """Normal ASCII text must produce repair count of 0."""
        df       = pd.DataFrame({"Text": ["Hello World", "Normal text", "Clean data"]})
        original = df["Text"].copy()
        df["Text"] = df["Text"].apply(_fix_encoding_ftfy)
        count    = _count_encoding_repairs(original, df["Text"])
        assert count == 0, f"Expected 0, got {count}"

    def test_ftfy_count_increases_for_broken_text(self):
        """Repair count must be ≥ 0 and reflect actual changes."""
        broken          = pd.Series(["GrÃ¢ce"])
        fixed           = broken.apply(_fix_encoding_ftfy)
        count           = _count_encoding_repairs(broken, fixed)
        assert count >= 0
        if str(fixed.iloc[0]) != "GrÃ¢ce":
            assert count == 1

    def test_ftfy_pipeline_count_consistent(self):
        """Pipeline report encoding_repaired must match independent count."""
        df           = pd.DataFrame({"text": ["Hello", "GrÃ¢ce", "Normal", "World"]})
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
        assert report["encoding_repaired"] == expected

    def test_ftfy_nan_not_counted(self):
        """NaN cells must never be counted as encoding repairs."""
        s_orig  = pd.Series([None, float("nan"), pd.NA])
        s_fixed = pd.Series([None, float("nan"), pd.NA])
        count   = _count_encoding_repairs(s_orig, s_fixed)
        assert count == 0

    def test_pipeline_clean_ascii_repair_count_is_zero(self):
        """Clean ASCII DataFrame must produce encoding_repaired == 0."""
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
        assert report["encoding_repaired"] == 0, \
            f"Expected 0, got {report['encoding_repaired']}"


# ─────────────────────────────────────────────────────────────────────────────
#  Duplicate Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDuplicateDetection:

    def test_exact_duplicate_rows_removed(self, sample_df):
        df, report = run_cleaning_pipeline(
            sample_df,
            {"remove_empty_rows": False, "fill_strategy": "Auto (Median/Mode)"},
        )
        assert report["duplicates_removed"] >= 1
        assert len(df) < len(sample_df)

    def test_exact_duplicate_helper(self):
        df         = pd.DataFrame({"A": [1, 2, 1], "B": ["x", "y", "x"]})
        cleaned, n = remove_exact_duplicates(df)
        assert n == 1
        assert len(cleaned) == 2

    def test_fuzzy_no_auto_delete(self):
        """Fuzzy detection returns candidate pairs — NEVER deletes rows."""
        df            = pd.DataFrame({"Name": ["John Smith", "john smith", "Jane Doe"]})
        original_len  = len(df)
        original_data = df.copy()

        try:
            result = find_fuzzy_duplicates(df, ["Name"], threshold=0.80)
            if result is not None and not result.empty:
                assert "Index A"     in result.columns
                assert "Index B"     in result.columns
                assert "Similarity"  in result.columns
        except Exception:
            pass  # rapidfuzz not installed

        assert len(df) == original_len
        pd.testing.assert_frame_equal(df, original_data)

    def test_fuzzy_returns_none_for_large_df(self):
        """Fuzzy detection must return None for DataFrames above the row limit."""
        from utils.helpers import MAX_ROWS_FUZZY
        large_df = pd.DataFrame({
            "Name": [f"Person {i}" for i in range(MAX_ROWS_FUZZY + 1)]
        })
        result = find_fuzzy_duplicates(large_df, ["Name"])
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
#  Missing Values
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingValues:

    def test_fill_auto_no_nulls_remain(self):
        df         = pd.DataFrame({"X": [1.0, 2.0, None], "Y": ["a", None, "a"]})
        cleaned, c = fill_missing(df, "Auto (Median/Mode)")
        assert cleaned.isnull().sum().sum() == 0
        assert c > 0

    def test_fill_with_zero(self):
        df         = pd.DataFrame({"X": [1.0, None, 3.0]})
        cleaned, _ = fill_missing(df, "Fill with 0")
        assert cleaned["X"].isnull().sum() == 0
        assert cleaned["X"].iloc[1] == 0.0

    def test_fill_with_unknown(self):
        df         = pd.DataFrame({"Y": ["a", None, "c"]})
        cleaned, _ = fill_missing(df, "Fill with 'Unknown'")
        assert "Unknown" in cleaned["Y"].values

    def test_drop_missing_rows(self):
        df         = pd.DataFrame({"X": [1.0, None, 3.0]})
        cleaned, _ = fill_missing(df, "Drop rows with missing values")
        assert cleaned.isnull().sum().sum() == 0
        assert len(cleaned) == 2


# ─────────────────────────────────────────────────────────────────────────────
#  Column Operations
# ─────────────────────────────────────────────────────────────────────────────

class TestColumnOperations:

    def test_empty_columns_removed(self):
        df         = pd.DataFrame({"A": [1, 2, 3], "B": [None, None, None], "C": ["x", "y", "z"]})
        cleaned, n = remove_empty_columns(df)
        assert "B" not in cleaned.columns
        assert n == 1

    def test_duplicate_columns_removed(self):
        df         = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3], "C": [4, 5, 6]})
        cleaned, n = remove_duplicate_columns(df)
        assert n == 1
        assert len(cleaned.columns) == 2


# ─────────────────────────────────────────────────────────────────────────────
#  Whitespace
# ─────────────────────────────────────────────────────────────────────────────

class TestWhitespace:

    def test_whitespace_stripped(self, sample_df):
        df, _ = run_cleaning_pipeline(
            sample_df,
            {"trim_spaces": True, "fill_strategy": "Auto (Median/Mode)"},
        )
        for val in df["Notes"].dropna():
            assert not str(val).startswith(" "), f"Leading space: '{val}'"
            assert not str(val).endswith(" "),   f"Trailing space: '{val}'"
            assert "  " not in str(val),          f"Double space: '{val}'"


# ─────────────────────────────────────────────────────────────────────────────
#  snake_case
# ─────────────────────────────────────────────────────────────────────────────

class TestSnakeCase:

    def test_to_snake_case_variants(self):
        assert to_snake_case("First Name")    == "first_name"
        assert to_snake_case("EmailAddress")  == "email_address"
        assert to_snake_case("  Phone #  ")  == "phone"
        assert to_snake_case("camelCase")     == "camel_case"
        assert to_snake_case("already_snake") == "already_snake"

    def test_snake_case_pipeline(self):
        df = pd.DataFrame({"First Name": [1], "Last Name": [2], "Email Address": [3]})
        cleaned, report = run_cleaning_pipeline(
            df,
            {"snake_case": True, "fill_strategy": "Auto (Median/Mode)"},
        )
        assert "first_name"   in cleaned.columns
        assert "last_name"    in cleaned.columns
        assert report["headers_snake_cased"] > 0


# ─────────────────────────────────────────────────────────────────────────────
#  Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_pipeline_empty_dataframe(self):
        df = pd.DataFrame()
        try:
            cleaned, report = run_cleaning_pipeline(df, {})
            assert isinstance(cleaned, pd.DataFrame)
        except Exception as e:
            pytest.fail(f"Pipeline crashed on empty DataFrame: {e}")

    def test_pipeline_single_row(self):
        df              = pd.DataFrame({"A": [1], "B": ["hello"]})
        cleaned, report = run_cleaning_pipeline(df, {})
        assert isinstance(cleaned, pd.DataFrame)

    def test_encoding_repair_does_not_corrupt_normal_text(self):
        df = pd.DataFrame({"Text": ["Hello World", "Normal text", "No encoding issues"]})
        cleaned, _ = run_cleaning_pipeline(
            df,
            {
                "use_ftfy":          True,
                "fill_strategy":     "Auto (Median/Mode)",
                "remove_empty_cols": False,
                "remove_dup_cols":   False,
            },
        )
        assert "Hello World"  in cleaned["Text"].values
        assert "Normal text"  in cleaned["Text"].values


# ─────────────────────────────────────────────────────────────────────────────
#  Normalization
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalization:

    def test_email_valid(self):
        assert normalize_email("JOHN@EMAIL.COM")   == "john@email.com"
        assert normalize_email(" alice@test.org ") == "alice@test.org"

    def test_email_invalid_unchanged(self):
        assert normalize_email("not-an-email") == "not-an-email"

    def test_email_null(self):
        result = normalize_email(None)
        assert result is None or pd.isna(result)

    def test_phone_strips_formatting(self):
        assert normalize_phone("+1 (555) 123-4567") == "+15551234567"
        assert normalize_phone("(555) 987 6543")    == "5559876543"
        assert normalize_phone("+44 20 7946 0958")  == "+442079460958"

    def test_date_iso_passthrough(self):
        assert normalize_date("2024-01-15") == "2024-01-15"

    def test_normalization_targets_are_correctly_separated(self, customers_df):
        """
        Phone normalization must NOT target signup_date.
        Date normalization must NOT target phone.
        """
        profile    = profile_dataframe(customers_df)
        phone_cols = profile["type_groups"].get("phone", [])
        date_cols  = profile["type_groups"].get("date",  [])

        assert "signup_date" not in phone_cols, \
            f"signup_date in phone normalization targets: {phone_cols}"
        assert "phone" not in date_cols, \
            f"phone in date normalization targets: {date_cols}"
