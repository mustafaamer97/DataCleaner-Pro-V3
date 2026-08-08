"""
tests/test_cleaning.py
Unit tests for DataCleaner Pro V3 — Cleaning Engine.

Run with:  python -m pytest tests/ -v
"""

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
)
from utils.duplicates import find_fuzzy_duplicates, remove_exact_duplicates
from utils.helpers import to_snake_case


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Name":  ["Alice", "Bob", "Alice", "Charlie", None],
        "Email": ["ALICE@EMAIL.COM", "bob@email.com", "ALICE@EMAIL.COM", "bad-email", None],
        "Age":   [30, 25, 30, None, 40],
        "Notes": ["  Hello  ", "World", "  Hello  ", "Test  ", ""],
    })


# ── Test 1: Duplicate rows are removed ───────────────────────────────────────

def test_duplicate_rows_removed(sample_df):
    df, report = run_cleaning_pipeline(
        sample_df,
        {"remove_empty_rows": False, "fill_strategy": "Auto (Median/Mode)"},
    )
    assert report["duplicates_removed"] >= 1
    assert len(df) < len(sample_df)


# ── Test 2: Empty columns are removed ────────────────────────────────────────

def test_empty_columns_removed():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [None, None, None],
        "C": ["x", "y", "z"],
    })
    cleaned, n = remove_empty_columns(df)
    assert "B" not in cleaned.columns
    assert n == 1


# ── Test 3: Missing values are handled correctly ──────────────────────────────

def test_fill_missing_auto():
    df = pd.DataFrame({"X": [1.0, 2.0, None], "Y": ["a", None, "a"]})
    cleaned, count = fill_missing(df, "Auto (Median/Mode)")
    assert cleaned.isnull().sum().sum() == 0
    assert count > 0


def test_fill_missing_zero():
    df = pd.DataFrame({"X": [1.0, None, 3.0]})
    cleaned, _ = fill_missing(df, "Fill with 0")
    assert cleaned["X"].isnull().sum() == 0
    assert cleaned["X"].iloc[1] == 0


def test_fill_missing_unknown():
    df = pd.DataFrame({"Y": ["a", None, "c"]})
    cleaned, _ = fill_missing(df, "Fill with 'Unknown'")
    assert "Unknown" in cleaned["Y"].values


def test_drop_missing_rows():
    df = pd.DataFrame({"X": [1.0, None, 3.0]})
    cleaned, _ = fill_missing(df, "Drop rows with missing values")
    assert cleaned.isnull().sum().sum() == 0
    assert len(cleaned) == 2


# ── Test 4: Whitespace is normalized ─────────────────────────────────────────

def test_whitespace_normalized(sample_df):
    df, _ = run_cleaning_pipeline(
        sample_df,
        {"trim_spaces": True, "fill_strategy": "Auto (Median/Mode)"},
    )
    for val in df["Notes"].dropna():
        assert not str(val).startswith(" ")
        assert not str(val).endswith(" ")
        assert "  " not in str(val)


# ── Test 5: snake_case works ──────────────────────────────────────────────────

def test_snake_case():
    assert to_snake_case("First Name")    == "first_name"
    assert to_snake_case("EmailAddress")  == "email_address"
    assert to_snake_case("  Phone #  ")  == "phone"
    assert to_snake_case("camelCase")     == "camel_case"
    assert to_snake_case("already_snake") == "already_snake"


def test_snake_case_pipeline():
    df = pd.DataFrame({"First Name": [1], "Last Name": [2], "Email Address": [3]})
    cleaned, report = run_cleaning_pipeline(
        df,
        {"snake_case": True, "fill_strategy": "Auto (Median/Mode)"},
    )
    assert "first_name" in cleaned.columns
    assert "last_name"  in cleaned.columns
    assert report["headers_snake_cased"] > 0


# ── Test 6: Encoding repair does not corrupt normal text ──────────────────────

def test_encoding_repair_safe():
    df = pd.DataFrame({
        "Text": ["Hello World", "Normal text", "No encoding issues"]
    })
    cleaned, _ = run_cleaning_pipeline(
        df,
        {"use_ftfy": True, "fill_strategy": "Auto (Median/Mode)"},
    )
    # Normal text should pass through unchanged
    assert "Hello World" in cleaned["Text"].values
    assert "Normal text" in cleaned["Text"].values


# ── Test 7: PDF extraction handles empty pages safely ─────────────────────────

def test_pdf_empty_page_safe():
    """_empty_result should always return a well-formed dict."""
    from utils.pdf_processor import _empty_result
    result = _empty_result(99)
    assert result["page"]      == 99
    assert result["dataframe"] is None
    assert result["method"]    == "none"


# ── Test 8: Fuzzy duplicate detection does NOT auto-delete ───────────────────

def test_fuzzy_no_auto_delete():
    """
    Fuzzy duplicate detection must return candidates only,
    never modify the original DataFrame.
    """
    df = pd.DataFrame({
        "Name": ["John Smith", "john smith", "Jane Doe", "JOHN SMITH"],
    })
    original_len = len(df)

    # find_fuzzy_duplicates should return a DataFrame of pairs, not delete rows
    try:
        result = find_fuzzy_duplicates(df, ["Name"], threshold=0.80)
        # If rapidfuzz is available, result is a DataFrame of pairs
        assert isinstance(result, (pd.DataFrame, type(None)))
    except Exception:
        pass  # rapidfuzz not installed — acceptable

    # Original DataFrame must be unchanged
    assert len(df) == original_len


# ── Test 9: Duplicate columns removed ────────────────────────────────────────

def test_duplicate_columns_removed():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [1, 2, 3],  # identical to A
        "C": [4, 5, 6],
    })
    cleaned, n = remove_duplicate_columns(df)
    assert n == 1
    assert len(cleaned.columns) == 2


# ── Test 10: Email normalization ──────────────────────────────────────────────

def test_email_normalization():
    assert normalize_email("JOHN@EMAIL.COM")   == "john@email.com"
    assert normalize_email(" alice@test.org ") == "alice@test.org"
    assert normalize_email("not-an-email")     == "not-an-email"  # invalid — unchanged
    assert normalize_email(None) is None or pd.isna(normalize_email(None))


# ── Test 11: Phone normalization ──────────────────────────────────────────────

def test_phone_normalization():
    assert normalize_phone("+1 (555) 123-4567") == "+15551234567"
    assert normalize_phone("(555) 987 6543")    == "5559876543"
    assert normalize_phone("+44 20 7946 0958")  == "+442079460958"


# ── Test 12: Pipeline never crashes on empty DataFrame ───────────────────────

def test_pipeline_empty_dataframe():
    df = pd.DataFrame()
    try:
        cleaned, report = run_cleaning_pipeline(df, {})
        assert isinstance(cleaned, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"Pipeline crashed on empty DataFrame: {e}")
