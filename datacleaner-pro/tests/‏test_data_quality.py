"""
tests/test_data_quality.py
==========================
Tests for the Data Quality Intelligence Engine.

Run with:  pytest tests/ -v --tb=short
"""

from __future__ import annotations

import sys
import os
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import pytest

from utils.data_quality import (
    analyze_data_quality,
    QualityReport,
    QualityIssue,
    SEV_HIGH,
    SEV_MEDIUM,
    SEV_LOW,
    SCORE_EXCELLENT,
    SCORE_GOOD,
    SCORE_ATTENTION,
    score_label,
    score_emoji,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_df(nrows: int = 100) -> pd.DataFrame:
    """Return a perfectly clean, varied DataFrame."""
    return pd.DataFrame({
        "id":      range(1, nrows + 1),
        "name":    [f"Person {i}" for i in range(nrows)],
        "email":   [f"user{i}@example.com" for i in range(nrows)],
        "country": ["USA", "Canada", "UK", "Australia", "Germany"] * (nrows // 5),
        "salary":  [50_000 + i * 100 for i in range(nrows)],
        "age":     [20 + (i % 60) for i in range(nrows)],
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PERFECT CLEAN DATASET → HIGH SCORE
# ═══════════════════════════════════════════════════════════════════════════════

class TestCleanDataset:

    def test_clean_dataset_returns_report(self):
        report = analyze_data_quality(_clean_df())
        assert isinstance(report, QualityReport)

    def test_clean_dataset_score_is_good_or_excellent(self):
        report = analyze_data_quality(_clean_df(200))
        assert report.score >= SCORE_GOOD, (
            f"Expected score >= {SCORE_GOOD}, got {report.score}"
        )

    def test_clean_dataset_no_high_priority_issues(self):
        report = analyze_data_quality(_clean_df(200))
        high   = report.high_priority_issues
        assert len(high) == 0, (
            f"Unexpected HIGH issues on clean data: {[i.title for i in high]}"
        )

    def test_clean_dataset_categories_all_above_70(self):
        report = analyze_data_quality(_clean_df(200))
        for cat, val in report.categories.items():
            assert val >= 70, f"Category '{cat}' = {val}, expected >= 70"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATASET WITH MISSING VALUES
# ═══════════════════════════════════════════════════════════════════════════════

class TestMissingValues:

    def test_high_missing_column_detected(self):
        df = pd.DataFrame({
            "id":    range(100),
            "name":  ["Alice"] * 100,
            "email": [None] * 80 + ["a@b.com"] * 20,   # 80% missing
        })
        report = analyze_data_quality(df)
        cols_in_issues = [i.column for i in report.issues]
        assert "email" in cols_in_issues

    def test_nearly_empty_column_is_high_severity(self):
        df = pd.DataFrame({
            "id":    range(100),
            "notes": [None] * 95 + ["x"] * 5,   # 95 % missing
        })
        report = analyze_data_quality(df)
        high_cols = [i.column for i in report.issues if i.severity == SEV_HIGH]
        assert "notes" in high_cols

    def test_missing_values_lower_completeness_score(self):
        clean   = analyze_data_quality(_clean_df(100))
        dirty   = pd.DataFrame({
            "a": [None] * 60 + list(range(40)),
            "b": list(range(100)),
        })
        missing = analyze_data_quality(dirty)
        assert missing.categories["completeness"] < clean.categories["completeness"]

    def test_missing_pct_in_stats(self):
        df = pd.DataFrame({"a": [None, 1, 2, None, 4]})
        report = analyze_data_quality(df)
        assert report.stats["missing_pct"] > 0

    def test_moderate_missing_is_medium_not_high(self):
        """15–70% missing → MEDIUM severity, not HIGH."""
        df = pd.DataFrame({
            "id":    range(100),
            "notes": [None] * 30 + ["x"] * 70,   # 30% missing
        })
        report = analyze_data_quality(df)
        notes_issues = [i for i in report.issues if i.column == "notes"]
        severities   = {i.severity for i in notes_issues}
        assert SEV_HIGH not in severities
        assert SEV_MEDIUM in severities


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATASET WITH DUPLICATE ROWS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDuplicateRows:

    def test_duplicate_rows_detected(self):
        base = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df   = pd.concat([base, base], ignore_index=True)
        report = analyze_data_quality(df)
        assert report.stats["duplicate_rows"] == 3

    def test_duplicate_rows_lower_uniqueness(self):
        clean = analyze_data_quality(_clean_df(100))
        base  = pd.DataFrame({"a": range(10), "b": range(10)})
        duped = pd.concat([base] * 5, ignore_index=True)
        dirty = analyze_data_quality(duped)
        assert dirty.categories["uniqueness"] < clean.categories["uniqueness"]

    def test_duplicate_issue_in_report(self):
        base   = pd.DataFrame({"a": [1, 2, 3]})
        df     = pd.concat([base, base], ignore_index=True)
        report = analyze_data_quality(df)
        dup_issues = [i for i in report.issues if i.category == "uniqueness"]
        assert len(dup_issues) >= 1

    def test_no_duplicates_no_uniqueness_issue(self):
        df     = pd.DataFrame({"a": range(50), "b": range(50)})
        report = analyze_data_quality(df)
        dup_issues = [i for i in report.issues if i.category == "uniqueness"]
        assert len(dup_issues) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NEARLY EMPTY COLUMN
# ═══════════════════════════════════════════════════════════════════════════════

class TestNearlyEmptyColumn:

    def test_nearly_empty_detected(self):
        df = pd.DataFrame({
            "id":    range(100),
            "ghost": [None] * 85 + ["x"] * 15,
        })
        report  = analyze_data_quality(df)
        titles  = [i.title for i in report.issues]
        assert any("nearly empty" in t.lower() for t in titles)

    def test_nearly_empty_severity_is_high(self):
        df = pd.DataFrame({
            "id":    range(100),
            "ghost": [None] * 85 + ["x"] * 15,
        })
        report = analyze_data_quality(df)
        assert any(
            i.severity == SEV_HIGH and i.column == "ghost"
            for i in report.issues
        )

    def test_nearly_empty_has_recommendation(self):
        df = pd.DataFrame({
            "id":    range(100),
            "ghost": [None] * 90 + ["x"] * 10,
        })
        report = analyze_data_quality(df)
        ghost_issues = [i for i in report.issues if i.column == "ghost"]
        assert any(i.recommendation for i in ghost_issues)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. POTENTIAL KEY CANDIDATE
# ═══════════════════════════════════════════════════════════════════════════════

class TestKeyCandidates:

    def test_id_column_detected_as_key_candidate(self):
        df = pd.DataFrame({
            "id":   range(1, 101),
            "name": [f"Person {i}" for i in range(100)],
        })
        report = analyze_data_quality(df)
        assert "id" in report.key_candidates

    def test_email_column_with_unique_values_is_key_candidate(self):
        df = pd.DataFrame({
            "email":   [f"user{i}@example.com" for i in range(100)],
            "country": ["USA"] * 100,
        })
        report = analyze_data_quality(df)
        assert "email" in report.key_candidates

    def test_non_unique_id_not_a_clean_key_candidate(self):
        """A column named 'id' with duplicates must appear in issues."""
        df = pd.DataFrame({
            "id":   [1, 1, 2, 3, 4] * 20,
            "name": [f"X{i}" for i in range(100)],
        })
        report = analyze_data_quality(df)
        high_issues = [
            i for i in report.issues
            if i.severity == SEV_HIGH and i.column == "id"
        ]
        assert len(high_issues) >= 1

    def test_repeated_value_column_not_key_candidate(self):
        df = pd.DataFrame({
            "country": ["USA", "Canada", "UK"] * 33 + ["USA"],
            "value":   range(100),
        })
        report = analyze_data_quality(df)
        assert "country" not in report.key_candidates


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CATEGORICAL CANDIDATE
# ═══════════════════════════════════════════════════════════════════════════════

class TestCategoricalCandidates:

    def test_low_cardinality_column_detected(self):
        df = pd.DataFrame({
            "country": ["USA", "Canada", "UK", "Australia", "Germany"] * 20,
            "value":   range(100),
        })
        report = analyze_data_quality(df)
        assert "country" in report.categorical_candidates

    def test_high_cardinality_column_not_categorical_candidate(self):
        df = pd.DataFrame({
            "name":  [f"Person {i}" for i in range(100)],
            "value": range(100),
        })
        report = analyze_data_quality(df)
        assert "name" not in report.categorical_candidates

    def test_categorical_candidate_has_low_issue_severity(self):
        df = pd.DataFrame({
            "status": ["active", "inactive", "pending"] * 34,
            "id":     range(102),
        })
        report = analyze_data_quality(df)
        status_issues = [i for i in report.issues if i.column == "status"]
        severities    = {i.severity for i in status_issues}
        if severities:
            assert SEV_HIGH not in severities


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MIXED-TYPE COLUMN
# ═══════════════════════════════════════════════════════════════════════════════

class TestMixedTypeColumn:

    def test_mixed_numeric_text_detected(self):
        df = pd.DataFrame({
            "age": (
                [str(i) for i in range(80)]
                + ["unknown"] * 10
                + ["N/A"] * 10
            ),
        })
        report = analyze_data_quality(df)
        consistency_issues = [
            i for i in report.issues if i.category == "consistency"
        ]
        assert len(consistency_issues) >= 1

    def test_clean_numeric_column_not_flagged_as_mixed(self):
        df = pd.DataFrame({
            "age":   [str(i) for i in range(100)],
            "value": range(100),
        })
        report = analyze_data_quality(df)
        mixed = [
            i for i in report.issues
            if i.category == "consistency" and "mixed" in i.title.lower()
        ]
        # All-numeric string column must not be flagged as mixed
        age_mixed = [m for m in mixed if m.column == "age"]
        assert len(age_mixed) == 0

    def test_mixed_type_recommendation_is_non_empty(self):
        df = pd.DataFrame({
            "age": [str(i) for i in range(80)] + ["unknown"] * 20,
        })
        report = analyze_data_quality(df)
        for issue in report.issues:
            if issue.column == "age":
                assert len(issue.recommendation) > 0
                break


# ═══════════════════════════════════════════════════════════════════════════════
# 8. EMPTY DATAFRAME
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmptyDataFrame:

    def test_empty_df_returns_report(self):
        report = analyze_data_quality(pd.DataFrame())
        assert isinstance(report, QualityReport)

    def test_empty_df_score_is_100(self):
        report = analyze_data_quality(pd.DataFrame())
        assert report.score == 100

    def test_empty_df_no_issues(self):
        report = analyze_data_quality(pd.DataFrame())
        assert report.issues == []

    def test_zero_rows_df_returns_report(self):
        df     = pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=str)})
        report = analyze_data_quality(df)
        assert isinstance(report, QualityReport)
        assert report.score == 100


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SINGLE-ROW DATAFRAME
# ═══════════════════════════════════════════════════════════════════════════════

class TestSingleRow:

    def test_single_row_does_not_crash(self):
        df     = pd.DataFrame({"id": [1], "name": ["Alice"]})
        report = analyze_data_quality(df)
        assert isinstance(report, QualityReport)

    def test_single_row_score_between_0_and_100(self):
        df     = pd.DataFrame({"id": [1], "name": ["Alice"]})
        report = analyze_data_quality(df)
        assert 0 <= report.score <= 100

    def test_single_row_no_duplicate_issues(self):
        df     = pd.DataFrame({"id": [1], "name": ["Alice"]})
        report = analyze_data_quality(df)
        dup    = [i for i in report.issues if i.category == "uniqueness"]
        assert len(dup) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 10. PERFORMANCE SANITY — large-ish dataset
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformance:

    def test_large_dataset_completes_in_reasonable_time(self):
        """50,000 rows must analyze in under 30 seconds."""
        nrows = 50_000
        df = pd.DataFrame({
            "id":      range(nrows),
            "name":    [f"Person {i}" for i in range(nrows)],
            "country": ["USA", "Canada", "UK", "AU", "DE"] * (nrows // 5),
            "salary":  [50_000 + i for i in range(nrows)],
            "notes":   [None] * (nrows // 2) + ["x"] * (nrows // 2),
        })
        start  = time.monotonic()
        report = analyze_data_quality(df)
        elapsed = time.monotonic() - start

        assert isinstance(report, QualityReport)
        assert elapsed < 30.0, (
            f"analyze_data_quality took {elapsed:.1f}s on {nrows} rows "
            "(expected < 30s)"
        )

    def test_large_dataset_score_in_range(self):
        nrows = 10_000
        df    = pd.DataFrame({
            "id":    range(nrows),
            "value": range(nrows),
        })
        report = analyze_data_quality(df)
        assert 0 <= report.score <= 100


# ═══════════════════════════════════════════════════════════════════════════════
# 11. SCORE ALWAYS BETWEEN 0 AND 100
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreBounds:

    @pytest.mark.parametrize("df", [
        pd.DataFrame(),
        pd.DataFrame({"a": [1]}),
        pd.DataFrame({"a": [None] * 100}),
        pd.DataFrame({"a": range(100), "b": range(100)}),
        pd.DataFrame({"a": [1, 1, 1, 1, 1]}),
    ])
    def test_score_in_range(self, df):
        report = analyze_data_quality(df)
        assert 0 <= report.score <= 100, (
            f"Score {report.score} out of range for df shape {df.shape}"
        )

    @pytest.mark.parametrize("df", [
        pd.DataFrame(),
        pd.DataFrame({"a": [1]}),
        pd.DataFrame({"a": [None] * 100}),
    ])
    def test_category_scores_in_range(self, df):
        report = analyze_data_quality(df)
        for cat, val in report.categories.items():
            assert 0 <= val <= 100, (
                f"Category '{cat}' = {val} is out of [0, 100]"
            )

    def test_worst_case_score_not_negative(self):
        """A dataset with every possible problem must still score >= 0."""
        df = pd.DataFrame({
            "id":      [1, 1, 2, 2, 3],
            "notes":   [None, None, None, None, "x"],
            "country": ["USA"] * 5,
            "age":     ["23", "unknown", "31", "N/A", "42"],
        })
        report = analyze_data_quality(df)
        assert report.score >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 12. SCORE LABELS AND EMOJIS
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreLabels:

    def test_score_label_excellent(self):
        assert score_label(95) == "Excellent"

    def test_score_label_good(self):
        assert score_label(75) == "Good"

    def test_score_label_needs_attention(self):
        assert score_label(55) == "Needs Attention"

    def test_score_label_poor(self):
        assert score_label(30) == "Poor"

    def test_score_emoji_excellent(self):
        assert score_emoji(90) == "🟢"

    def test_score_emoji_good(self):
        assert score_emoji(70) == "🟡"

    def test_score_emoji_attention(self):
        assert score_emoji(50) == "🟠"

    def test_score_emoji_poor(self):
        assert score_emoji(49) == "🔴"

    def test_report_score_label_property(self):
        df     = _clean_df(100)
        report = analyze_data_quality(df)
        assert report.score_label in {"Excellent", "Good", "Needs Attention", "Poor"}

    def test_report_score_emoji_property(self):
        df     = _clean_df(100)
        report = analyze_data_quality(df)
        assert report.score_emoji in {"🟢", "🟡", "🟠", "🔴"}


# ═══════════════════════════════════════════════════════════════════════════════
# 13. REPORT STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

class TestReportStructure:

    def test_report_has_all_category_keys(self):
        report = analyze_data_quality(_clean_df())
        expected = {"completeness", "uniqueness", "consistency", "validity"}
        assert set(report.categories.keys()) == expected

    def test_report_has_stats(self):
        report = analyze_data_quality(_clean_df())
        for key in ("rows", "cols", "missing_cells", "missing_pct",
                    "duplicate_rows", "duplicate_pct"):
            assert key in report.stats, f"Missing stat key: '{key}'"

    def test_issues_are_quality_issue_instances(self):
        df = pd.DataFrame({
            "id":    range(100),
            "notes": [None] * 90 + ["x"] * 10,
        })
        report = analyze_data_quality(df)
        for issue in report.issues:
            assert isinstance(issue, QualityIssue)

    def test_every_issue_has_recommendation(self):
        df = pd.DataFrame({
            "id":      [1, 1, 2, 2, 3] * 20,
            "country": ["USA", "Canada"] * 50,
            "notes":   [None] * 80 + ["x"] * 20,
        })
        report = analyze_data_quality(df)
        for issue in report.issues:
            assert issue.recommendation, (
                f"Issue '{issue.title}' has an empty recommendation."
            )

    def test_high_priority_issues_property(self):
        df = pd.DataFrame({
            "id":    range(100),
            "ghost": [None] * 90 + ["x"] * 10,
        })
        report = analyze_data_quality(df)
        for issue in report.high_priority_issues:
            assert issue.severity == SEV_HIGH

    def test_recommendations_property_is_list_of_strings(self):
        report = analyze_data_quality(_clean_df())
        assert isinstance(report.recommendations, list)
        for r in report.recommendations:
            assert isinstance(r, str)

    def test_non_dataframe_raises_type_error(self):
        with pytest.raises(TypeError):
            analyze_data_quality("not a dataframe")  # type: ignore[arg-type]

    def test_non_dataframe_list_raises_type_error(self):
        with pytest.raises(TypeError):
            analyze_data_quality([[1, 2], [3, 4]])  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
# 14. EXISTING FUNCTIONALITY UNAFFECTED
# ═══════════════════════════════════════════════════════════════════════════════

class TestExistingFunctionalityUnaffected:
    """Smoke tests confirming the new engine does not disturb existing modules."""

    def test_cleaning_pipeline_still_works(self):
        from utils.cleaning import run_cleaning_pipeline
        df = pd.DataFrame({
            "Name":  ["Alice", "Bob", "Alice"],
            "Email": ["A@B.COM", "b@b.com", "A@B.COM"],
            "Age":   [30, 25, 30],
        })
        cleaned, report = run_cleaning_pipeline(df, {})
        assert isinstance(cleaned, pd.DataFrame)
        assert report["duplicates_removed"] >= 1

    def test_profiling_still_works(self):
        from utils.profiling import profile_dataframe
        df      = pd.DataFrame({"email": ["a@b.com", "c@d.com"], "age": [25, 30]})
        profile = profile_dataframe(df)
        assert "type_groups" in profile

    def test_helpers_max_rows_still_importable(self):
        from utils.helpers import MAX_ROWS, RowLimitExceeded
        assert MAX_ROWS == 500_000
        assert issubclass(RowLimitExceeded, ValueError)

    def test_data_quality_does_not_modify_input_df(self):
        """The engine must never mutate the caller's DataFrame."""
        df       = _clean_df(50)
        df_copy  = df.copy()
        analyze_data_quality(df)
        pd.testing.assert_frame_equal(df, df_copy)
