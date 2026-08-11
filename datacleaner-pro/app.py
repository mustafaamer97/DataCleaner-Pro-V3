"""
DataCleaner Pro V3 — Commercial Edition
Clean. Analyze. Export.

Streamlit application entry point.
All business logic lives in utils/ modules — this file is UI only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

# ── Internal imports ──────────────────────────────────────────────────────────
from utils.helpers import (
    MAX_ROWS,
    MAX_ROWS_FUZZY,
    RowLimitExceeded,
    df_to_bytes,          # noqa: F401
    error_box,
    get_df_memory,
    get_excel_sheet_names,
    info_box,
    load_dataframe,
    metric_card,
    reset_state_if_new_files,
    section_header,
    success_box,
    validate_uploaded_file,
    warning_box,
)
from utils.cleaning      import run_cleaning_pipeline
from utils.profiling     import profile_dataframe
from utils.outliers      import detect_all_outliers
from utils.duplicates    import find_fuzzy_duplicates
from utils.pdf_processor import extract_pdf_tables, MAX_PDF_PAGES
from utils.exporters     import build_batch_zip, df_to_csv_bytes, df_to_excel_bytes
from utils.reports       import build_text_report
from utils.data_quality  import analyze_data_quality, QualityReport


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="DataCleaner Pro V3",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.35);
}
.main-header h1 {
    font-size: 2.4rem; margin: 0;
    font-weight: 800; letter-spacing: -0.5px;
}
.main-header .subtitle {
    font-size: 1.05rem; margin: 0.4rem 0 0; opacity: 0.88;
}
.v-badge {
    display: inline-block;
    background: rgba(255,255,255,0.22);
    padding: 0.2rem 0.9rem;
    border-radius: 20px;
    font-size: 0.78rem;
    margin-top: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.2rem 0.8rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    height: 100%;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.10);
}
.metric-card .icon  { font-size: 1.5rem; margin-bottom: 0.3rem; }
.metric-card .value { font-size: 1.8rem; font-weight: 800; color: #667eea; line-height: 1.1; }
.metric-card .label { font-size: 0.79rem; color: #718096; margin-top: 0.3rem; font-weight: 500; }
.success-box {
    background: #f0fff4; border-left: 4px solid #48bb78;
    padding: 0.9rem 1.1rem; border-radius: 8px; margin: 0.8rem 0; font-size: 0.93rem;
}
.info-box {
    background: #ebf8ff; border-left: 4px solid #4299e1;
    padding: 0.9rem 1.1rem; border-radius: 8px; margin: 0.8rem 0; font-size: 0.93rem;
}
.warning-box {
    background: #fffaf0; border-left: 4px solid #ed8936;
    padding: 0.9rem 1.1rem; border-radius: 8px; margin: 0.8rem 0; font-size: 0.93rem;
}
.error-box {
    background: #fff5f5; border-left: 4px solid #fc8181;
    padding: 0.9rem 1.1rem; border-radius: 8px; margin: 0.8rem 0; font-size: 0.93rem;
}
.section-header {
    font-size: 1.12rem; font-weight: 700; color: #2d3748;
    border-bottom: 2px solid #667eea;
    padding-bottom: 0.35rem; margin: 1.6rem 0 1rem;
}
.profile-warning {
    background: #fffaf0; border: 1px solid #fbd38d;
    border-radius: 8px; padding: 0.7rem 1rem;
    margin: 0.4rem 0; font-size: 0.88rem; color: #744210;
}
.profile-ok {
    background: #f0fff4; border: 1px solid #9ae6b4;
    border-radius: 8px; padding: 0.7rem 1rem;
    margin: 0.4rem 0; font-size: 0.88rem; color: #22543d;
}
.profile-rec {
    background: #ebf8ff; border: 1px solid #90cdf4;
    border-radius: 8px; padding: 0.7rem 1rem;
    margin: 0.4rem 0; font-size: 0.88rem; color: #2a4365;
}
.compare-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.88rem; border-radius: 8px; overflow: hidden;
}
.compare-table th {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; padding: 0.7rem 1rem; text-align: left; font-weight: 600;
}
.compare-table td {
    padding: 0.55rem 1rem; border-bottom: 1px solid #e2e8f0;
}
.compare-table tr:nth-child(even) td { background: #f7fafc; }
.compare-table tr:last-child td      { border-bottom: none; }
.outlier-badge {
    display: inline-block; background: #fff5f5;
    border: 1px solid #fc8181; border-radius: 6px;
    padding: 0.15rem 0.6rem; font-size: 0.78rem;
    color: #c53030; font-weight: 600; margin: 0.1rem;
}
.workflow-step {
    display: inline-block; background: #667eea; color: white;
    border-radius: 50%; width: 28px; height: 28px;
    text-align: center; line-height: 28px;
    font-weight: 700; font-size: 0.85rem; margin-right: 0.5rem;
}
.report-box {
    background: #1a202c; color: #e2e8f0;
    border-radius: 10px; padding: 1.4rem;
    font-family: 'Courier New', monospace;
    font-size: 0.80rem; line-height: 1.7;
    white-space: pre-wrap; overflow-x: auto;
}
.file-badge {
    background: #f7fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 0.6rem 0.9rem;
    font-size: 0.82rem; margin-bottom: 0.4rem;
}
.dq-score-card {
    background: white;
    border: 2px solid #667eea;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(102,126,234,0.15);
}
.dq-score-number {
    font-size: 3rem;
    font-weight: 900;
    color: #667eea;
    line-height: 1;
}
.dq-score-label {
    font-size: 1rem;
    font-weight: 600;
    color: #4a5568;
    margin-top: 0.3rem;
}
.dq-category-bar {
    background: #edf2f7;
    border-radius: 8px;
    height: 8px;
    margin-top: 0.3rem;
    overflow: hidden;
}
.stProgress > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2) !important;
    border-radius: 4px !important;
}
.stButton > button {
    border-radius: 9px !important; font-weight: 600 !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.14) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important; border: none !important;
    border-radius: 9px !important; font-weight: 600 !important;
}
.stDownloadButton > button:hover { opacity: 0.91 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600 !important; padding: 0.5rem 1.3rem !important;
}
section[data-testid="stSidebar"] { min-width: 270px !important; }
@media (max-width: 768px) {
    .main-header h1        { font-size: 1.55rem; }
    .main-header .subtitle { font-size: 0.88rem; }
    .metric-card .value    { font-size: 1.35rem; }
    .metric-card           { padding: 0.9rem 0.5rem; }
    .section-header        { font-size: 1rem; }
}
@media (max-width: 480px) {
    .main-header    { padding: 1.3rem 1rem; }
    .main-header h1 { font-size: 1.25rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
<div class="main-header">
    <h1>🧹 DataCleaner Pro</h1>
    <div class="subtitle">Clean. Analyze. Export.</div>
    <div class="v-badge">V3 — Commercial Edition</div>
</div>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

sidebar_opts: dict = {
    "fill_strategy":     "Auto (Median/Mode)",
    "use_ftfy":          True,
    "remove_empty_cols": True,
    "remove_dup_cols":   True,
    "remove_const_cols": False,
    "snake_case":        False,
    "trim_spaces":       True,
    "remove_empty_rows": True,
    "normalize_emails":  False,
    "normalize_phones":  False,
    "normalize_dates":   False,
    "date_target_fmt":   "%Y-%m-%d",
    "fuzzy_check":       False,
    "fuzzy_threshold":   0.85,
    "pdf_page_mode":     "All Pages",
    "pdf_pages_input":   "",
}

with st.sidebar:
    st.markdown("## 🧹 DataCleaner Pro")
    st.markdown(
        '<span style="background:#667eea;color:white;padding:2px 10px;'
        'border-radius:12px;font-size:0.73rem;font-weight:600;">'
        "V3 Commercial</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### 📁 Upload Files")
    uploaded_files = st.file_uploader(
        "Drag & Drop or Browse",
        type=["csv", "xlsx", "xls", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="CSV · Excel (.xlsx / .xls) · PDF — multiple files supported",
    )

    if uploaded_files:
        reset_state_if_new_files(uploaded_files)

    st.markdown("---")

    if st.button("🎯 Try Demo Dataset", use_container_width=True):
        for _k in [k for k in st.session_state if not k.startswith("_")]:
            del st.session_state[_k]
        st.session_state["demo_mode"] = True
        st.rerun()

    if st.session_state.get("demo_mode") and not uploaded_files:
        st.markdown(
            '<div class="info-box" style="font-size:0.82rem">'
            "✅ Demo mode active — using built-in sample dataset.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    data_files: list = [
        f for f in (uploaded_files or [])
        if Path(f.name).suffix.lower() in (".csv", ".xlsx", ".xls")
    ]

    pdf_files: list = [
        f for f in (uploaded_files or [])
        if Path(f.name).suffix.lower() == ".pdf"
    ]

    if data_files or st.session_state.get("demo_mode"):
        st.markdown("### ⚙️ Clean Settings")

        sidebar_opts["fill_strategy"] = st.selectbox(
            "Missing values strategy:",
            [
                "Auto (Median/Mode)",
                "Fill with 0",
                "Fill with 'Unknown'",
                "Drop rows with missing values",
            ],
            help="How to handle NaN / empty cells",
        )

        st.markdown("**Cleaning Options:**")
        sidebar_opts["use_ftfy"]          = st.checkbox("🔧 Repair encoding (ftfy)", value=True)
        sidebar_opts["remove_empty_cols"] = st.checkbox("🗑️ Remove empty columns", value=True)
        sidebar_opts["remove_dup_cols"]   = st.checkbox("🔁 Remove duplicate columns", value=True)
        sidebar_opts["remove_const_cols"] = st.checkbox("📌 Remove constant columns", value=False)
        sidebar_opts["snake_case"]        = st.checkbox("🐍 Headers → snake_case", value=False)
        sidebar_opts["trim_spaces"]       = st.checkbox("✂️ Trim extra whitespace", value=True)
        sidebar_opts["remove_empty_rows"] = st.checkbox("🧹 Remove empty rows", value=True)

        st.markdown("**Advanced Normalization:**")
        sidebar_opts["normalize_emails"] = st.checkbox("📧 Normalize detected emails", value=False)
        sidebar_opts["normalize_phones"] = st.checkbox("📞 Normalize detected phones", value=False)
        sidebar_opts["normalize_dates"]  = st.checkbox("📅 Normalize detected dates", value=False)

        if sidebar_opts["normalize_dates"]:
            sidebar_opts["date_target_fmt"] = st.selectbox(
                "Target date format:",
                ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"],
            )

        st.markdown("**Duplicate Detection:**")
        sidebar_opts["fuzzy_check"] = st.checkbox(
            "🔍 Smart fuzzy duplicate check", value=False
        )

        if sidebar_opts["fuzzy_check"]:
            sidebar_opts["fuzzy_threshold"] = st.slider(
                "Similarity threshold:", 0.70, 1.00, 0.85, 0.01
            )

    if pdf_files:
        st.markdown("### ⚙️ PDF Settings")
        sidebar_opts["pdf_page_mode"] = st.radio(
            "Pages to extract:", ["All Pages", "Specific Pages"], horizontal=True
        )

        if sidebar_opts["pdf_page_mode"] == "Specific Pages":
            sidebar_opts["pdf_pages_input"] = st.text_input(
                "Page numbers (e.g. 1,3,5):",
                placeholder="1,2,3"
            )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.74rem;color:#718096;text-align:center;line-height:1.7'>"
        "🚀 <strong>DataCleaner Pro V3</strong><br>"
        "Streamlit · Pandas · pdfplumber · ftfy<br>"
        "<span style='color:#48bb78'>● Commercial Edition</span>"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY — file bytes cache
# ═══════════════════════════════════════════════════════════════════════════════

def _get_file_bytes(uploaded_file) -> bytes:
    key = f"_bytes_{uploaded_file.name}_{uploaded_file.size}"
    if key not in st.session_state:
        st.session_state[key] = uploaded_file.read()
    return st.session_state[key]


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY — demo dataset
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_demo() -> pd.DataFrame | None:
    try:
        df = pd.read_csv("sample_data/sample_customers.csv")
        if not df.empty:
            return df
    except Exception:
        pass

    import io as _io

    _inline = (
        "id,first_name,last_name,email,phone,age,signup_date,country,salary,notes\n"
        "1,John,Smith,JOHN@EMAIL.COM,+1 (555) 123-4567,34,2024-01-15,USA,55000,Good customer\n"
        "2,john,smith,john@email.com,+15551234567,34,15/01/2024,USA,55000,Good customer\n"
        "3,Jane,Doe,jane.doe@company.com,(555) 987-6543,28,Feb 3 2024,Canada,62000,\n"
        "4,ALICE,JOHNSON, ALICE@DOMAIN.COM ,555.222.3333,999,2024-03-10,UK,75000,Outlier age\n"
        "5,Bob,Williams,bob_at_email.com,5551112222,41,2024-02-20,Australia,48000,Invalid email\n"
        "6,Carol,  Brown  ,carol@web.org,+44 20 7946 0958,36,2024-04-01,UK,58000,Extra spaces\n"
        "7,,,,,,,,,Missing everything\n"
        "8,Dave,Jones,dave@jones.net,+61 2 9876 5432,52,2024-05-12,Australia,91000,\n"
        "9,Eve,Wilson,eve@wilson.com,,29,2024-06-30,Canada,67000,No phone\n"
        "10,Frank,Moore,frank@moore.io,+1 800 555 0199,38,2024-07-04,USA,53000,\n"
    )

    try:
        return pd.read_csv(_io.StringIO(_inline))
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _render_profile_panel(profile: dict) -> None:
    section_header("🔎 Data Quality Analysis")

    if profile["warnings"]:
        for w in profile["warnings"]:
            st.markdown(
                f'<div class="profile-warning">⚠️ {w}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="profile-ok">✅ No major data quality issues detected.</div>',
            unsafe_allow_html=True,
        )

    tg = profile["type_groups"]

    for sem in ("email", "phone", "date", "url", "currency", "id"):
        cols_of_type = tg.get(sem, [])
        if cols_of_type:
            st.markdown(
                f'<div class="profile-ok">'
                f'✅ <strong>{sem.title()}</strong> columns detected: '
                f'{", ".join(cols_of_type)}'
                f"</div>",
                unsafe_allow_html=True,
            )

    if profile["recommendations"]:
        st.markdown("**Recommended actions:**")
        for rec in profile["recommendations"]:
            st.markdown(
                f'<div class="profile-rec">▸ {rec}</div>',
                unsafe_allow_html=True,
            )


def _render_data_quality_intelligence(df: pd.DataFrame, context: str = "") -> None:
    """
    Render the Data Quality Intelligence panel for a given DataFrame.

    context : a short string used to namespace session_state keys so the
              panel can be shown for both raw and cleaned data independently.
    """
    section_header("🧠 Data Quality Intelligence")

    dq_key = f"dq_report_{context}"

    if dq_key not in st.session_state:
        with st.spinner("Analyzing data quality…"):
            try:
                st.session_state[dq_key] = analyze_data_quality(df)
            except Exception as _dqe:
                warning_box(f"Data quality analysis unavailable: {_dqe}")
                return

    report: QualityReport = st.session_state[dq_key]

    # ── Score row ─────────────────────────────────────────────────────────────
    col_score, col_cats = st.columns([1, 3])

    with col_score:
        st.markdown(
            f'<div class="dq-score-card">'
            f'<div style="font-size:2rem">{report.score_emoji}</div>'
            f'<div class="dq-score-number">{report.score}</div>'
            f'<div style="font-size:0.72rem;color:#718096;margin-top:0.2rem">out of 100</div>'
            f'<div class="dq-score-label">{report.score_label}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_cats:
        st.markdown("**Category Scores**")
        cat_labels = {
            "completeness": "✅ Completeness",
            "uniqueness":   "🔁 Uniqueness",
            "consistency":  "🔗 Consistency",
            "validity":     "🔍 Validity",
        }
        for cat_key, cat_label in cat_labels.items():
            val = report.categories.get(cat_key, 0)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.7rem;'
                f'margin-bottom:0.4rem">'
                f'<span style="font-size:0.82rem;width:130px;color:#4a5568">'
                f"{cat_label}</span>"
                f'<div style="flex:1;background:#edf2f7;border-radius:6px;height:10px">'
                f'<div style="width:{val}%;background:linear-gradient(90deg,#667eea,#764ba2);'
                f'height:10px;border-radius:6px"></div></div>'
                f'<span style="font-size:0.82rem;font-weight:700;color:#2d3748;'
                f'width:36px;text-align:right">{val}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Summary stats row ─────────────────────────────────────────────────────
    _s1, _s2, _s3, _s4 = st.columns(4)
    for _col, _icon, _val, _lbl in [
        (_s1, "⚠️",  str(len(report.issues)),             "Issues Found"),
        (_s2, "🔴",  str(len(report.high_priority_issues)), "High Priority"),
        (_s3, "🔑",  str(len(report.key_candidates)),       "Key Candidates"),
        (_s4, "🏷️",  str(len(report.categorical_candidates)), "Categorical Hints"),
    ]:
        _col.markdown(metric_card(_icon, _val, _lbl), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Issues panel ──────────────────────────────────────────────────────────
    if report.issues:
        with st.expander(
            f"🔍 Detected Issues ({len(report.issues)} total, "
            f"{len(report.high_priority_issues)} high priority)",
            expanded=bool(report.high_priority_issues),
        ):
            _sev_color = {
                "HIGH":   "#fc8181",
                "MEDIUM": "#f6ad55",
                "LOW":    "#68d391",
            }
            for issue in report.issues:
                color = _sev_color.get(issue.severity, "#cbd5e0")
                col_tag  = f" · <code>{issue.column}</code>" if issue.column else ""
                st.markdown(
                    f'<div style="border-left:4px solid {color};'
                    f'padding:0.7rem 1rem;margin:0.4rem 0;'
                    f'background:#fafafa;border-radius:0 8px 8px 0">'
                    f'<div style="font-weight:700;font-size:0.88rem;color:#2d3748">'
                    f'<span style="background:{color};color:white;'
                    f'padding:1px 8px;border-radius:10px;font-size:0.72rem;'
                    f'margin-right:0.5rem">{issue.severity}</span>'
                    f"{issue.title}{col_tag}</div>"
                    f'<div style="font-size:0.82rem;color:#4a5568;margin-top:0.3rem">'
                    f"{issue.detail}</div>"
                    f'<div style="font-size:0.80rem;color:#667eea;margin-top:0.3rem;'
                    f'font-style:italic">▸ {issue.recommendation}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
    else:
        success_box("✅ No data quality issues detected. Your dataset looks clean!")

    # ── Key candidates ────────────────────────────────────────────────────────
    if report.key_candidates:
        info_box(
            "🔑 <strong>Potential Key Candidates:</strong> "
            + ", ".join(f"<code>{c}</code>" for c in report.key_candidates)
            + " — these columns have high uniqueness and low missingness."
        )

    # ── Categorical candidates ────────────────────────────────────────────────
    if report.categorical_candidates:
        info_box(
            "🏷️ <strong>Categorical Encoding Candidates:</strong> "
            + ", ".join(f"<code>{c}</code>" for c in report.categorical_candidates)
            + " — converting these columns to <code>category</code> type "
            "can reduce memory usage."
        )


def _render_analytics(df: pd.DataFrame, profile: dict) -> None:
    section_header("📊 Dataset Analytics")

    c1, c2, c3, c4, c5 = st.columns(5)
    _dup_pct  = f"{df.duplicated().mean() * 100:.1f}%"
    _miss_pct = f"{df.isnull().mean().mean() * 100:.1f}%"

    for _col, _icon, _val, _lbl in [
        (c1, "💾", get_df_memory(df), "Memory"),
        (c2, "📏", f"{len(df):,}", "Rows"),
        (c3, "📋", str(len(df.columns)), "Columns"),
        (c4, "🔁", _dup_pct, "Duplicate %"),
        (c5, "❓", _miss_pct, "Missing %"),
    ]:
        _col.markdown(metric_card(_icon, _val, _lbl), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs([
        "📈 Numeric Summary",
        "🔤 Column Details",
        "❓ Missing Values",
        "🚨 Outliers",
    ])

    with t1:
        num_df = df.select_dtypes(include="number")
        if num_df.empty:
            info_box("No numeric columns found.")
        else:
            st.dataframe(num_df.describe().round(3).T, use_container_width=True)

    with t2:
        _rows_list = []
        for col in df.columns:
            cp = profile["col_profiles"].get(col, {})
            _rows_list.append({
                "Column":    col,
                "Dtype":     str(df[col].dtype),
                "Semantic":  cp.get("semantic", "—"),
                "Non-Null":  int(df[col].notna().sum()),
                "Missing":   cp.get("missing", 0),
                "Missing %": f"{cp.get('missing_pct', 0):.1f}%",
                "Unique":    int(df[col].nunique()),
                "Sample":    ", ".join(cp.get("sample", [])),
            })
        st.dataframe(pd.DataFrame(_rows_list), use_container_width=True, height=320)

    with t3:
        miss = df.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if miss.empty:
            success_box("✅ No missing values!")
        else:
            st.dataframe(
                pd.DataFrame({
                    "Column":    miss.index,
                    "Missing":   miss.values,
                    "Missing %": (miss.values / len(df) * 100).round(2),
                }),
                use_container_width=True,
            )

    with t4:
        try:
            _outlier_rep = detect_all_outliers(df)
        except Exception:
            _outlier_rep = {}

        if not _outlier_rep:
            success_box("✅ No outliers detected (IQR method).")
        else:
            for _col_name, _rep in _outlier_rep.items():
                with st.expander(
                    f"⚠️ {_col_name} — {_rep['count']} potential outlier(s)",
                    expanded=False,
                ):
                    st.markdown(
                        f"- **Normal range (IQR):** {_rep['lower']} – {_rep['upper']}\n"
                        f"- **Q1 / Q3:** {_rep['q1']} / {_rep['q3']}\n"
                        f"- **Count:** {_rep['count']}"
                    )
                    for _v in _rep.get("values", [])[:10]:
                        st.markdown(
                            f'<span class="outlier-badge">{_v}</span>',
                            unsafe_allow_html=True,
                        )
                    warning_box(
                        "⚠️ Flagged for review only — "
                        "DataCleaner Pro never deletes outliers automatically."
                    )


def _render_comparison(
    report: dict,
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
) -> None:
    section_header("📊 Before vs After")

    _miss_before = int(df_before.isnull().sum().sum())
    _miss_after  = int(df_after.isnull().sum().sum())
    _dup_before  = int(df_before.duplicated().sum())
    _dup_after   = int(df_after.duplicated().sum())
    _enc         = int(report.get("encoding_repaired", 0))

    _rows = [
        ("Rows",                  f"{len(df_before):,}",    f"{len(df_after):,}"),
        ("Columns",               str(len(df_before.columns)), str(len(df_after.columns))),
        ("Missing values",        f"{_miss_before:,}",      f"{_miss_after:,}"),
        ("Duplicate rows",        f"{_dup_before:,}",       f"{_dup_after:,}"),
        ("Encoding repairs (ftfy)", f"{_enc:,}",            "0 ✓" if _enc > 0 else "0"),
        ("Memory",                get_df_memory(df_before), get_df_memory(df_after)),
    ]

    _html = (
        '<table class="compare-table">'
        "<thead><tr><th>Metric</th><th>Before</th><th>After</th></tr></thead>"
        "<tbody>"
    )
    for _metric, _before, _after in _rows:
        _html += (
            f"<tr><td>{_metric}</td>"
            f"<td>{_before}</td>"
            f"<td>{_after}</td></tr>"
        )
    _html += "</tbody></table>"
    st.markdown(_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  WELCOME SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

_demo_mode = st.session_state.get("demo_mode", False)

if not uploaded_files and not _demo_mode:
    _c1, _c2, _c3, _c4 = st.columns(4)
    _features = [
        ("📊", "Multi-File",    "CSV · Excel · PDF"),
        ("✨", "Smart Clean",   "Dedup · Encode · Fill · Normalize"),
        ("🔎", "Data Profiling","Types · Outliers · Issues"),
        ("📥", "Pro Export",    "Excel · CSV · ZIP · Reports"),
    ]
    for _col, (_icon, _title, _desc) in zip([_c1, _c2, _c3, _c4], _features):
        _col.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:2rem">{_icon}</div>'
            f'<div style="font-weight:700;font-size:0.95rem;'
            f'color:#2d3748;margin:0.4rem 0">{_title}</div>'
            f'<div class="label">{_desc}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    info_box(
        "👈 <strong>Upload files from the sidebar</strong> or click "
        "<strong>🎯 Try Demo Dataset</strong> to get started.<br>"
        "<span style='font-size:0.85rem'>Supports "
        "<code>.csv</code> · <code>.xlsx</code> · <code>.xls</code> · "
        "<code>.pdf</code> — multiple files welcome</span>"
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════

_tab_labels: list[str] = []

if data_files or _demo_mode:
    _tab_labels.append("📊 CSV / Excel")

if pdf_files:
    _tab_labels.append("📄 PDF Tables")

if len(data_files) > 1:
    _tab_labels.append("⚡ Batch Process")

if not _tab_labels:
    _tab_labels = ["📊 CSV / Excel"]

_all_tabs = st.tabs(_tab_labels)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — CSV / EXCEL
# ═══════════════════════════════════════════════════════════════════════════════

if "📊 CSV / Excel" in _tab_labels:
    with _all_tabs[_tab_labels.index("📊 CSV / Excel")]:

        if _demo_mode and not data_files:
            _chosen_name = "sample_customers.csv (Demo)"
            _chosen_ext  = ".csv"

            with st.spinner("Loading demo dataset…"):
                df_raw = _load_demo()

            if df_raw is None:
                error_box("❌ Could not load demo dataset.")
                st.stop()

            success_box(
                "🎯 <strong>Demo mode</strong> — using built-in sample dataset "
                "with intentional data quality issues."
            )

        else:
            if len(data_files) == 1:
                _chosen_file = data_files[0]
            else:
                _sel_name = st.selectbox(
                    "📂 Select file to process:",
                    [f.name for f in data_files],
                    key="sel_data_file",
                )
                _chosen_file = next(f for f in data_files if f.name == _sel_name)

            _chosen_name = _chosen_file.name
            _chosen_ext  = Path(_chosen_name).suffix.lower()

            _ok, _err = validate_uploaded_file(_chosen_file)
            if not _ok:
                error_box(f"❌ {_err}")
                st.stop()

            _file_bytes = _get_file_bytes(_chosen_file)
            _sheet_name: str | None = None

            if _chosen_ext in (".xlsx", ".xls"):
                _sheets = get_excel_sheet_names(_file_bytes, _chosen_name)
                if len(_sheets) > 1:
                    _chosen_sheet = st.selectbox(
                        f"📋 Select sheet ({len(_sheets)} sheets found):",
                        _sheets,
                        key=f"_sheet_{_chosen_name}",
                    )
                    _sheet_name = _chosen_sheet
                elif len(_sheets) == 1:
                    _sheet_name = _sheets[0]
                    info_box(f"📋 Loading sheet: <strong>{_sheet_name}</strong>")

            with st.spinner(f"Loading **{_chosen_name}**…"):
                try:
                    df_raw = load_dataframe(
                        _file_bytes,
                        _chosen_name,
                        sheet_name=_sheet_name,
                    )
                except RowLimitExceeded as _rle:
                    error_box(
                        f"❌ <strong>{_chosen_name}</strong> contains "
                        f"{_rle.row_count:,} rows, which exceeds the "
                        f"maximum supported size of {_rle.max_rows:,} rows. "
                        "Please reduce the dataset before uploading."
                    )
                    st.stop()

            if df_raw is None or df_raw.empty:
                error_box(
                    f"❌ Could not load **{_chosen_name}**. "
                    "Please verify the file is a valid CSV or Excel document."
                )
                st.stop()

        section_header(
            f"<span class='workflow-step'>1</span> "
            f"Raw Data Preview — {_chosen_name}"
        )

        _raw_missing = int(df_raw.isnull().sum().sum())
        _raw_dups    = int(df_raw.duplicated().sum())

        _rc1, _rc2, _rc3, _rc4 = st.columns(4)
        for _col, _icon, _val, _lbl in [
            (_rc1, "📏", f"{len(df_raw):,}", "Rows"),
            (_rc2, "📋", str(len(df_raw.columns)), "Columns"),
            (_rc3, "❓", f"{_raw_missing:,}", "Missing Values"),
            (_rc4, "🔁", f"{_raw_dups:,}", "Duplicate Rows"),
        ]:
            _col.markdown(metric_card(_icon, _val, _lbl), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_raw.head(10), use_container_width=True, height=260)

        # ── Data Quality Intelligence (raw data) ──────────────────────────────
        _render_data_quality_intelligence(df_raw, context=f"raw_{_chosen_name}")

        section_header(
            "<span class='workflow-step'>2</span> Data Quality Analysis"
        )

        _profile_key = f"profile_{_chosen_name}"
        if _profile_key not in st.session_state:
            with st.spinner("Profiling dataset…"):
                try:
                    st.session_state[_profile_key] = profile_dataframe(df_raw)
                except Exception as _e:
                    error_box(f"❌ Profiling failed: {_e}")
                    st.stop()

        _profile: dict = st.session_state[_profile_key]
        _render_profile_panel(_profile)

        with st.expander("📊 Full Dataset Analytics", expanded=False):
            _render_analytics(df_raw, _profile)

        if sidebar_opts["fuzzy_check"]:
            section_header(
                "<span class='workflow-step'>3</span> Smart Duplicate Review"
            )
            _str_cols = [c for c in df_raw.columns if df_raw[c].dtype == object]

            if not _str_cols:
                info_box("No text columns available for fuzzy matching.")
            elif len(df_raw) > MAX_ROWS_FUZZY:
                warning_box(
                    f"⚠️ Dataset has {len(df_raw):,} rows. "
                    f"Fuzzy matching is limited to {MAX_ROWS_FUZZY:,} rows."
                )
            else:
                _fuzzy_cols = st.multiselect(
                    "Columns to compare:",
                    _str_cols,
                    default=_str_cols[:2],
                    key="fuzzy_cols",
                )
                if _fuzzy_cols and st.button("🔍 Run Fuzzy Check", key="btn_fuzzy"):
                    with st.spinner("Checking for similar records…"):
                        try:
                            _fuzzy_result = find_fuzzy_duplicates(
                                df_raw,
                                _fuzzy_cols,
                                threshold=sidebar_opts["fuzzy_threshold"],
                            )
                        except Exception as _fe:
                            _fuzzy_result = None
                            warning_box(f"Fuzzy check error: {_fe}")

                    if _fuzzy_result is None:
                        warning_box(
                            "⚠️ Install rapidfuzz for fuzzy matching: "
                            "`pip install rapidfuzz`"
                        )
                    elif _fuzzy_result.empty:
                        success_box("✅ No fuzzy duplicates found above the threshold.")
                    else:
                        warning_box(
                            f"⚠️ {len(_fuzzy_result)} potential duplicate "
                            "pair(s) found. Review below — nothing is "
                            "deleted automatically."
                        )
                        st.dataframe(_fuzzy_result, use_container_width=True)

        st.markdown("---")

        section_header(
            "<span class='workflow-step'>4</span> Configure & Clean"
        )
        info_box(
            "📋 Review the analysis above, adjust settings in the sidebar, "
            "then click <strong>Run Auto-Clean</strong>."
        )

        _btn_col, _rst_col = st.columns([4, 1])
        with _btn_col:
            _run_btn = st.button(
                "🚀 Run Auto-Clean",
                type="primary",
                use_container_width=True,
                key=f"run_clean_{_chosen_name}",
            )
        with _rst_col:
            if st.button(
                "🔄 Reset",
                use_container_width=True,
                key=f"reset_{_chosen_name}",
            ):
                for _k in ["df_clean", "df_raw_snap", "clean_report", "active_file"]:
                    st.session_state.pop(_k, None)
                st.rerun()

        if _run_btn:
            _clean_opts: dict = {
                "fill_strategy":    sidebar_opts["fill_strategy"],
                "use_ftfy":         sidebar_opts["use_ftfy"],
                "remove_empty_cols":sidebar_opts["remove_empty_cols"],
                "remove_dup_cols":  sidebar_opts["remove_dup_cols"],
                "remove_const_cols":sidebar_opts["remove_const_cols"],
                "snake_case":       sidebar_opts["snake_case"],
                "trim_spaces":      sidebar_opts["trim_spaces"],
                "remove_empty_rows":sidebar_opts["remove_empty_rows"],
                "normalize_emails": sidebar_opts["normalize_emails"],
                "normalize_phones": sidebar_opts["normalize_phones"],
                "normalize_dates":  sidebar_opts["normalize_dates"],
                "date_target_fmt":  sidebar_opts["date_target_fmt"],
                "email_columns":    _profile["type_groups"].get("email", []),
                "phone_columns":    _profile["type_groups"].get("phone", []),
                "date_columns":     _profile["type_groups"].get("date", []),
            }

            _prog_ph  = st.empty()
            _prog_bar = _prog_ph.progress(0, text="Starting…")

            def _progress_cb(frac: float, msg: str) -> None:
                _prog_bar.progress(min(float(frac), 1.0), text=msg)

            try:
                _df_clean, _report = run_cleaning_pipeline(
                    df_raw, _clean_opts, progress_cb=_progress_cb,
                )
                _report["filename"] = _chosen_name
                st.session_state["df_clean"]      = _df_clean
                st.session_state["df_raw_snap"]   = df_raw.copy()
                st.session_state["clean_report"]  = _report
                st.session_state["active_file"]   = _chosen_name
                # Invalidate previous DQ report for cleaned data
                st.session_state.pop(f"dq_report_clean_{_chosen_name}", None)
                _prog_ph.empty()
                success_box("✅ <strong>Cleaning complete!</strong> See results below.")

            except Exception as _ce:
                _prog_ph.empty()
                error_box(
                    f"❌ Cleaning failed: {_ce}<br>"
                    "<span style='font-size:0.82rem'>"
                    "Please check your file and try again. "
                    "If the problem persists, try a different "
                    "missing-value strategy."
                    "</span>"
                )

        _active = st.session_state.get("active_file")

        if "df_clean" in st.session_state and _active == _chosen_name:
            df_clean     = st.session_state["df_clean"]
            _report_data = st.session_state["clean_report"]
            _df_snap     = st.session_state.get("df_raw_snap", df_raw)

            section_header(
                "<span class='workflow-step'>5</span> Cleaning Results"
            )

            _m1, _m2, _m3, _m4, _m5, _m6 = st.columns(6)
            for _col, _val, _lbl in [
                (_m1, _report_data["duplicates_removed"],  "Dupes Removed"),
                (_m2, _report_data["empty_rows_removed"],  "Empty Rows"),
                (_m3, _report_data["missing_filled"],      "Nulls Filled"),
                (_m4, _report_data["empty_cols_removed"],  "Empty Cols"),
                (_m5, _report_data["encoding_repaired"],   "Encoding Fixed"),
                (_m6, _report_data["rows_after"],          "Final Rows"),
            ]:
                _col.markdown(
                    metric_card("", f"{_val:,}", _lbl),
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            _render_comparison(_report_data, _df_snap, df_clean)

            section_header(
                "<span class='workflow-step'>6</span> Cleaned Data Preview"
            )
            st.dataframe(df_clean.head(10), use_container_width=True, height=260)

            # ── Data Quality Intelligence (cleaned data) ──────────────────────
            _render_data_quality_intelligence(
                df_clean,
                context=f"clean_{_chosen_name}",
            )

            with st.expander("📊 Cleaned Dataset Analytics", expanded=False):
                try:
                    _clean_profile = profile_dataframe(df_clean)
                    _render_analytics(df_clean, _clean_profile)
                except Exception as _ae:
                    warning_box(f"Analytics error: {_ae}")

            section_header(
                "<span class='workflow-step'>7</span> Export"
            )
            _base = Path(_chosen_name.replace(" (Demo)", "")).stem
            _e1, _e2, _e3 = st.columns(3)

            with _e1:
                try:
                    st.download_button(
                        label="📥 Download Excel",
                        data=df_to_excel_bytes(df_clean),
                        file_name=f"cleaned_{_base}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key=f"dl_xlsx_{_base}",
                    )
                except Exception as _ex:
                    error_box(f"Excel export error: {_ex}")

            with _e2:
                try:
                    st.download_button(
                        label="📥 Download CSV",
                        data=df_to_csv_bytes(df_clean),
                        file_name=f"cleaned_{_base}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"dl_csv_{_base}",
                    )
                except Exception as _ex:
                    error_box(f"CSV export error: {_ex}")

            with _e3:
                try:
                    _rpt_text = build_text_report(
                        _report_data, _chosen_name, _df_snap, df_clean,
                    )
                    st.download_button(
                        label="📋 Download Report",
                        data=_rpt_text.encode("utf-8"),
                        file_name=f"report_{_base}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key=f"dl_rpt_{_base}",
                    )
                except Exception as _ex:
                    error_box(f"Report export error: {_ex}")

            with st.expander("📋 View Cleaning Report", expanded=False):
                try:
                    _rpt_text = build_text_report(
                        _report_data, _chosen_name, _df_snap, df_clean,
                    )
                    st.markdown(
                        f'<div class="report-box">{_rpt_text}</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as _rx:
                    error_box(f"Report render error: {_rx}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PDF TABLES
# ═══════════════════════════════════════════════════════════════════════════════

if "📄 PDF Tables" in _tab_labels:
    with _all_tabs[_tab_labels.index("📄 PDF Tables")]:

        if len(pdf_files) == 1:
            _chosen_pdf   = pdf_files[0]
            _chosen_pname = _chosen_pdf.name
        else:
            _chosen_pname = st.selectbox(
                "📂 Select PDF:",
                [f.name for f in pdf_files],
                key="sel_pdf",
            )
            _chosen_pdf = next(f for f in pdf_files if f.name == _chosen_pname)

        _pok, _perr = validate_uploaded_file(_chosen_pdf)
        if not _pok:
            error_box(f"❌ {_perr}")
            st.stop()

        info_box(
            f"📄 <strong>{_chosen_pname}</strong>"
            f" &nbsp;|&nbsp; 📦 "
            f"{_chosen_pdf.size / 1024:.1f} KB"
        )

        _sel_mode:  str          = "all"
        _sel_pages: list[int] | None = None

        if sidebar_opts["pdf_page_mode"] == "Specific Pages":
            _raw_pg = sidebar_opts["pdf_pages_input"].strip()
            if _raw_pg:
                try:
                    _sel_mode  = "specific"
                    _sel_pages = [
                        int(p.strip())
                        for p in _raw_pg.split(",")
                        if p.strip().isdigit()
                    ]
                except Exception:
                    pass

        if st.button(
            "🔍 Extract Tables",
            type="primary",
            use_container_width=True,
            key=f"pdf_btn_{_chosen_pname}",
        ):
            _pdf_bytes    = _get_file_bytes(_chosen_pdf)
            _pdf_prog_ph  = st.empty()
            _pdf_prog     = _pdf_prog_ph.progress(0, text="Reading PDF…")

            def _pdf_cb(frac: float, msg: str) -> None:
                _pdf_prog.progress(min(float(frac), 1.0), text=msg)

            try:
                _pdf_results = extract_pdf_tables(
                    _pdf_bytes,
                    _chosen_pname,
                    page_selection=_sel_mode,
                    specific_pages=_sel_pages,
                    progress_cb=_pdf_cb,
                )
                st.session_state[f"pdf_{_chosen_pname}"] = _pdf_results
            except Exception as _pe:
                error_box(f"❌ PDF extraction failed: {_pe}")
            finally:
                _pdf_prog_ph.empty()

        _pdf_key = f"pdf_{_chosen_pname}"

        if _pdf_key in st.session_state:
            _results = st.session_state[_pdf_key]

            # ── Page limit exceeded ───────────────────────────────────────────
            _limit_result = next(
                (r for r in _results if r.get("method") == "page_limit_exceeded"),
                None,
            )
            if _limit_result is not None:
                error_box(
                    f"❌ <strong>{_chosen_pname}</strong> contains "
                    f"{_limit_result['total_pages']:,} pages. "
                    f"DataCleaner Pro supports a maximum of "
                    f"{_limit_result['max_pages']:,} pages in All Pages mode. "
                    "Use <strong>Specific Pages</strong> to select a subset."
                )
                st.stop()

            _good     = [r for r in _results if r["dataframe"] is not None]
            _empty_pg = [r for r in _results if r["dataframe"] is None]
            _tbl_r    = [r for r in _good if "text" not in r["method"]]
            _txt_r    = [r for r in _good if "text" in r["method"]]

            _pp1, _pp2, _pp3, _pp4 = st.columns(4)
            for _col, _icon, _val, _lbl in [
                (_pp1, "📊", str(len(_tbl_r)),    "Tables Extracted"),
                (_pp2, "📝", str(len(_txt_r)),    "Text Pages"),
                (_pp3, "❌", str(len(_empty_pg)), "Empty Pages"),
                (_pp4, "📄", str(len(_results)),  "Pages Scanned"),
            ]:
                _col.markdown(metric_card(_icon, _val, _lbl), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if not _good:
                warning_box(
                    "⚠️ <strong>No extractable content found.</strong><br>"
                    "This PDF may be image-based. "
                    "Consider running OCR software first."
                )
            else:
                if _empty_pg:
                    info_box(
                        f"ℹ️ {len(_empty_pg)} page(s) had no extractable "
                        "content and were skipped."
                    )

                _method_badges = {
                    "extract_tables()":                  "🟢 Tier 1",
                    "extract_table()":                   "🟡 Tier 2",
                    "extract_text() ← plain text fallback": "🔵 Text",
                }

                for _r in _good:
                    _badge = _method_badges.get(_r["method"], _r["method"])
                    with st.expander(
                        f"Page {_r['page']} · Table {_r['table_index']} · "
                        f"{_r['rows']} rows × {_r['cols']} cols · {_badge}",
                        expanded=(len(_good) <= 4),
                    ):
                        st.dataframe(_r["dataframe"], use_container_width=True)
                        _d1, _d2 = st.columns(2)
                        _bn = f"pdf_p{_r['page']}_t{_r['table_index']}"

                        with _d1:
                            try:
                                st.download_button(
                                    "📥 Excel",
                                    df_to_excel_bytes(_r["dataframe"]),
                                    f"{_bn}.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"pdf_xl_{_chosen_pname}_{_bn}",
                                    use_container_width=True,
                                )
                            except Exception as _dx:
                                error_box(f"Export error: {_dx}")

                        with _d2:
                            try:
                                st.download_button(
                                    "📥 CSV",
                                    df_to_csv_bytes(_r["dataframe"]),
                                    f"{_bn}.csv",
                                    "text/csv",
                                    key=f"pdf_csv_{_chosen_pname}_{_bn}",
                                    use_container_width=True,
                                )
                            except Exception as _dx:
                                error_box(f"Export error: {_dx}")

                if len(_good) > 1:
                    section_header("🔗 Combined Export")
                    try:
                        _combined = pd.concat(
                            [
                                _r["dataframe"].assign(
                                    _page=_r["page"],
                                    _table=_r["table_index"],
                                    _method=_r["method"],
                                )
                                for _r in _good
                            ],
                            ignore_index=True,
                        )
                        st.dataframe(_combined.head(20), use_container_width=True)
                        _cc1, _cc2  = st.columns(2)
                        _stem_p = Path(_chosen_pname).stem

                        with _cc1:
                            st.download_button(
                                "📥 All Tables — Excel",
                                df_to_excel_bytes(_combined),
                                f"all_tables_{_stem_p}.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"comb_xl_{_chosen_pname}",
                                use_container_width=True,
                            )
                        with _cc2:
                            st.download_button(
                                "📥 All Tables — CSV",
                                df_to_csv_bytes(_combined),
                                f"all_tables_{_stem_p}.csv",
                                "text/csv",
                                key=f"comb_csv_{_chosen_pname}",
                                use_container_width=True,
                            )
                    except Exception as _cx:
                        error_box(f"Combined export error: {_cx}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — BATCH PROCESS
# ═══════════════════════════════════════════════════════════════════════════════

if "⚡ Batch Process" in _tab_labels:
    with _all_tabs[_tab_labels.index("⚡ Batch Process")]:

        section_header("⚡ Batch Processing")
        _total_f = len(data_files)
        info_box(
            f"📂 <strong>{_total_f} data file(s)</strong> "
            "ready for batch processing."
        )

        for _bf in data_files:
            st.markdown(
                f'<div class="file-badge">📊 {_bf.name} '
                f'<span style="color:#718096">({_bf.size / 1024:.1f} KB)</span></div>',
                unsafe_allow_html=True,
            )

        _batch_btn = st.button(
            f"🚀 Clean All {_total_f} File(s)",
            type="primary",
            use_container_width=True,
            disabled=(_total_f == 0),
        )

        if _batch_btn and data_files:
            _batch_prog    = st.progress(0, text="Batch processing…")
            _batch_results: list[tuple[str, pd.DataFrame, dict]] = []
            _before_map:    dict[str, pd.DataFrame]              = {}
            _errors:        list[str]                            = []

            _batch_opts: dict = {
                "fill_strategy":     sidebar_opts["fill_strategy"],
                "use_ftfy":          sidebar_opts["use_ftfy"],
                "remove_empty_cols": sidebar_opts["remove_empty_cols"],
                "remove_dup_cols":   sidebar_opts["remove_dup_cols"],
                "remove_const_cols": sidebar_opts["remove_const_cols"],
                "snake_case":        sidebar_opts["snake_case"],
                "trim_spaces":       sidebar_opts["trim_spaces"],
                "remove_empty_rows": sidebar_opts["remove_empty_rows"],
                "normalize_emails":  False,
                "normalize_phones":  False,
                "normalize_dates":   False,
            }

            for _i, _bf in enumerate(data_files):
                _batch_prog.progress(
                    (_i + 1) / _total_f,
                    text=f"Cleaning {_bf.name} ({_i + 1}/{_total_f})…",
                )
                _fok, _ferr = validate_uploaded_file(_bf)
                if not _fok:
                    _errors.append(f"{_bf.name}: {_ferr}")
                    continue

                try:
                    _fbytes = _get_file_bytes(_bf)

                    try:
                        _df_b = load_dataframe(_fbytes, _bf.name)
                    except RowLimitExceeded as _rle:
                        _errors.append(
                            f"{_bf.name}: contains {_rle.row_count:,} rows — "
                            f"exceeds maximum of {_rle.max_rows:,} rows"
                        )
                        continue

                    if _df_b is None or _df_b.empty:
                        _errors.append(f"{_bf.name}: could not load file")
                        continue

                    _before_map[_bf.name] = _df_b.copy()
                    _df_bc, _rep_b = run_cleaning_pipeline(_df_b, _batch_opts)
                    _rep_b["filename"] = _bf.name
                    _batch_results.append((_bf.name, _df_bc, _rep_b))

                except Exception as _bex:
                    _errors.append(f"{_bf.name}: {_bex}")

            _batch_prog.empty()

            for _e in _errors:
                warning_box(f"⚠️ {_e}")

            if _batch_results:
                st.session_state["batch_results"] = _batch_results
                st.session_state["batch_before"]  = _before_map
                success_box(
                    f"✅ Batch complete! "
                    f"{len(_batch_results)}/{_total_f} file(s) cleaned."
                )

        if "batch_results" in st.session_state:
            _br = st.session_state["batch_results"]
            _bm = st.session_state.get("batch_before", {})

            _summary = []
            for _fn, _dbc, _rb in _br:
                _summary.append({
                    "File":         _fn,
                    "Rows (before)": _rb["rows_before"],
                    "Rows (after)":  _rb["rows_after"],
                    "Dupes removed": _rb["duplicates_removed"],
                    "Nulls filled":  _rb["missing_filled"],
                    "Cols removed":  (
                        _rb["empty_cols_removed"] + _rb["dup_cols_removed"]
                    ),
                })
            st.dataframe(pd.DataFrame(_summary), use_container_width=True)

            section_header("📦 Download All Results")

            try:
                _zip_bytes = build_batch_zip(
                    _br,
                    build_text_report,
                    before_map=_bm,
                )
                st.download_button(
                    "📦 Download All as ZIP",
                    data=_zip_bytes,
                    file_name="datacleaner_pro_batch.zip",
                    mime="application/zip",
                    use_container_width=True,
                    type="primary",
                )
                info_box(
                    "📦 ZIP contains:<br>"
                    "• <code>cleaned_*.xlsx</code> and "
                    "<code>cleaned_*.csv</code> for each file<br>"
                    "• <code>reports/report_*.txt</code> for each file"
                )
            except Exception as _zex:
                error_box(f"ZIP creation error: {_zex}")

            st.markdown("<br>", unsafe_allow_html=True)

            for _fn, _dbc, _rb in _br:
                _base_b   = Path(_fn).stem
                _df_before = _bm.get(_fn)

                with st.expander(
                    f"📄 {_fn} — {_rb['rows_after']:,} rows",
                    expanded=False,
                ):
                    st.dataframe(_dbc.head(5), use_container_width=True)
                    _bc1, _bc2, _bc3 = st.columns(3)

                    with _bc1:
                        st.download_button(
                            "📥 Excel",
                            df_to_excel_bytes(_dbc),
                            f"cleaned_{_base_b}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"b_xl_{_fn}",
                            use_container_width=True,
                        )
                    with _bc2:
                        st.download_button(
                            "📥 CSV",
                            df_to_csv_bytes(_dbc),
                            f"cleaned_{_base_b}.csv",
                            "text/csv",
                            key=f"b_csv_{_fn}",
                            use_container_width=True,
                        )
                    with _bc3:
                        try:
                            _rt = build_text_report(_rb, _fn, _df_before, _dbc)
                        except Exception:
                            _rt = build_text_report(_rb, _fn)

                        st.download_button(
                            "📋 Report",
                            _rt.encode("utf-8"),
                            f"report_{_base_b}.txt",
                            "text/plain",
                            key=f"b_rpt_{_fn}",
                            use_container_width=True,
                        )
