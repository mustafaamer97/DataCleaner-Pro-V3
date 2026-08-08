"""
DataCleaner Pro V3 — Commercial Edition
Clean. Analyze. Export.

Main Streamlit application entry point.
All heavy logic lives in utils/ modules.
"""

from __future__ import annotations

import zipfile
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.helpers import (
    validate_uploaded_file,
    load_dataframe,
    df_to_bytes,
    reset_state_if_new_files,
    get_df_memory,
    info_box, success_box, warning_box, error_box, section_header, metric_card,
    MAX_ROWS_FUZZY,
)
from utils.profiling   import profile_dataframe
from utils.cleaning    import run_cleaning_pipeline
from utils.outliers    import detect_all_outliers
from utils.duplicates  import find_fuzzy_duplicates
from utils.pdf_processor import extract_pdf_tables
from utils.exporters   import df_to_excel_bytes, df_to_csv_bytes, build_batch_zip
from utils.reports     import build_text_report, build_comparison_df


# ═══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="DataCleaner Pro V3",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════
#  GLOBAL CSS
# ═══════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Reset & Base ─────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

/* ── Main Header ──────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(102,126,234,0.35);
}
.main-header h1 {
    font-size: 2.4rem; margin: 0;
    font-weight: 800; letter-spacing: -0.5px;
}
.main-header .subtitle {
    font-size: 1.05rem; margin: 0.4rem 0 0; opacity: 0.88;
    font-weight: 400;
}
.version-badge {
    display: inline-block;
    background: rgba(255,255,255,0.22);
    padding: 0.2rem 0.9rem;
    border-radius: 20px;
    font-size: 0.78rem;
    margin-top: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* ── Metric Cards ─────────────────────────────── */
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

/* ── Status Boxes ─────────────────────────────── */
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

/* ── Section Headers ──────────────────────────── */
.section-header {
    font-size: 1.12rem; font-weight: 700; color: #2d3748;
    border-bottom: 2px solid #667eea;
    padding-bottom: 0.35rem; margin: 1.6rem 0 1rem;
}

/* ── Profile Warnings ─────────────────────────── */
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

/* ── Before/After Table ───────────────────────── */
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
.compare-table tr:last-child td     { border-bottom: none; }

/* ── Outlier Badge ────────────────────────────── */
.outlier-badge {
    display: inline-block; background: #fff5f5;
    border: 1px solid #fc8181; border-radius: 6px;
    padding: 0.15rem 0.6rem; font-size: 0.78rem;
    color: #c53030; font-weight: 600; margin: 0.1rem;
}

/* ── Workflow Steps ───────────────────────────── */
.workflow-step {
    display: inline-block; background: #667eea; color: white;
    border-radius: 50%; width: 28px; height: 28px;
    text-align: center; line-height: 28px;
    font-weight: 700; font-size: 0.85rem; margin-right: 0.5rem;
}

/* ── Report Box ───────────────────────────────── */
.report-box {
    background: #1a202c; color: #e2e8f0;
    border-radius: 10px; padding: 1.4rem;
    font-family: 'Courier New', monospace;
    font-size: 0.80rem; line-height: 1.7;
    white-space: pre-wrap; overflow-x: auto;
}

/* ── Progress ─────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2) !important;
    border-radius: 4px !important;
}

/* ── Buttons ──────────────────────────────────── */
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
    transition: all 0.18s ease !important;
}
.stDownloadButton > button:hover {
    opacity: 0.91 !important; transform: translateY(-1px) !important;
}

/* ── Tabs ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.3rem !important;
}

/* ── Sidebar ──────────────────────────────────── */
section[data-testid="stSidebar"] { min-width: 270px !important; }

/* ── Mobile ───────────────────────────────────── */
@media (max-width: 768px) {
    .main-header h1 { font-size: 1.55rem; }
    .main-header .subtitle { font-size: 0.88rem; }
    .metric-card .value { font-size: 1.35rem; }
    .metric-card { padding: 0.9rem 0.5rem; }
    .section-header { font-size: 1rem; }
}
@media (max-width: 480px) {
    .main-header { padding: 1.3rem 1rem; }
    .main-header h1 { font-size: 1.25rem; }
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🧹 DataCleaner Pro</h1>
    <div class="subtitle">Clean. Analyze. Export.</div>
    <div class="version-badge">V3 — Commercial Edition</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧹 DataCleaner Pro")
    st.markdown(
        '<span style="background:#667eea;color:white;padding:2px 10px;'
        'border-radius:12px;font-size:0.73rem;font-weight:600;">V3 Commercial</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── File Upload ───────────────────────────────────────
    st.markdown("### 📁 Upload Files")
    uploaded_files = st.file_uploader(
        "Drag & Drop or Browse",
        type=["csv", "xlsx", "xls", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="CSV, Excel (.xlsx/.xls), PDF — multiple files supported",
    )

    if uploaded_files:
        reset_state_if_new_files(uploaded_files)

    # ── Demo Mode ─────────────────────────────────────────
    st.markdown("---")
    if st.button("🎯 Try Demo Dataset", use_container_width=True):
        st.session_state["demo_mode"] = True
        # Clear previous file state
        for k in [k for k in st.session_state if not k.startswith("_") and k != "demo_mode"]:
            del st.session_state[k]

    if st.session_state.get("demo_mode") and not uploaded_files:
        st.markdown(
            '<div class="info-box">✅ Demo mode active.<br>Using built-in sample dataset.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Clean Settings (only shown when data file present) ─
    data_files = [
        f for f in (uploaded_files or [])
        if Path(f.name).suffix.lower() in (".csv", ".xlsx", ".xls")
    ]
    pdf_files = [
        f for f in (uploaded_files or [])
        if Path(f.name).suffix.lower() == ".pdf"
    ]

    if data_files or st.session_state.get("demo_mode"):
        st.markdown("### ⚙️ Clean Settings")

        fill_strategy = st.selectbox(
            "Missing values:",
            [
                "Auto (Median/Mode)",
                "Fill with 0",
                "Fill with 'Unknown'",
                "Drop rows with missing values",
            ],
        )

        st.markdown("**Cleaning Options:**")
        opt_ftfy       = st.checkbox("🔧 Repair encoding (ftfy)",           value=True)
        opt_emp_cols   = st.checkbox("🗑️ Remove empty columns",             value=True)
        opt_dup_cols   = st.checkbox("🔁 Remove duplicate columns",         value=True)
        opt_const_cols = st.checkbox("📌 Remove constant columns",          value=False)
        opt_snake      = st.checkbox("🐍 Headers → snake_case",             value=False)
        opt_spaces     = st.checkbox("✂️ Trim extra whitespace",            value=True)
        opt_emp_rows   = st.checkbox("🧹 Remove empty rows",                value=True)

        st.markdown("**Advanced Normalization:**")
        opt_emails = st.checkbox("📧 Normalize detected emails",    value=False)
        opt_phones = st.checkbox("📞 Normalize detected phones",    value=False)
        opt_dates  = st.checkbox("📅 Normalize detected dates",     value=False)

        if opt_dates:
            date_fmt = st.selectbox(
                "Target date format:",
                ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"],
            )
        else:
            date_fmt = "%Y-%m-%d"

        st.markdown("**Duplicate Detection:**")
        opt_fuzzy = st.checkbox("🔍 Smart fuzzy duplicate check",   value=False)
        if opt_fuzzy:
            fuzzy_threshold = st.slider(
                "Similarity threshold:", 0.70, 1.00, 0.85, 0.01
            )
        else:
            fuzzy_threshold = 0.85

    if pdf_files:
        st.markdown("### ⚙️ PDF Settings")
        pdf_page_mode = st.radio(
            "Pages:", ["All Pages", "Specific Pages"], horizontal=True
        )
        pdf_pages_input = ""
        if pdf_page_mode == "Specific Pages":
            pdf_pages_input = st.text_input(
                "Page numbers (e.g. 1,3,5):", placeholder="1,2,3"
            )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.74rem;color:#718096;text-align:center;line-height:1.7'>
        🚀 <strong>DataCleaner Pro V3</strong><br>
        Streamlit · Pandas · pdfplumber · ftfy<br>
        <span style='color:#48bb78'>● Commercial Edition</span>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  HELPER: Render Profile Panel
# ═══════════════════════════════════════════════════════════

def render_profile_panel(profile: dict) -> None:
    """Render the smart profiling results."""
    section_header("🔎 Data Quality Analysis")

    # Warnings
    if profile["warnings"]:
        for w in profile["warnings"]:
            st.markdown(f'<div class="profile-warning">⚠️ {w}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="profile-ok">✅ No major data quality issues detected.</div>',
                    unsafe_allow_html=True)

    # Detected types
    type_groups = profile["type_groups"]
    detected = []
    for sem, cols in type_groups.items():
        if cols and sem not in ("numeric", "categorical", "text", "unknown"):
            detected.append(f"✅ **{sem.title()}** columns: {', '.join(cols)}")
    if detected:
        for d in detected:
            st.markdown(f'<div class="profile-ok">{d}</div>', unsafe_allow_html=True)

    # Recommendations
    if profile["recommendations"]:
        st.markdown("**Recommended actions:**")
        for r in profile["recommendations"]:
            st.markdown(f'<div class="profile-rec">▸ {r}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  HELPER: Render Analytics Tabs
# ═══════════════════════════════════════════════════════════

def render_analytics(df: pd.DataFrame, profile: dict, key_prefix: str = "") -> None:
    """Render comprehensive analytics tabs."""
    section_header("📊 Dataset Analytics")

    mem = get_df_memory(df)
    dup_pct  = f"{df.duplicated().mean() * 100:.1f}%"
    miss_pct = f"{df.isnull().mean().mean() * 100:.1f}%"

    c1, c2, c3, c4, c5 = st.columns(5)
    for col_ui, icon, val, lbl in [
        (c1, "💾", mem,                "Memory"),
        (c2, "📏", f"{len(df):,}",     "Rows"),
        (c3, "📋", len(df.columns),    "Columns"),
        (c4, "🔁", dup_pct,            "Duplicate %"),
        (c5, "❓", miss_pct,           "Missing %"),
    ]:
        col_ui.markdown(metric_card(icon, val, lbl), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(
        ["📈 Numeric Summary", "🔤 Column Details", "❓ Missing Values", "🚨 Outliers"]
    )

    with t1:
        num_df = df.select_dtypes(include="number")
        if num_df.empty:
            info_box("No numeric columns found.")
        else:
            st.dataframe(num_df.describe().round(3).T, use_container_width=True)

    with t2:
        col_info_rows = []
        for col in df.columns:
            cp = profile["col_profiles"].get(col, {})
            col_info_rows.append({
                "Column":    col,
                "Type":      str(df[col].dtype),
                "Semantic":  cp.get("semantic", "—"),
                "Non-Null":  int(df[col].notna().sum()),
                "Missing":   cp.get("missing", 0),
                "Missing %": f"{cp.get('missing_pct', 0):.1f}%",
                "Unique":    int(df[col].nunique()),
                "Sample":    ", ".join(cp.get("sample", [])),
            })
        st.dataframe(pd.DataFrame(col_info_rows), use_container_width=True, height=340)

    with t3:
        miss = df.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if miss.empty:
            success_box("✅ No missing values!")
        else:
            miss_df = pd.DataFrame({
                "Column":   miss.index,
                "Missing":  miss.values,
                "Missing %": (miss.values / len(df) * 100).round(2),
            })
            st.dataframe(miss_df, use_container_width=True)

    with t4:
        outlier_report = detect_all_outliers(df)
        if not outlier_report:
            success_box("✅ No outliers detected using IQR method.")
        else:
            for col_name, rep in outlier_report.items():
                with st.expander(
                    f"⚠️ {col_name} — {rep['count']} potential outlier(s)", expanded=False
                ):
                    st.markdown(f"""
                    - **Normal range (IQR):** {rep['lower']} – {rep['upper']}
                    - **Q1 / Q3:** {rep['q1']} / {rep['q3']}
                    - **Count:** {rep['count']}
                    - **Sample values:**
                    """)
                    for v in rep["values"][:10]:
                        st.markdown(
                            f'<span class="outlier-badge">{v}</span>',
                            unsafe_allow_html=True,
                        )
                    warning_box(
                        "⚠️ These values are flagged for review only. "
                        "DataCleaner Pro never automatically deletes outliers."
                    )


# ═══════════════════════════════════════════════════════════
#  HELPER: Render Before/After Comparison
# ═══════════════════════════════════════════════════════════

def render_comparison(report: dict) -> None:
    """Render a before/after comparison table."""
    section_header("📊 Before vs After")

    # encoding_repaired is a count of actions, not a before→after value.
    # We show it as "X repairs performed" rather than a misleading 0.
    enc = report.get("encoding_repaired", 0)
    enc_after_label = "0 (repaired)" if enc > 0 else "0"

    rows = [
        # (Metric,                    Before,                          After)
        ("Rows",
         f"{report['rows_before']:,}",
         f"{report['rows_after']:,}"),

        ("Columns",
         str(report["cols_before"]),
         str(report["cols_after"])),

        ("Duplicate rows",
         str(report["duplicates_removed"]),
         "0"),

        ("Empty rows",
         str(report["empty_rows_removed"]),
         "0"),

        ("Missing values filled",
         str(report["missing_filled"]),
         "0"),

        ("Empty columns removed",
         str(report["empty_cols_removed"]),
         "0"),

        ("Duplicate columns removed",
         str(report["dup_cols_removed"]),
         "0"),

        # Bug fix: show actual repair count; never contradict the summary KPI
        (f"Encoding repairs (ftfy)",
         str(enc),
         enc_after_label),
    ]

    table_html = """
    <table class="compare-table">
        <thead><tr><th>Metric</th><th>Before / Count</th><th>After / Result</th></tr></thead>
        <tbody>
    """
    for metric, before, after in rows:
        table_html += f"<tr><td>{metric}</td><td>{before}</td><td>{after}</td></tr>"
    table_html += "</tbody></table>"

    st.markdown(table_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  HELPER: Build cleaning options dict from sidebar
# ═══════════════════════════════════════════════════════════

def get_cleaning_options(profile: dict) -> dict:
    """Read sidebar controls and build the options dict for run_cleaning_pipeline."""
    # These sidebar variables are set inside the `with st.sidebar` block above.
    # Use st.session_state fallback so the function is safe even if sidebar
    # is not rendered (e.g. demo mode on first load).
    email_cols = profile["type_groups"].get("email", [])
    phone_cols = profile["type_groups"].get("phone", [])
    date_cols  = profile["type_groups"].get("date",  [])

    return {
        "fill_strategy":     st.session_state.get("fill_strategy",  "Auto (Median/Mode)"),
        "use_ftfy":          opt_ftfy   if "opt_ftfy"   in dir() else True,
        "remove_empty_cols": opt_emp_cols  if "opt_emp_cols"  in dir() else True,
        "remove_dup_cols":   opt_dup_cols  if "opt_dup_cols"  in dir() else True,
        "remove_const_cols": opt_const_cols if "opt_const_cols" in dir() else False,
        "snake_case":        opt_snake  if "opt_snake"  in dir() else False,
        "trim_spaces":       opt_spaces if "opt_spaces" in dir() else True,
        "remove_empty_rows": opt_emp_rows if "opt_emp_rows" in dir() else True,
        "normalize_emails":  opt_emails if "opt_emails" in dir() else False,
        "normalize_phones":  opt_phones if "opt_phones" in dir() else False,
        "normalize_dates":   opt_dates  if "opt_dates"  in dir() else False,
        "date_target_fmt":   date_fmt   if "date_fmt"   in dir() else "%Y-%m-%d",
        "email_columns":     email_cols,
        "phone_columns":     phone_cols,
        "date_columns":      date_cols,
    }


# ═══════════════════════════════════════════════════════════
#  WELCOME SCREEN
# ═══════════════════════════════════════════════════════════

demo_mode    = st.session_state.get("demo_mode", False)
has_files    = bool(uploaded_files)
has_data     = bool(data_files) or demo_mode

if not has_files and not demo_mode:
    c1, c2, c3, c4 = st.columns(4)
    features = [
        ("📊", "Multi-File",       "CSV · Excel · PDF"),
        ("✨", "Smart Clean",      "Dedup · Encode · Fill · Normalize"),
        ("🔎", "Data Profiling",   "Types · Outliers · Issues"),
        ("📥", "Pro Export",       "Excel · CSV · ZIP · Reports"),
    ]
    for col_ui, (icon, title, desc) in zip([c1, c2, c3, c4], features):
        col_ui.markdown(
            f"""<div class="metric-card">
                <div style='font-size:2rem'>{icon}</div>
                <div style='font-weight:700;font-size:0.95rem;color:#2d3748;margin:0.4rem 0'>{title}</div>
                <div class="label">{desc}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    info_box(
        "👈 <strong>Upload files from the sidebar</strong> or click "
        "<strong>🎯 Try Demo Dataset</strong> to see DataCleaner Pro V3 in action.<br>"
        "<span style='font-size:0.85rem'>Supports "
        "<code>.csv</code> · <code>.xlsx</code> · <code>.xls</code> · <code>.pdf</code> "
        "— multiple files welcome</span>"
    )

    # Workflow steps
    section_header("📋 How It Works")
    steps_html = ""
    for n, step in enumerate(
        ["Upload", "Profile", "Review Issues", "Configure", "Clean", "Compare", "Export"], 1
    ):
        steps_html += (
            f'<span class="workflow-step">{n}</span>'
            f'<strong>{step}</strong>&nbsp;&nbsp;'
        )
    st.markdown(f"<p style='line-height:2.2'>{steps_html}</p>", unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════
#  LOAD DATA (file or demo)
# ═══════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_demo() -> pd.DataFrame:
    """Load the built-in sample dataset."""
    try:
        return pd.read_csv("sample_data/sample_customers.csv")
    except FileNotFoundError:
        # Inline fallback
        import io as _io
        csv_data = """id,first_name,last_name,email,phone,age,signup_date,country,salary,notes
1,John,Smith,JOHN@EMAIL.COM,+1 (555) 123-4567,34,2024-01-15,USA,55000,Good customer
2,john,smith,john@email.com,+15551234567,34,15/01/2024,USA,55000,Good customer
3,Jane,Doe,jane.doe@company.com,(555) 987-6543,28,Feb 3 2024,Canada,62000,
4,ALICE,JOHNSON, ALICE@DOMAIN.COM ,555.222.3333,999,2024-03-10,UK,75000,Outlier age
5,Bob,Williams,bob_at_email.com,5551112222,41,2024-02-20,Australia,48000,Invalid email
6,Carol,  Brown  ,carol@web.org,+44 20 7946 0958,36,2024-04-01,UK,58000,Extra spaces
7,,,,,,,,,Missing everything
8,Dave,Jones,dave@jones.net,+61 2 9876 5432,52,2024-05-12,Australia,91000,
9,Eve,Wilson,eve@wilson.com,,29,2024-06-30,Canada,67000,No phone
10,Frank,Moore,frank@moore.io,+1 800 555 0199,38,2024-07-04,USA,53000,"""
        return pd.read_csv(_io.StringIO(csv_data))


# ═══════════════════════════════════════════════════════════
#  MAIN TABS
# ═══════════════════════════════════════════════════════════

tab_labels = []
if has_data:
    tab_labels.append("📊 CSV / Excel")
if pdf_files:
    tab_labels.append("📄 PDF Tables")
if (has_data or pdf_files) and len((data_files or []) + (pdf_files or [])) > 1:
    tab_labels.append("⚡ Batch Process")

if not tab_labels:
    tab_labels = ["📊 CSV / Excel"]

all_tabs = st.tabs(tab_labels)


# ═══════════════════════════════════════════════════════════
#  TAB 1: CSV / EXCEL
# ═══════════════════════════════════════════════════════════

if "📊 CSV / Excel" in tab_labels:
    with all_tabs[tab_labels.index("📊 CSV / Excel")]:

        # ── Select or load DataFrame ──────────────────────
        if demo_mode and not data_files:
            chosen_name = "sample_customers.csv"
            with st.spinner("Loading demo dataset…"):
                df_raw = load_demo()
            success_box("🎯 Demo mode — using built-in sample dataset with intentional data quality issues.")

        else:
            if len(data_files) == 1:
                chosen_file = data_files[0]
                chosen_name = chosen_file.name
            else:
                chosen_name = st.selectbox(
                    "📂 Select file:", [f.name for f in data_files], key="sel_data_file"
                )
                chosen_file = next(f for f in data_files if f.name == chosen_name)

            # Validate
            ok, err = validate_uploaded_file(chosen_file)
            if not ok:
                error_box(f"❌ {err}")
                st.stop()

            with st.spinner(f"Loading **{chosen_name}**…"):
                df_raw = load_dataframe(chosen_file.read(), chosen_name)
                chosen_file.seek(0)

            if df_raw is None:
                error_box(
                    f"❌ Could not load **{chosen_name}**. "
                    "Please check the file format and try again."
                )
                st.stop()

        # ── STEP 1: Raw Preview ───────────────────────────
        section_header(f"<span class='workflow-step'>1</span> Raw Data Preview — {chosen_name}")

        r1, r2, r3, r4 = st.columns(4)
        for col_ui, icon, val, lbl in [
            (r1, "📏", f"{len(df_raw):,}",                "Rows"),
            (r2, "📋", len(df_raw.columns),               "Columns"),
            (r3, "❓", f"{df_raw.isnull().sum().sum():,}", "Missing Values"),
            (r4, "🔁", f"{df_raw.duplicated().sum():,}",  "Duplicate Rows"),
        ]:
            col_ui.markdown(metric_card(icon, val, lbl), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_raw.head(10), use_container_width=True, height=260)

        # ── STEP 2: Profiling ─────────────────────────────
        section_header("<span class='workflow-step'>2</span> Data Quality Analysis")

        profile_key = f"profile_{chosen_name}"
        if profile_key not in st.session_state:
            with st.spinner("Profiling dataset…"):
                st.session_state[profile_key] = profile_dataframe(df_raw)

        profile = st.session_state[profile_key]
        render_profile_panel(profile)

        with st.expander("📊 Full Dataset Analytics", expanded=False):
            render_analytics(df_raw, profile, key_prefix="raw")

        # ── STEP 3: Fuzzy Duplicates (optional) ──────────
        if "opt_fuzzy" in dir() and opt_fuzzy:
            section_header("<span class='workflow-step'>3</span> Smart Duplicate Review")

            str_cols = [c for c in df_raw.columns if df_raw[c].dtype == object]
            if not str_cols:
                info_box("No text columns available for fuzzy matching.")
            elif len(df_raw) > MAX_ROWS_FUZZY:
                warning_box(
                    f"⚠️ Dataset has {len(df_raw):,} rows. "
                    f"Fuzzy matching is limited to {MAX_ROWS_FUZZY:,} rows "
                    "to prevent timeouts. Run on a sampled subset."
                )
            else:
                fuzzy_cols = st.multiselect(
                    "Columns to compare:",
                    str_cols,
                    default=str_cols[:2],
                    key="fuzzy_cols",
                )
                if fuzzy_cols and st.button("🔍 Run Fuzzy Check", key="btn_fuzzy"):
                    with st.spinner("Checking for similar records…"):
                        try:
                            fuzzy_df = find_fuzzy_duplicates(
                                df_raw, fuzzy_cols, threshold=fuzzy_threshold
                            )
                        except Exception as e:
                            fuzzy_df = None
                            warning_box(f"Fuzzy check unavailable: {e}")

                    if fuzzy_df is None:
                        warning_box(
                            "⚠️ rapidfuzz is not installed. "
                            "Install it with: `pip install rapidfuzz`"
                        )
                    elif fuzzy_df.empty:
                        success_box("✅ No fuzzy duplicates found above the threshold.")
                    else:
                        warning_box(
                            f"⚠️ {len(fuzzy_df)} potential fuzzy duplicate pair(s) found. "
                            "Review below — nothing is deleted automatically."
                        )
                        st.dataframe(fuzzy_df, use_container_width=True)

        st.markdown("---")

        # ── STEP 4: Configure + Clean ─────────────────────
        section_header("<span class='workflow-step'>4</span> Configure & Clean")

        info_box(
            "📋 Review the analysis above, adjust settings in the sidebar, "
            "then click <strong>Run Auto-Clean</strong>."
        )

        col_run, col_reset = st.columns([4, 1])
        with col_run:
            run_btn = st.button(
                "🚀 Run Auto-Clean",
                type="primary",
                use_container_width=True,
                key=f"run_clean_{chosen_name}",
            )
        with col_reset:
            if st.button("🔄 Reset", use_container_width=True, key=f"reset_{chosen_name}"):
                for k in ["df_clean", "clean_report", "active_file"]:
                    st.session_state.pop(k, None)
                st.rerun()

        if run_btn:
            # Build options safely from sidebar variables
            clean_opts = {
                "fill_strategy":     fill_strategy if "fill_strategy" in dir() else "Auto (Median/Mode)",
                "use_ftfy":          opt_ftfy      if "opt_ftfy"      in dir() else True,
                "remove_empty_cols": opt_emp_cols  if "opt_emp_cols"  in dir() else True,
                "remove_dup_cols":   opt_dup_cols  if "opt_dup_cols"  in dir() else True,
                "remove_const_cols": opt_const_cols if "opt_const_cols" in dir() else False,
                "snake_case":        opt_snake     if "opt_snake"     in dir() else False,
                "trim_spaces":       opt_spaces    if "opt_spaces"    in dir() else True,
                "remove_empty_rows": opt_emp_rows  if "opt_emp_rows"  in dir() else True,
                "normalize_emails":  opt_emails    if "opt_emails"    in dir() else False,
                "normalize_phones":  opt_phones    if "opt_phones"    in dir() else False,
                "normalize_dates":   opt_dates     if "opt_dates"     in dir() else False,
                "date_target_fmt":   date_fmt      if "date_fmt"      in dir() else "%Y-%m-%d",
                "email_columns":     profile["type_groups"].get("email", []),
                "phone_columns":     profile["type_groups"].get("phone", []),
                "date_columns":      profile["type_groups"].get("date",  []),
            }

            prog_ph  = st.empty()
            prog_bar = prog_ph.progress(0, text="Starting…")

            def _cb(frac, msg):
                prog_bar.progress(frac, text=msg)

            try:
                df_clean, report = run_cleaning_pipeline(
                    df_raw, clean_opts, progress_cb=_cb
                )
                report["filename"] = chosen_name
                st.session_state["df_clean"]    = df_clean
                st.session_state["df_raw_snap"] = df_raw
                st.session_state["clean_report"] = report
                st.session_state["active_file"]  = chosen_name
                prog_ph.empty()
                success_box("✅ <strong>Cleaning complete!</strong> See results below.")
            except Exception as e:
                prog_ph.empty()
                error_box(f"❌ Cleaning failed unexpectedly: {e}")

        # ── STEP 5+6: Results & Comparison ───────────────
        if (
            "df_clean" in st.session_state
            and st.session_state.get("active_file") == chosen_name
        ):
            df_clean = st.session_state["df_clean"]
            report   = st.session_state["clean_report"]
            df_snap  = st.session_state.get("df_raw_snap", df_raw)

            # Summary KPIs
            section_header("<span class='workflow-step'>5</span> Cleaning Results")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            for col_ui, val, lbl in [
                (m1, report["duplicates_removed"],  "Dupes Removed"),
                (m2, report["empty_rows_removed"],  "Empty Rows"),
                (m3, report["missing_filled"],      "Nulls Filled"),
                (m4, report["empty_cols_removed"],  "Empty Cols"),
                (m5, report["encoding_repaired"],   "Encoding Fixed"),
                (m6, report["rows_after"],          "Final Rows"),
            ]:
                col_ui.markdown(
                    metric_card("", f"{val:,}", lbl), unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Before / After comparison
            render_comparison(report)

            # Cleaned preview
            section_header("<span class='workflow-step'>6</span> Cleaned Data Preview")
            st.dataframe(df_clean.head(10), use_container_width=True, height=260)

            with st.expander("📊 Cleaned Dataset Analytics", expanded=False):
                clean_profile = profile_dataframe(df_clean)
                render_analytics(df_clean, clean_profile, key_prefix="clean")

            # ── STEP 7: Export ────────────────────────────
            section_header("<span class='workflow-step'>7</span> Export")
            base = Path(chosen_name).stem

            e1, e2, e3 = st.columns(3)
            with e1:
                try:
                    st.download_button(
                        "📥 Download Excel",
                        data=df_to_excel_bytes(df_clean),
                        file_name=f"cleaned_{base}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key=f"dl_xlsx_{base}",
                    )
                except Exception as ex:
                    error_box(f"Excel export error: {ex}")

            with e2:
                try:
                    st.download_button(
                        "📥 Download CSV",
                        data=df_to_csv_bytes(df_clean),
                        file_name=f"cleaned_{base}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"dl_csv_{base}",
                    )
                except Exception as ex:
                    error_box(f"CSV export error: {ex}")

            with e3:
                rpt_text = build_text_report(report, chosen_name, df_snap, df_clean)
                st.download_button(
                    "📋 Download Report",
                    data=rpt_text.encode("utf-8"),
                    file_name=f"report_{base}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key=f"dl_rpt_{base}",
                )

            with st.expander("📋 View Cleaning Report", expanded=False):
                rpt_text = build_text_report(report, chosen_name, df_snap, df_clean)
                st.markdown(
                    f'<div class="report-box">{rpt_text}</div>',
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════
#  TAB 2: PDF TABLES
# ═══════════════════════════════════════════════════════════

if "📄 PDF Tables" in tab_labels:
    with all_tabs[tab_labels.index("📄 PDF Tables")]:

        if len(pdf_files) == 1:
            chosen_pdf  = pdf_files[0]
            chosen_pname = chosen_pdf.name
        else:
            chosen_pname = st.selectbox(
                "📂 Select PDF:", [f.name for f in pdf_files], key="sel_pdf"
            )
            chosen_pdf = next(f for f in pdf_files if f.name == chosen_pname)

        ok, err = validate_uploaded_file(chosen_pdf)
        if not ok:
            error_box(f"❌ {err}")
            st.stop()

        info_box(
            f"📄 <strong>{chosen_pname}</strong> "
            f"&nbsp;|&nbsp; 📦 {chosen_pdf.size / 1024:.1f} KB"
        )

        # Page mode
        try:
            _pdf_page_mode  = pdf_page_mode
            _pdf_pages_input = pdf_pages_input
        except NameError:
            _pdf_page_mode   = "All Pages"
            _pdf_pages_input = ""

        sel_mode  = "all"
        sel_pages = None
        if _pdf_page_mode == "Specific Pages" and _pdf_pages_input:
            try:
                sel_mode  = "specific"
                sel_pages = [int(p.strip()) for p in _pdf_pages_input.split(",") if p.strip().isdigit()]
            except Exception:
                pass

        if st.button(
            "🔍 Extract Tables", type="primary",
            use_container_width=True,
            key=f"pdf_btn_{chosen_pname}",
        ):
            pdf_bytes = chosen_pdf.read()
            chosen_pdf.seek(0)

            prog_ph  = st.empty()
            prog_bar = prog_ph.progress(0, text="Reading PDF…")

            def _pdf_cb(frac, msg):
                prog_bar.progress(frac, text=msg)

            results = extract_pdf_tables(
                pdf_bytes, chosen_pname,
                page_selection=sel_mode,
                specific_pages=sel_pages,
                progress_cb=_pdf_cb,
            )
            prog_ph.empty()
            st.session_state[f"pdf_{chosen_pname}"] = results

        results_key = f"pdf_{chosen_pname}"
        if results_key in st.session_state:
            results = st.session_state[results_key]

            good  = [r for r in results if r["dataframe"] is not None]
            empty = [r for r in results if r["dataframe"] is None]

            p1, p2, p3, p4 = st.columns(4)
            tbl_r = [r for r in good if "text" not in r["method"]]
            txt_r = [r for r in good if "text" in r["method"]]
            for col_ui, icon, val, lbl in [
                (p1, "📊", len(tbl_r),    "Tables Extracted"),
                (p2, "📝", len(txt_r),    "Text-only Pages"),
                (p3, "❌", len(empty),    "Empty Pages"),
                (p4, "📄", len(results),  "Pages Scanned"),
            ]:
                col_ui.markdown(metric_card(icon, str(val), lbl), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if not good:
                warning_box(
                    "⚠️ <strong>No extractable content found.</strong><br>"
                    "This PDF may contain scanned/image-based pages. "
                    "Consider using OCR software first."
                )
            else:
                if empty:
                    info_box(f"ℹ️ {len(empty)} page(s) had no extractable content and were skipped.")

                method_labels = {
                    "extract_tables()":               "🟢 Tier 1",
                    "extract_table()":                "🟡 Tier 2",
                    "extract_text() ← plain text fallback": "🔵 Text",
                }

                for r in good:
                    badge = method_labels.get(r["method"], r["method"])
                    with st.expander(
                        f"Page {r['page']} · Table {r['table_index']} "
                        f"· {r['rows']} rows × {r['cols']} cols · {badge}",
                        expanded=(len(good) <= 4),
                    ):
                        st.dataframe(r["dataframe"], use_container_width=True)
                        d1, d2 = st.columns(2)
                        base_n = f"pdf_p{r['page']}_t{r['table_index']}"
                        with d1:
                            try:
                                st.download_button(
                                    "📥 Excel",
                                    df_to_excel_bytes(r["dataframe"]),
                                    f"{base_n}.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"pdf_xl_{chosen_pname}_{base_n}",
                                    use_container_width=True,
                                )
                            except Exception as ex:
                                error_box(f"Export error: {ex}")
                        with d2:
                            try:
                                st.download_button(
                                    "📥 CSV",
                                    df_to_csv_bytes(r["dataframe"]),
                                    f"{base_n}.csv",
                                    "text/csv",
                                    key=f"pdf_csv_{chosen_pname}_{base_n}",
                                    use_container_width=True,
                                )
                            except Exception as ex:
                                error_box(f"Export error: {ex}")

                # Combined
                if len(good) > 1:
                    section_header("🔗 Combined Export")
                    try:
                        combined = pd.concat(
                            [r["dataframe"].assign(
                                _page=r["page"],
                                _table=r["table_index"],
                                _method=r["method"],
                            ) for r in good],
                            ignore_index=True,
                        )
                        st.dataframe(combined.head(20), use_container_width=True)
                        cc1, cc2 = st.columns(2)
                        stem_p   = Path(chosen_pname).stem
                        with cc1:
                            st.download_button(
                                "📥 All Tables — Excel",
                                df_to_excel_bytes(combined),
                                f"all_tables_{stem_p}.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"comb_xl_{chosen_pname}",
                                use_container_width=True,
                            )
                        with cc2:
                            st.download_button(
                                "📥 All Tables — CSV",
                                df_to_csv_bytes(combined),
                                f"all_tables_{stem_p}.csv",
                                "text/csv",
                                key=f"comb_csv_{chosen_pname}",
                                use_container_width=True,
                            )
                    except Exception as ex:
                        error_box(f"Combined export error: {ex}")


# ═══════════════════════════════════════════════════════════
#  TAB 3: BATCH PROCESSING
# ═══════════════════════════════════════════════════════════

if "⚡ Batch Process" in tab_labels:
    with all_tabs[tab_labels.index("⚡ Batch Process")]:
        section_header("⚡ Batch Processing")

        total_f = len(data_files) + len(pdf_files)
        info_box(
            f"📂 <strong>{total_f} file(s)</strong> ready for batch processing: "
            f"{len(data_files)} data file(s) + {len(pdf_files)} PDF(s)."
        )

        if data_files:
            st.markdown(f"**Data files ({len(data_files)}):**")
            for f in data_files:
                st.markdown(
                    f'<div class="file-badge">📊 {f.name} '
                    f'<span style="color:#718096">({f.size/1024:.1f} KB)</span></div>',
                    unsafe_allow_html=True,
                )

        col_b1, col_b2 = st.columns([3, 1])
        with col_b1:
            batch_btn = st.button(
                f"🚀 Clean All {len(data_files)} Data File(s)",
                type="primary",
                use_container_width=True,
                disabled=not bool(data_files),
            )

        if batch_btn and data_files:
            batch_prog = st.progress(0, text="Batch processing…")
            batch_results: list[tuple[str, pd.DataFrame, dict]] = []
            n_files = len(data_files)

            # Collect clean options
            batch_opts = {
                "fill_strategy":     fill_strategy   if "fill_strategy"   in dir() else "Auto (Median/Mode)",
                "use_ftfy":          opt_ftfy         if "opt_ftfy"        in dir() else True,
                "remove_empty_cols": opt_emp_cols     if "opt_emp_cols"    in dir() else True,
                "remove_dup_cols":   opt_dup_cols     if "opt_dup_cols"    in dir() else True,
                "remove_const_cols": opt_const_cols   if "opt_const_cols"  in dir() else False,
                "snake_case":        opt_snake        if "opt_snake"       in dir() else False,
                "trim_spaces":       opt_spaces       if "opt_spaces"      in dir() else True,
                "remove_empty_rows": opt_emp_rows     if "opt_emp_rows"    in dir() else True,
                "normalize_emails":  False,
                "normalize_phones":  False,
                "normalize_dates":   False,
            }

            errors = []
            for i, f in enumerate(data_files):
                batch_prog.progress(
                    (i + 1) / n_files,
                    text=f"Cleaning {f.name} ({i+1}/{n_files})…",
                )
                ok, err = validate_uploaded_file(f)
                if not ok:
                    errors.append(f"{f.name}: {err}")
                    continue
                try:
                    df_b = load_dataframe(f.read(), f.name)
                    f.seek(0)
                    if df_b is None:
                        errors.append(f"{f.name}: could not load file")
                        continue
                    df_bc, rep_b = run_cleaning_pipeline(df_b, batch_opts)
                    rep_b["filename"] = f.name
                    batch_results.append((f.name, df_bc, rep_b))
                except Exception as ex:
                    errors.append(f"{f.name}: {ex}")

            batch_prog.empty()

            if errors:
                for e in errors:
                    warning_box(f"⚠️ {e}")

            if batch_results:
                st.session_state["batch_results"] = batch_results
                success_box(
                    f"✅ Batch complete! "
                    f"{len(batch_results)}/{n_files} file(s) cleaned successfully."
                )

        if "batch_results" in st.session_state:
            batch_results = st.session_state["batch_results"]

            # Summary table
            summary_rows = []
            for fname, df_bc, rep_b in batch_results:
                summary_rows.append({
                    "File": fname,
                    "Rows (before)": rep_b["rows_before"],
                    "Rows (after)":  rep_b["rows_after"],
                    "Dupes removed": rep_b["duplicates_removed"],
                    "Nulls filled":  rep_b["missing_filled"],
                    "Cols removed":  rep_b["empty_cols_removed"] + rep_b["dup_cols_removed"],
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            # ZIP download
            section_header("📦 Download All Results")
            try:
                zip_bytes = build_batch_zip(
                    batch_results,
                    lambda rep, fn: build_text_report(rep, fn),
                )
                st.download_button(
                    "📦 Download All as ZIP",
                    data=zip_bytes,
                    file_name="datacleaner_pro_batch.zip",
                    mime="application/zip",
                    use_container_width=True,
                    type="primary",
                )
                info_box(
                    "📦 ZIP contains:<br>"
                    "• <code>cleaned_*.xlsx</code> and <code>cleaned_*.csv</code> "
                    "for each file<br>"
                    "• <code>reports/report_*.txt</code> for each file"
                )
            except Exception as ex:
                error_box(f"ZIP creation error: {ex}")

            # Individual file expanders
            st.markdown("<br>", unsafe_allow_html=True)
            for fname, df_bc, rep_b in batch_results:
                base_b = Path(fname).stem
                with st.expander(
                    f"📄 {fname} — {rep_b['rows_after']:,} rows", expanded=False
                ):
                    st.dataframe(df_bc.head(5), use_container_width=True)
                    bc1, bc2, bc3 = st.columns(3)
                    with bc1:
                        st.download_button(
                            "📥 Excel",
                            df_to_excel_bytes(df_bc),
                            f"cleaned_{base_b}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"b_xl_{fname}",
                            use_container_width=True,
                        )
                    with bc2:
                        st.download_button(
                            "📥 CSV",
                            df_to_csv_bytes(df_bc),
                            f"cleaned_{base_b}.csv",
                            "text/csv",
                            key=f"b_csv_{fname}",
                            use_container_width=True,
                        )
                    with bc3:
                        rt = build_text_report(rep_b, fname)
                        st.download_button(
                            "📋 Report",
                            rt.encode("utf-8"),
                            f"report_{base_b}.txt",
                            "text/plain",
                            key=f"b_rpt_{fname}",
                            use_container_width=True,
                        )
