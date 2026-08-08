"""
DataCleaner Pro V3 — Commercial Edition
Clean. Analyze. Export.

Entry point for the Streamlit application.
All heavy logic lives in utils/ modules.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.helpers import (
    validate_uploaded_file,
    load_dataframe,
    get_excel_sheet_names,
    df_to_bytes,
    reset_state_if_new_files,
    get_df_memory,
    file_signature,
    MAX_ROWS_FUZZY,
)
from utils.profiling   import profile_dataframe
from utils.cleaning    import run_cleaning_pipeline
from utils.outliers    import detect_all_outliers
from utils.duplicates  import find_fuzzy_duplicates
from utils.pdf_processor import extract_pdf_tables
from utils.exporters   import df_to_excel_bytes, df_to_csv_bytes, build_batch_zip
from utils.reports     import build_text_report


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
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

/* ── Header ──────────────────────────────────── */
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
    font-size: 2.2rem; margin: 0;
    font-weight: 800; letter-spacing: -0.5px;
}
.main-header .subtitle {
    font-size: 1rem; margin: 0.4rem 0 0; opacity: 0.88;
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
    padding: 1.1rem 0.7rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    height: 100%;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.10);
}
.metric-card .icon  { font-size: 1.4rem; margin-bottom: 0.3rem; }
.metric-card .value { font-size: 1.7rem; font-weight: 800; color: #667eea; line-height: 1.1; }
.metric-card .label { font-size: 0.78rem; color: #718096; margin-top: 0.3rem; font-weight: 500; }

/* ── Status Boxes ─────────────────────────────── */
.success-box {
    background: #f0fff4; border-left: 4px solid #48bb78;
    padding: 0.9rem 1.1rem; border-radius: 8px; margin: 0.7rem 0; font-size: 0.92rem;
}
.info-box {
    background: #ebf8ff; border-left: 4px solid #4299e1;
    padding: 0.9rem 1.1rem; border-radius: 8px; margin: 0.7rem 0; font-size: 0.92rem;
}
.warning-box {
    background: #fffaf0; border-left: 4px solid #ed8936;
    padding: 0.9rem 1.1rem; border-radius: 8px; margin: 0.7rem 0; font-size: 0.92rem;
}
.error-box {
    background: #fff5f5; border-left: 4px solid #fc8181;
    padding: 0.9rem 1.1rem; border-radius: 8px; margin: 0.7rem 0; font-size: 0.92rem;
}

/* ── Section Headers ──────────────────────────── */
.section-header {
    font-size: 1.08rem; font-weight: 700; color: #2d3748;
    border-bottom: 2px solid #667eea;
    padding-bottom: 0.35rem; margin: 1.5rem 0 0.9rem;
}

/* ── Profile Items ────────────────────────────── */
.profile-warning {
    background: #fffaf0; border: 1px solid #fbd38d;
    border-radius: 8px; padding: 0.65rem 1rem;
    margin: 0.35rem 0; font-size: 0.87rem; color: #744210;
}
.profile-ok {
    background: #f0fff4; border: 1px solid #9ae6b4;
    border-radius: 8px; padding: 0.65rem 1rem;
    margin: 0.35rem 0; font-size: 0.87rem; color: #22543d;
}
.profile-rec {
    background: #ebf8ff; border: 1px solid #90cdf4;
    border-radius: 8px; padding: 0.65rem 1rem;
    margin: 0.35rem 0; font-size: 0.87rem; color: #2a4365;
}

/* ── Before/After Table ───────────────────────── */
.compare-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.87rem; border-radius: 8px; overflow: hidden;
}
.compare-table th {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; padding: 0.65rem 1rem; text-align: left; font-weight: 600;
}
.compare-table td {
    padding: 0.5rem 1rem; border-bottom: 1px solid #e2e8f0;
}
.compare-table tr:nth-child(even) td { background: #f7fafc; }
.compare-table tr:last-child td      { border-bottom: none; }

/* ── Outlier Badge ────────────────────────────── */
.outlier-badge {
    display: inline-block; background: #fff5f5;
    border: 1px solid #fc8181; border-radius: 6px;
    padding: 0.15rem 0.6rem; font-size: 0.77rem;
    color: #c53030; font-weight: 600; margin: 0.1rem;
}

/* ── Report Box ───────────────────────────────── */
.report-box {
    background: #1a202c; color: #e2e8f0;
    border-radius: 10px; padding: 1.3rem;
    font-family: 'Courier New', monospace;
    font-size: 0.79rem; line-height: 1.7;
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
    width: 100%;
}
.stDownloadButton > button:hover {
    opacity: 0.91 !important; transform: translateY(-1px) !important;
}

/* ── Tabs ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.1rem !important;
}

/* ── Sidebar ──────────────────────────────────── */
section[data-testid="stSidebar"] { min-width: 265px !important; }

/* ── Mobile ───────────────────────────────────── */
@media (max-width: 768px) {
    .main-header h1 { font-size: 1.5rem; }
    .main-header .subtitle { font-size: 0.86rem; }
    .metric-card .value { font-size: 1.3rem; }
    .metric-card { padding: 0.85rem 0.5rem; }
    .section-header { font-size: 0.97rem; }
}
@media (max-width: 480px) {
    .main-header { padding: 1.2rem 1rem; }
    .main-header h1 { font-size: 1.2rem; }
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  SMALL UI HELPERS
# ═══════════════════════════════════════════════════════════

def _info(text: str)    -> None: st.markdown(f'<div class="info-box">{text}</div>',    unsafe_allow_html=True)
def _success(text: str) -> None: st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)
def _warning(text: str) -> None: st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)
def _error(text: str)   -> None: st.markdown(f'<div class="error-box">{text}</div>',   unsafe_allow_html=True)
def _sh(text: str)      -> None: st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

def _card(icon: str, value: str, label: str) -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="icon">{icon}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="label">{label}</div>'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════
#  CACHED LOADERS
# ═══════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _cached_load(file_bytes: bytes, filename: str, sheet_name: str | int = 0) -> pd.DataFrame | None:
    """Cache-wrapped DataFrame loader."""
    return load_dataframe(file_bytes, filename, sheet_name=sheet_name)


@st.cache_data(show_spinner=False)
def _cached_sheet_names(file_bytes: bytes, filename: str) -> list[str]:
    """Cache-wrapped Excel sheet name reader."""
    return get_excel_sheet_names(file_bytes, filename)


@st.cache_data(show_spinner=False)
def _load_demo() -> pd.DataFrame:
    """Load the built-in sample dataset."""
    try:
        return pd.read_csv("sample_data/sample_customers.csv")
    except FileNotFoundError:
        import io as _io
        _CSV = """\
id,first_name,last_name,email,phone,age,signup_date,country,salary,notes
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
        return pd.read_csv(_io.StringIO(_CSV))


# ═══════════════════════════════════════════════════════════
#  PROFILING PANEL
# ═══════════════════════════════════════════════════════════

def render_profile_panel(profile: dict) -> None:
    """Render the smart data quality analysis section."""
    _sh("🔎 Data Quality Analysis")

    if profile["warnings"]:
        for w in profile["warnings"]:
            st.markdown(f'<div class="profile-warning">⚠️ {w}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="profile-ok">✅ No major data quality issues detected.</div>',
                    unsafe_allow_html=True)

    # Detected semantic types (interesting ones only)
    tg = profile["type_groups"]
    for sem, cols in tg.items():
        if cols and sem not in ("numeric", "categorical", "text", "unknown"):
            label = sem.title()
            st.markdown(
                f'<div class="profile-ok">✅ <strong>{label}</strong> '
                f'column(s): {", ".join(cols)}</div>',
                unsafe_allow_html=True,
            )

    if profile["recommendations"]:
        st.markdown("**Recommended actions:**")
        for r in profile["recommendations"]:
            st.markdown(f'<div class="profile-rec">▸ {r}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  ANALYTICS TABS
# ═══════════════════════════════════════════════════════════

def render_analytics(df: pd.DataFrame, profile: dict) -> None:
    """Render dataset analytics tabs."""
    _sh("📊 Dataset Analytics")

    mem      = get_df_memory(df)
    dup_pct  = f"{df.duplicated().mean() * 100:.1f}%"
    miss_pct = f"{df.isnull().mean().mean() * 100:.1f}%"

    c1, c2, c3, c4, c5 = st.columns(5)
    for col_ui, icon, val, lbl in [
        (c1, "💾", mem,             "Memory"),
        (c2, "📏", f"{len(df):,}", "Rows"),
        (c3, "📋", len(df.columns), "Columns"),
        (c4, "🔁", dup_pct,         "Duplicate %"),
        (c5, "❓", miss_pct,        "Missing %"),
    ]:
        col_ui.markdown(_card(icon, val, lbl), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(
        ["📈 Numeric Summary", "🔤 Column Details", "❓ Missing Values", "🚨 Outliers"]
    )

    with t1:
        num_df = df.select_dtypes(include="number")
        if num_df.empty:
            _info("No numeric columns found.")
        else:
            st.dataframe(num_df.describe().round(3).T, use_container_width=True)

    with t2:
        rows = []
        for col in df.columns:
            cp = profile["col_profiles"].get(col, {})
            rows.append({
                "Column":    col,
                "Type":      str(df[col].dtype),
                "Semantic":  cp.get("semantic", "—"),
                "Non-Null":  int(df[col].notna().sum()),
                "Missing":   cp.get("missing", 0),
                "Missing %": f"{cp.get('missing_pct', 0):.1f}%",
                "Unique":    int(df[col].nunique()),
                "Sample":    ", ".join(cp.get("sample", [])),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=320)

    with t3:
        miss = df.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if miss.empty:
            _success("✅ No missing values!")
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
            _success("✅ No outliers detected (IQR method).")
        else:
            for col_name, rep in outlier_report.items():
                with st.expander(
                    f"⚠️ {col_name} — {rep['count']} potential outlier(s)", expanded=False
                ):
                    st.markdown(
                        f"- **Normal range (IQR):** {rep['lower']} – {rep['upper']}\n"
                        f"- **Q1 / Q3:** {rep['q1']} / {rep['q3']}\n"
                        f"- **Outlier count:** {rep['count']}"
                    )
                    for v in rep["values"][:10]:
                        st.markdown(
                            f'<span class="outlier-badge">{v}</span>',
                            unsafe_allow_html=True,
                        )
                    _warning(
                        "⚠️ These values are flagged for review only. "
                        "DataCleaner Pro never auto-deletes outliers."
                    )


# ═══════════════════════════════════════════════════════════
#  BEFORE / AFTER COMPARISON
# ═══════════════════════════════════════════════════════════

def render_comparison(report: dict) -> None:
    """Render a before/after comparison table from the cleaning report."""
    _sh("📊 Before vs After")

    enc = report.get("encoding_repaired", 0)

    rows = [
        ("Rows",                        f"{report['rows_before']:,}",      f"{report['rows_after']:,}"),
        ("Columns",                      str(report["cols_before"]),         str(report["cols_after"])),
        ("Duplicate rows removed",       str(report["duplicates_removed"]),  "0 ✓"),
        ("Empty rows removed",           str(report["empty_rows_removed"]),  "0 ✓"),
        ("Missing values filled",        str(report["missing_filled"]),      "0 ✓"),
        ("Rows dropped (missing)",       str(report["missing_dropped_rows"]), "n/a"),
        ("Empty columns removed",        str(report["empty_cols_removed"]),  "0 ✓"),
        ("Duplicate columns removed",    str(report["dup_cols_removed"]),    "0 ✓"),
        (f"Encoding repairs (ftfy)",     str(enc),                           f"{enc} cells fixed" if enc else "0"),
    ]

    html = (
        '<table class="compare-table">'
        "<thead><tr><th>Metric</th><th>Before / Count</th><th>After / Result</th></tr></thead>"
        "<tbody>"
    )
    for metric, before, after in rows:
        html += f"<tr><td>{metric}</td><td>{before}</td><td>{after}</td></tr>"
    html += "</tbody></table>"

    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  SIDEBAR CLEANING OPTIONS  (returns a plain dict — no fragile dir() hacks)
# ═══════════════════════════════════════════════════════════

def _collect_sidebar_options(profile: dict | None = None) -> dict:
    """
    Read all sidebar cleaning controls and return a clean options dict.

    This function is the single source of truth for cleaning options.
    It replaces the fragile `opt_x if "opt_x" in dir() else default` pattern.
    All values come from st.session_state keys set by the sidebar widgets.
    """
    email_cols = profile["type_groups"].get("email", []) if profile else []
    phone_cols = profile["type_groups"].get("phone", []) if profile else []
    date_cols  = profile["type_groups"].get("date",  []) if profile else []

    return {
        "fill_strategy":     st.session_state.get("sb_fill_strategy",  "Auto (Median/Mode)"),
        "use_ftfy":          st.session_state.get("sb_use_ftfy",        True),
        "remove_empty_cols": st.session_state.get("sb_empty_cols",      True),
        "remove_dup_cols":   st.session_state.get("sb_dup_cols",        True),
        "remove_const_cols": st.session_state.get("sb_const_cols",      False),
        "snake_case":        st.session_state.get("sb_snake_case",      False),
        "trim_spaces":       st.session_state.get("sb_trim_spaces",     True),
        "remove_empty_rows": st.session_state.get("sb_empty_rows",      True),
        "normalize_emails":  st.session_state.get("sb_norm_emails",     False),
        "normalize_phones":  st.session_state.get("sb_norm_phones",     False),
        "normalize_dates":   st.session_state.get("sb_norm_dates",      False),
        "date_target_fmt":   st.session_state.get("sb_date_fmt",        "%Y-%m-%d"),
        "email_columns":     email_cols,
        "phone_columns":     phone_cols,
        "date_columns":      date_cols,
    }


# ═══════════════════════════════════════════════════════════
#  PAGE HEADER
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
        "Drop files here or click Browse",
        type=["csv", "xlsx", "xls", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="CSV, Excel (.xlsx / .xls), PDF — multiple files supported",
    )

    if uploaded_files:
        reset_state_if_new_files(uploaded_files, st.session_state)

    # ── Demo Mode ─────────────────────────────────────────
    st.markdown("---")
    if st.button("🎯 Try Demo Dataset", use_container_width=True):
        # Isolate demo mode completely from any file-upload state
        keys_to_clear = [
            k for k in st.session_state
            if k not in {"_file_sig"} and not k.startswith("_")
        ]
        for k in keys_to_clear:
            del st.session_state[k]
        st.session_state["demo_mode"] = True

    if st.session_state.get("demo_mode") and not uploaded_files:
        _info("✅ Demo mode — built-in sample dataset loaded.")

    st.markdown("---")

    # ── Derive file groups ────────────────────────────────
    data_files = [
        f for f in (uploaded_files or [])
        if Path(f.name).suffix.lower() in (".csv", ".xlsx", ".xls")
    ]
    pdf_files = [
        f for f in (uploaded_files or [])
        if Path(f.name).suffix.lower() == ".pdf"
    ]

    has_data = bool(data_files) or st.session_state.get("demo_mode", False)

    # ── Cleaning Options (only when data files present) ───
    if has_data:
        st.markdown("### ⚙️ Clean Settings")

        st.selectbox(
            "Missing values:",
            [
                "Auto (Median/Mode)",
                "Fill with 0",
                "Fill with 'Unknown'",
                "Drop rows with missing values",
            ],
            key="sb_fill_strategy",
        )

        st.markdown("**Options:**")
        st.checkbox("🔧 Repair encoding (ftfy)",        value=True,  key="sb_use_ftfy")
        st.checkbox("🗑️ Remove empty columns",          value=True,  key="sb_empty_cols")
        st.checkbox("🔁 Remove duplicate columns",      value=True,  key="sb_dup_cols")
        st.checkbox("📌 Remove constant columns",       value=False, key="sb_const_cols")
        st.checkbox("🐍 Headers → snake_case",          value=False, key="sb_snake_case")
        st.checkbox("✂️ Trim extra whitespace",         value=True,  key="sb_trim_spaces")
        st.checkbox("🧹 Remove empty rows",             value=True,  key="sb_empty_rows")

        st.markdown("**Normalization:**")
        st.checkbox("📧 Normalize detected emails",     value=False, key="sb_norm_emails")
        st.checkbox("📞 Normalize detected phones",     value=False, key="sb_norm_phones")
        st.checkbox("📅 Normalize detected dates",      value=False, key="sb_norm_dates")

        if st.session_state.get("sb_norm_dates"):
            st.selectbox(
                "Target date format:",
                ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"],
                key="sb_date_fmt",
            )

        st.markdown("**Duplicate Detection:**")
        st.checkbox("🔍 Smart fuzzy duplicate check",   value=False, key="sb_fuzzy")
        if st.session_state.get("sb_fuzzy"):
            st.slider("Similarity threshold:", 0.70, 1.00, 0.85, 0.01, key="sb_fuzzy_threshold")

    # ── PDF Options ───────────────────────────────────────
    if pdf_files:
        st.markdown("### ⚙️ PDF Settings")
        st.radio("Pages:", ["All Pages", "Specific Pages"], horizontal=True, key="sb_pdf_pages")
        if st.session_state.get("sb_pdf_pages") == "Specific Pages":
            st.text_input("Page numbers (e.g. 1,3,5):", placeholder="1,2,3", key="sb_pdf_page_nums")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.73rem;color:#718096;text-align:center;line-height:1.7'>
        🚀 <strong>DataCleaner Pro V3</strong><br>
        Streamlit · Pandas · pdfplumber · ftfy<br>
        <span style='color:#48bb78'>● Commercial Edition</span>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  WELCOME SCREEN
# ═══════════════════════════════════════════════════════════

demo_mode = st.session_state.get("demo_mode", False)
has_files = bool(uploaded_files)

if not has_files and not demo_mode:
    c1, c2, c3, c4 = st.columns(4)
    for col_ui, icon, title, desc in [
        (c1, "📊", "Multi-File",    "CSV · Excel · PDF"),
        (c2, "✨", "Smart Clean",   "Dedup · Encode · Fill · Normalize"),
        (c3, "🔎", "Data Profile",  "Types · Outliers · Issues"),
        (c4, "📥", "Pro Export",    "Excel · CSV · ZIP · Reports"),
    ]:
        col_ui.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:2rem">{icon}</div>'
            f'<div style="font-weight:700;font-size:0.93rem;color:#2d3748;margin:0.4rem 0">{title}</div>'
            f'<div class="label">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)
    _info(
        "👈 <strong>Upload files from the sidebar</strong> or click "
        "<strong>🎯 Try Demo Dataset</strong> to get started.<br>"
        "<span style='font-size:0.84rem'>Supports "
        "<code>.csv</code> · <code>.xlsx</code> · <code>.xls</code> · <code>.pdf</code>"
        " — multiple files welcome</span>"
    )
    st.stop()


# ═══════════════════════════════════════════════════════════
#  BUILD TAB LIST
# ═══════════════════════════════════════════════════════════

tab_labels: list[str] = []
if has_data:
    tab_labels.append("📊 CSV / Excel")
if pdf_files:
    tab_labels.append("📄 PDF Tables")
if len(data_files) > 1:
    tab_labels.append("⚡ Batch Process")

if not tab_labels:
    tab_labels = ["📊 CSV / Excel"]

all_tabs = st.tabs(tab_labels)


# ═══════════════════════════════════════════════════════════
#  TAB: CSV / EXCEL
# ═══════════════════════════════════════════════════════════

if "📊 CSV / Excel" in tab_labels:
    with all_tabs[tab_labels.index("📊 CSV / Excel")]:

        # ── Load DataFrame ────────────────────────────────
        if demo_mode and not data_files:
            chosen_name = "sample_customers.csv"
            with st.spinner("Loading demo dataset…"):
                df_raw = _load_demo()
            _success("🎯 Demo mode — using built-in sample dataset with intentional data issues.")

        else:
            # File selection
            if len(data_files) == 1:
                chosen_file = data_files[0]
                chosen_name = chosen_file.name
            else:
                chosen_name = st.selectbox(
                    "📂 Select file to process:",
                    [f.name for f in data_files],
                    key="sel_data_file",
                )
                chosen_file = next(f for f in data_files if f.name == chosen_name)

            # Validate
            ok, err_msg = validate_uploaded_file(chosen_file)
            if not ok:
                _error(f"❌ {err_msg}")
                st.stop()

            # Read bytes once — reuse for both sheet detection and loading
            file_bytes = chosen_file.read()
            chosen_file.seek(0)

            # Excel sheet selector
            ext = Path(chosen_name).suffix.lower()
            sheet_choice: str | int = 0
            if ext in (".xlsx", ".xls"):
                sheet_names = _cached_sheet_names(file_bytes, chosen_name)
                if len(sheet_names) > 1:
                    sheet_choice = st.selectbox(
                        f"📋 Select sheet ({len(sheet_names)} available):",
                        sheet_names,
                        key=f"sheet_{chosen_name}",
                    )
                    _info(f"ℹ️ Loading sheet: **{sheet_choice}**")
                elif len(sheet_names) == 1:
                    sheet_choice = sheet_names[0]

            with st.spinner(f"Loading **{chosen_name}**…"):
                df_raw = _cached_load(file_bytes, chosen_name, sheet_name=sheet_choice)

            if df_raw is None:
                _error(
                    f"❌ Could not load **{chosen_name}**. "
                    "Please check that the file is a valid CSV or Excel file "
                    "and try again."
                )
                st.stop()

        # Guard against empty DataFrame
        if df_raw is None or df_raw.empty:
            _warning("⚠️ The file loaded but contains no data.")
            st.stop()

        # ── STEP 1: Raw Preview ───────────────────────────
        _sh(f"<span>📋 1. Raw Data — {chosen_name}</span>")

        k1, k2, k3, k4 = st.columns(4)
        for col_ui, icon, val, lbl in [
            (k1, "📏", f"{len(df_raw):,}",                "Rows"),
            (k2, "📋", len(df_raw.columns),               "Columns"),
            (k3, "❓", f"{df_raw.isnull().sum().sum():,}", "Missing Values"),
            (k4, "🔁", f"{df_raw.duplicated().sum():,}",  "Duplicate Rows"),
        ]:
            col_ui.markdown(_card(icon, val, lbl), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_raw.head(10), use_container_width=True, height=260)

        # ── STEP 2: Profile ───────────────────────────────
        _sh("🔎 2. Data Quality Analysis")

        profile_key = f"profile_{chosen_name}"
        # Recompute profile if stale (e.g. after file switch)
        if profile_key not in st.session_state:
            with st.spinner("Profiling dataset…"):
                try:
                    st.session_state[profile_key] = profile_dataframe(df_raw)
                except Exception as e:
                    st.session_state[profile_key] = None
                    _error(f"❌ Profiling failed: {e}")

        profile = st.session_state.get(profile_key)

        if profile:
            render_profile_panel(profile)

            with st.expander("📊 Full Dataset Analytics", expanded=False):
                render_analytics(df_raw, profile)

        # ── STEP 2b: Fuzzy duplicates ─────────────────────
        if st.session_state.get("sb_fuzzy") and profile:
            _sh("🔍 Smart Duplicate Review")
            str_cols = [c for c in df_raw.columns if df_raw[c].dtype == object]
            if not str_cols:
                _info("No text columns available for fuzzy matching.")
            elif len(df_raw) > MAX_ROWS_FUZZY:
                _warning(
                    f"⚠️ Dataset has {len(df_raw):,} rows. "
                    f"Fuzzy matching is limited to {MAX_ROWS_FUZZY:,} rows."
                )
            else:
                fuzzy_cols = st.multiselect(
                    "Columns to compare:", str_cols,
                    default=str_cols[:min(2, len(str_cols))],
                    key="fuzzy_col_select",
                )
                threshold = st.session_state.get("sb_fuzzy_threshold", 0.85)
                if fuzzy_cols and st.button("🔍 Run Fuzzy Check", key="btn_fuzzy"):
                    with st.spinner("Checking for similar records…"):
                        try:
                            fuzzy_df = find_fuzzy_duplicates(df_raw, fuzzy_cols, threshold=threshold)
                        except Exception as exc:
                            fuzzy_df = None
                            _warning(f"Fuzzy check error: {exc}")
                    if fuzzy_df is None:
                        _warning(
                            "⚠️ rapidfuzz is not installed. "
                            "Run: `pip install rapidfuzz`"
                        )
                    elif fuzzy_df.empty:
                        _success("✅ No fuzzy duplicates found above the threshold.")
                    else:
                        _warning(
                            f"⚠️ {len(fuzzy_df)} potential duplicate pair(s) found. "
                            "Review below — nothing is deleted automatically."
                        )
                        st.dataframe(fuzzy_df, use_container_width=True)

        st.markdown("---")

        # ── STEP 3: Configure & Clean ─────────────────────
        _sh("✨ 3. Configure & Clean")
        _info(
            "Review the analysis above, adjust settings in the sidebar, "
            "then click <strong>Run Auto-Clean</strong>."
        )

        col_run, col_rst = st.columns([4, 1])
        with col_run:
            run_btn = st.button(
                "🚀 Run Auto-Clean",
                type="primary",
                use_container_width=True,
                key=f"run_clean_{chosen_name}",
            )
        with col_rst:
            if st.button("🔄 Reset", use_container_width=True, key=f"rst_{chosen_name}"):
                for k in ["df_clean", "clean_report", "df_raw_snap", "active_file"]:
                    st.session_state.pop(k, None)
                st.rerun()

        if run_btn:
            clean_opts = _collect_sidebar_options(profile)
            prog_ph    = st.empty()
            prog_bar   = prog_ph.progress(0, text="Starting…")

            def _cb(frac: float, msg: str) -> None:
                prog_bar.progress(min(frac, 1.0), text=msg)

            try:
                df_clean, report = run_cleaning_pipeline(df_raw, clean_opts, progress_cb=_cb)
                report["filename"]        = chosen_name
                st.session_state["df_clean"]     = df_clean
                st.session_state["df_raw_snap"]  = df_raw.copy()
                st.session_state["clean_report"] = report
                st.session_state["active_file"]  = chosen_name
                prog_ph.empty()
                _success("✅ <strong>Cleaning complete!</strong> See results below.")
            except Exception as exc:
                prog_ph.empty()
                _error(f"❌ Cleaning failed: {exc}")

        # ── STEP 4+5+6: Results ───────────────────────────
        if (
            "df_clean" in st.session_state
            and st.session_state.get("active_file") == chosen_name
        ):
            df_clean = st.session_state["df_clean"]
            report   = st.session_state["clean_report"]
            df_snap  = st.session_state.get("df_raw_snap", df_raw)

            # KPI summary
            _sh("📈 4. Cleaning Results")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            for col_ui, val, lbl in [
                (m1, report["duplicates_removed"],   "Dupes Removed"),
                (m2, report["empty_rows_removed"],   "Empty Rows"),
                (m3, report["missing_filled"],       "Nulls Filled"),
                (m4, report["empty_cols_removed"],   "Empty Cols"),
                (m5, report["encoding_repaired"],    "Encoding Fixed"),
                (m6, report["rows_after"],           "Final Rows"),
            ]:
                col_ui.markdown(_card("", f"{val:,}", lbl), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Before / After
            render_comparison(report)

            # Cleaned data preview
            _sh("✅ 5. Cleaned Data Preview")
            st.dataframe(df_clean.head(10), use_container_width=True, height=260)

            with st.expander("📊 Cleaned Dataset Analytics", expanded=False):
                clean_profile = profile_dataframe(df_clean)
                render_analytics(df_clean, clean_profile)

            # ── STEP 7: Export ────────────────────────────
            _sh("⬇️ 6. Export")
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
                        key=f"dl_xl_{base}",
                    )
                except Exception as exc:
                    _error(f"Excel export error: {exc}")

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
                except Exception as exc:
                    _error(f"CSV export error: {exc}")

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
#  TAB: PDF TABLES
# ═══════════════════════════════════════════════════════════

if "📄 PDF Tables" in tab_labels:
    with all_tabs[tab_labels.index("📄 PDF Tables")]:

        # File picker
        if len(pdf_files) == 1:
            chosen_pdf   = pdf_files[0]
            chosen_pname = chosen_pdf.name
        else:
            chosen_pname = st.selectbox(
                "📂 Select PDF:", [f.name for f in pdf_files], key="sel_pdf"
            )
            chosen_pdf = next(f for f in pdf_files if f.name == chosen_pname)

        ok, err_msg = validate_uploaded_file(chosen_pdf)
        if not ok:
            _error(f"❌ {err_msg}")
            st.stop()

        _info(
            f"📄 <strong>{chosen_pname}</strong> "
            f"&nbsp;|&nbsp; 📦 {chosen_pdf.size / 1024:.1f} KB"
        )

        # Page selection
        pdf_page_mode  = st.session_state.get("sb_pdf_pages", "All Pages")
        pdf_pages_text = st.session_state.get("sb_pdf_page_nums", "")

        sel_mode  = "all"
        sel_pages = None
        if pdf_page_mode == "Specific Pages" and pdf_pages_text:
            try:
                sel_mode  = "specific"
                sel_pages = [
                    int(p.strip()) for p in pdf_pages_text.split(",")
                    if p.strip().isdigit()
                ]
            except Exception:
                pass

        if st.button(
            "🔍 Extract Tables", type="primary",
            use_container_width=True, key=f"pdf_btn_{chosen_pname}"
        ):
            pdf_bytes = chosen_pdf.read()
            chosen_pdf.seek(0)

            prog_ph  = st.empty()
            prog_bar = prog_ph.progress(0, text="Reading PDF…")

            def _pdf_cb(frac: float, msg: str) -> None:
                prog_bar.progress(min(frac, 1.0), text=msg)

            try:
                results = extract_pdf_tables(
                    pdf_bytes, chosen_pname,
                    page_selection=sel_mode,
                    specific_pages=sel_pages,
                    progress_cb=_pdf_cb,
                )
                st.session_state[f"pdf_{chosen_pname}"] = results
            except Exception as exc:
                _error(f"❌ PDF extraction failed: {exc}")
                results = []
            finally:
                prog_ph.empty()

        rkey = f"pdf_{chosen_pname}"
        if rkey in st.session_state:
            results = st.session_state[rkey]
            good    = [r for r in results if r["dataframe"] is not None]
            empty   = [r for r in results if r["dataframe"] is None]
            tbl_r   = [r for r in good if "text" not in r["method"]]
            txt_r   = [r for r in good if "text" in r["method"]]

            p1, p2, p3, p4 = st.columns(4)
            for col_ui, icon, val, lbl in [
                (p1, "📊", len(tbl_r),   "Tables"),
                (p2, "📝", len(txt_r),   "Text Pages"),
                (p3, "❌", len(empty),   "Empty Pages"),
                (p4, "📄", len(results), "Scanned"),
            ]:
                col_ui.markdown(_card(icon, str(val), lbl), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if not good:
                _warning(
                    "⚠️ <strong>No extractable content found.</strong><br>"
                    "This PDF may be image/scan-based. Consider running OCR first."
                )
            else:
                if empty:
                    _info(f"ℹ️ {len(empty)} page(s) had no content and were skipped.")

                method_labels = {
                    "extract_tables()":                      "🟢 Tier 1",
                    "extract_table()":                       "🟡 Tier 2",
                    "extract_text() ← plain text fallback":  "🔵 Text",
                }

                for r in good:
                    badge = method_labels.get(r["method"], r["method"])
                    with st.expander(
                        f"Page {r['page']} · Table 
