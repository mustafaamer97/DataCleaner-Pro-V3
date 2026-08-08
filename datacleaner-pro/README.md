# 🧹 DataCleaner Pro V3 — Commercial Edition

**Clean. Analyze. Export.**

A professional-grade data cleaning web application built with Streamlit.

## Features

| Feature | Description |
|---|---|
| Multi-file upload | CSV, Excel (.xlsx/.xls), PDF |
| Smart Profiling | Auto-detect emails, phones, dates, outliers |
| One-Click Clean | Dedup, encode repair, fill nulls, normalize |
| PDF Extraction | 3-tier strategy with text fallback |
| Batch Processing | Clean many files → download as ZIP |
| Demo Mode | Built-in sample dataset — no upload needed |
| Professional Reports | Detailed before/after cleaning reports |

## Quick Start

```bash
# 1. Clone or download
git clone https://github.com/yourrepo/datacleaner-pro

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

## Project Structure

```
datacleaner-pro/
├── app.py                 ← Streamlit UI
├── utils/
│   ├── cleaning.py        ← Core cleaning engine
│   ├── profiling.py       ← Smart data profiling
│   ├── duplicates.py      ← Duplicate detection
│   ├── outliers.py        ← Outlier detection
│   ├── pdf_processor.py   ← PDF extraction
│   ├── exporters.py       ← Export engine
│   ├── reports.py         ← Report generation
│   └── helpers.py         ← Shared utilities
├── sample_data/
│   └── sample_customers.csv
├── tests/
│   └── test_cleaning.py
└── requirements.txt
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Deployment (Streamlit Cloud)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo → `app.py` → Deploy
