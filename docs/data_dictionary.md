# Data Dictionary (Auto-Generated Schema Snapshot)

This file is generated from the *actual current* datasets in `data\processed\`.
Regenerate anytime with:
`python -m src.analysis.generate_data_dictionary`

## news_articles.csv
- Path: `C:/apps/sa_motoring_pilot/sa-auto-competition-pulse/data/processed/news_articles.csv`
- Rows: 3100
- Columns: 13

| column | dtype | missing_count | missing_pct | example_value |
|---|---:|---:|---:|---|
| article_id | object | 0 | 0.0 | 626c98ace46091d6 |
| author | object | 1426 | 46.0 | Malcolm Libera |
| fetched_at_utc | object | 0 | 0.0 | 2026-01-19T02:44:25Z |
| published_at | object | 0 | 0.0 | 2026-01-18T08:00:00Z |
| published_raw | object | 15 | 0.48 | 2026-01-18T08:00:00+00:00 |
| simhash64 | uint64 | 0 | 0.0 | 9891045436295211187 |
| snapshot_path | object | 0 | 0.0 | data\raw\news_html\businesstech_motoring\news_motoring_847949_big-changes-com... |
| snapshot_sha256 | object | 0 | 0.0 | 6474cbf123129f4262b65a87665851183a1a1b06116d3d49994f1ff52f87ae3c |
| source_id | object | 0 | 0.0 | businesstech_motoring |
| text | object | 0 | 0.0 | Motoring Malcolm Libera 18 Jan 2026 Motoring Big changes coming for major die... |
| text_sha256 | object | 0 | 0.0 | 626c98ace46091d67a6e79ce7a301c6a4e7573a6a118b99d3b0afc87d6ab40a5 |
| title | object | 0 | 0.0 | Big changes coming for major diesel users in South Africa |
| url | object | 0 | 0.0 | https://businesstech.co.za/news/motoring/847949/big-changes-coming-for-major-... |

## monthly_event_counts.csv
- Path: `C:/apps/sa_motoring_pilot/sa-auto-competition-pulse/data/processed/monthly_event_counts.csv`
- Rows: 51
- Columns: 9

| column | dtype | missing_count | missing_pct | example_value |
|---|---:|---:|---:|---|
| CAPACITY_SUPPLY | int64 | 0 | 0.0 | 1 |
| DEALER_NETWORK | int64 | 0 | 0.0 | 0 |
| ENTRY_EXIT | int64 | 0 | 0.0 | 0 |
| EV_POLICY_INFRA | int64 | 0 | 0.0 | 0 |
| LAUNCH_PRODUCT | int64 | 0 | 0.0 | 3 |
| PRICE_ACTION | int64 | 0 | 0.0 | 0 |
| RECALL_QUALITY | int64 | 0 | 0.0 | 0 |
| period_month | int64 | 0 | 0.0 | 1 |
| period_year | int64 | 0 | 0.0 | 2019 |

## naamsa_monthly_sales_headlines.csv
- Path: `C:/apps/sa_motoring_pilot/sa-auto-competition-pulse/data/processed/naamsa_monthly_sales_headlines.csv`
- Rows: 63
- Columns: 9

| column | dtype | missing_count | missing_pct | example_value |
|---|---:|---:|---:|---|
| evidence_pattern | object | 1 | 1.59 | NEW_VEHICLE_SALES_ANCHORED_SCAN |
| evidence_window | object | 1 | 1.59 | released today for public consumption via the website of the Department of Tr... |
| extraction_confidence | object | 1 | 1.59 | high |
| extraction_method | object | 1 | 1.59 | HEADLINE_REGEX |
| pdf_path | object | 0 | 0.0 | data\raw\naamsa_pdfs\2020_06_Comment-On-The-January-2019-Industry-New-Vehicle... |
| period_month | int64 | 0 | 0.0 | 1 |
| period_year | int64 | 0 | 0.0 | 2019 |
| total_sales_units | float64 | 1 | 1.59 | 42374.0 |
| warning_flag | object | 60 | 95.24 | MULTIPLE_TOTALS_NEARBY |

## panel_monthly_events_sales.csv
- Path: `C:/apps/sa_motoring_pilot/sa-auto-competition-pulse/data/processed/panel_monthly_events_sales.csv`
- Rows: 51
- Columns: 11

| column | dtype | missing_count | missing_pct | example_value |
|---|---:|---:|---:|---|
| CAPACITY_SUPPLY | int64 | 0 | 0.0 | 1 |
| DEALER_NETWORK | int64 | 0 | 0.0 | 0 |
| ENTRY_EXIT | int64 | 0 | 0.0 | 0 |
| EV_POLICY_INFRA | int64 | 0 | 0.0 | 0 |
| LAUNCH_PRODUCT | int64 | 0 | 0.0 | 3 |
| PRICE_ACTION | int64 | 0 | 0.0 | 0 |
| RECALL_QUALITY | int64 | 0 | 0.0 | 0 |
| month_id | object | 0 | 0.0 | 2019-01 |
| period_month | int64 | 0 | 0.0 | 1 |
| period_year | int64 | 0 | 0.0 | 2019 |
| total_sales_units | float64 | 7 | 13.73 | 42374.0 |
