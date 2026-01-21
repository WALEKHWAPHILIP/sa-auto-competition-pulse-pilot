@echo off
setlocal EnableExtensions

REM ============================================================
REM SA Auto Competition Pulse - One-command rebuild (Windows CMD)
REM Run from repo root (folder that contains src\, configs\, data\)
REM ============================================================

echo.
echo [1/5] Compile
echo CMD: python -m compileall src
python -u -m compileall src || exit /b 1

REM ------------------------------------------------------------
REM [2/5] News scrape (DISABLED)
REM ------------------------------------------------------------
REM echo.
REM echo [2/5] News scrape
REM echo CMD: python -m src.news.scrape_news --pipeline configs\pipeline.yaml --sources configs\sources.yaml
REM python -u -m src.news.scrape_news --pipeline configs\pipeline.yaml --sources configs\sources.yaml || exit /b 1

echo.
echo [3/5] Event labeling + monthly aggregation
echo CMD: python -m src.events.label_events --pipeline configs\pipeline.yaml --schema configs\labeling_schema.yaml
python -u -m src.events.label_events --pipeline configs\pipeline.yaml --schema configs\labeling_schema.yaml || exit /b 1

echo CMD: python -m src.news.count_labels --pipeline configs\pipeline.yaml
python -u -m src.news.count_labels --pipeline configs\pipeline.yaml || exit /b 1

echo.
echo [4/5] NAAMSA pipeline

REM 4.1 Discover links
echo ---- 4.1 START discover_links ----
echo CMD: python -m src.naamsa.discover_links --pipeline configs\pipeline.yaml --sources configs\sources.yaml --out data\processed\naamsa_links_to_fetch.csv
python -u -m src.naamsa.discover_links --pipeline configs\pipeline.yaml --sources configs\sources.yaml --out data\processed\naamsa_links_to_fetch.csv || exit /b 1
if not exist data\processed\naamsa_links_to_fetch.csv (
  echo ERROR: discover_links did not create data\processed\naamsa_links_to_fetch.csv
  exit /b 1
)
echo ---- 4.1 END discover_links ----
dir data\processed\naamsa_links_to_fetch.csv

REM 4.2 Download PDFs
echo ---- 4.2 START download_pdfs ----
echo CMD: python -m src.naamsa.download_pdfs --pipeline configs\pipeline.yaml --in_links data\processed\naamsa_links_to_fetch.csv
python -u -m src.naamsa.download_pdfs --pipeline configs\pipeline.yaml --in_links data\processed\naamsa_links_to_fetch.csv || exit /b 1
echo ---- 4.2 END download_pdfs ----
if not exist data\raw\naamsa_pdfs (
  echo ERROR: missing folder data\raw\naamsa_pdfs
  exit /b 1
)
dir data\raw\naamsa_pdfs\*.pdf 2>nul | find /c ".pdf"
if not exist data\raw\naamsa_pdfs\manifest.jsonl (
  echo ERROR: missing data\raw\naamsa_pdfs\manifest.jsonl
  exit /b 1
)
dir data\raw\naamsa_pdfs\manifest.jsonl

REM 4.3 Parse monthly totals
echo ---- 4.3 START parse_sales_headlines ----
echo CMD: python -m src.naamsa.parse_sales_headlines --pipeline configs\pipeline.yaml
python -u -m src.naamsa.parse_sales_headlines --pipeline configs\pipeline.yaml || exit /b 1
echo ---- 4.3 END parse_sales_headlines ----
if not exist data\processed\naamsa_monthly_sales_headlines.csv (
  echo ERROR: missing data\processed\naamsa_monthly_sales_headlines.csv
  exit /b 1
)
dir data\processed\naamsa_monthly_sales_headlines.csv

echo.
echo [5/5] Analysis outputs

echo CMD: python -m src.analysis.build_audit_tables
python -u -m src.analysis.build_audit_tables || exit /b 1

echo CMD: python -m src.analysis.generate_data_dictionary
python -u -m src.analysis.generate_data_dictionary || exit /b 1

echo CMD: python -m src.analysis.plot_sales_vs_events
python -u -m src.analysis.plot_sales_vs_events || exit /b 1

echo CMD: python -m src.analysis.run_lag_model_baseline
python -u -m src.analysis.run_lag_model_baseline || exit /b 1

echo.
echo DONE: pipeline + analysis outputs rebuilt successfully.
endlocal
