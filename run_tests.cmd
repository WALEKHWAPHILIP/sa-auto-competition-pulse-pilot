@echo off
setlocal EnableExtensions

REM ============================================================
REM SA Auto Competition Pulse - Deterministic Test Runner
REM - NO scraping / NO downloading
REM - Runs: compile smoke test + pytest unit tests + QA gate
REM Run from repo root
REM ============================================================

REM ---------- QA Gate parameters (explicit = deterministic) ----------
set MIN_MONTHS_NEWS=6
set MIN_MONTHS_EVENTS=12
set CHECK_SNAPSHOTS=1
set SNAPSHOT_SAMPLE=10

echo.
echo [0/5] Repo root sanity
if not exist src\ (
  echo ERROR: Missing src\ . Run from repo root.
  exit /b 1
)
if not exist configs\pipeline.yaml (
  echo ERROR: Missing configs\pipeline.yaml .
  exit /b 1
)

echo.
echo [1/5] Environment sanity
python -c "import sys; print('python=',sys.executable); print('version=',sys.version.split()[0])" || exit /b 1

echo.
echo [2/5] Compile smoke test (syntax)
python -u -m compileall src || exit /b 1

echo.
echo [3/5] Unit tests (pytest)
python -c "import pytest; print('pytest=',pytest.__version__)" || (
  echo ERROR: pytest is not installed in this venv.
  echo FIX  : pip install pytest==9.0.2
  exit /b 1
)
python -u -m pytest || exit /b 1

echo.
echo [4/5] QA gate (Python, deterministic, offline)
python -u -m src.analysis.qa_gate ^
  --min-months-news %MIN_MONTHS_NEWS% ^
  --min-months-events %MIN_MONTHS_EVENTS% ^
  --check-snapshots %CHECK_SNAPSHOTS% ^
  --snapshot-sample %SNAPSHOT_SAMPLE%
if errorlevel 1 exit /b 1

echo.
echo PASS: All tests + QA gate completed successfully.
endlocal
