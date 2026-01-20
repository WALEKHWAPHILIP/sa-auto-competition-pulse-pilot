# Definition of Done (Batch 7)

## Rebuild commands (make-like, deterministic)
From repo root:

1) Audit tables
- `python -m src.analysis.build_audit_tables`

2) Data dictionary (schema snapshot from actual files)
- `python -m src.analysis.generate_data_dictionary`

3) Figure
- `python -m src.analysis.plot_sales_vs_events`

4) Baseline lag model table
- `python -m src.analysis.run_lag_model_baseline`

## Proof artifacts (what “done” produces)
- `reports\tables\coverage_audit.csv`
- `reports\tables\missingness_audit.csv`
- `docs\data_dictionary.md`
- `reports\figures\sales_vs_event_intensity.png`
- `reports\tables\lag_model_baseline.csv`

## Pipeline integrity checks (must be visible in outputs)
- Coverage audit shows month coverage across:
  - news volume,
  - event intensity,
  - NAAMSA availability,
  - merged panel availability.
- Missingness audit lists months with missing sales in the merged panel.

## Known limitations (accepted)
Monthly NAAMSA totals remain missing for the following months (documented limitation):
- 2019-02
- 2019-03
- 2019-04
- 2023-02
- 2023-04
- 2024-04
- 2026-01

Rationale: we stop expanding PDF discovery at this stage; the repo focuses on research-grade governance + analysis outputs built on the current stable dataset set.
