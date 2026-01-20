# Reproducibility Checklist

This checklist is a “PI/interviewer runnable” standard: a reviewer should be able to regenerate the core artifacts deterministically from the repository.

## Environment

- [ ] Repository cloned cleanly
- [ ] Python environment created and activated
- [ ] Dependencies installed
- [ ] `python -m compileall src` succeeds

## Inputs (raw/processed)

- [ ] Required processed inputs exist:
  - `data/processed/news_articles.csv`
  - `data/processed/monthly_event_counts.csv`
  - `data/processed/naamsa_monthly_sales_headlines.csv`
  - `data/processed/panel_monthly_events_sales.csv`

## Deterministic research outputs (this batch)

- [ ] Coverage audit tables generated:
  - `reports/tables/coverage_audit.csv`
  - `reports/tables/missingness_audit.csv`
- [ ] Data dictionary generated from actual dataset schemas:
  - `docs/data_dictionary.md`
- [ ] Figure generated:
  - `reports/figures/sales_vs_event_intensity.png`
- [ ] Baseline lag model table generated:
  - `reports/tables/lag_model_baseline.csv`

## Integrity checks

- [ ] All scripts print `Saved: ... | rows= ...` (or file saved confirmation)
- [ ] Missing sales months are explicitly listed in `docs/definition_of_done.md`
- [ ] Outputs are produced by scripts in `src/analysis/` (not notebooks)

## Change control

- [ ] Any change to outputs is accompanied by:
  - updated configs (if applicable)
  - re-run of scripts
  - committed updated tables/figures and docs
