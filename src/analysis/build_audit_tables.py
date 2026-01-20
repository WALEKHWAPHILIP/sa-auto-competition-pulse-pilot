from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd


def _project_root() -> Path:
    # src/analysis/build_audit_tables.py -> src -> repo root
    return Path(__file__).resolve().parents[2]


def _pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_month_id(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=False)
    return dt.dt.strftime("%Y-%m")


def _numeric_event_columns(df: pd.DataFrame, exclude_cols: set[str]) -> list[str]:
    cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _build_month_id_from_year_month(df: pd.DataFrame, year_col: str, month_col: str) -> pd.Series:
    return (
        df[year_col].astype(int).astype(str)
        + "-"
        + df[month_col].astype(int).astype(str).str.zfill(2)
    )


def main() -> int:
    root = _project_root()

    news_path = root / "data" / "processed" / "news_articles.csv"
    events_monthly_path = root / "data" / "processed" / "monthly_event_counts.csv"
    naamsa_path = root / "data" / "processed" / "naamsa_monthly_sales_headlines.csv"
    panel_path = root / "data" / "processed" / "panel_monthly_events_sales.csv"

    out_cov = root / "reports" / "tables" / "coverage_audit.csv"
    out_miss = root / "reports" / "tables" / "missingness_audit.csv"
    out_cov.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # NEWS: counts by month
    # -------------------------
    if not news_path.exists():
        raise FileNotFoundError(f"Missing input: {news_path}")

    news = pd.read_csv(news_path)
    news_date_col = _pick_first_existing_col(
        news,
        ["published_at", "published_date", "date", "published", "published_time", "time_published"],
    )
    if news_date_col is None:
        raise ValueError(
            "news_articles.csv must contain a publication date column. "
            "Expected one of: published_at, published_date, date, published, published_time, time_published"
        )

    news["month_id"] = _ensure_month_id(news[news_date_col])
    news_counts = (
        news.dropna(subset=["month_id"])
        .groupby("month_id", as_index=False)
        .size()
        .rename(columns={"size": "news_articles"})
    )

    # -------------------------
    # EVENTS: intensity by month
    # -------------------------
    if not events_monthly_path.exists():
        raise FileNotFoundError(f"Missing input: {events_monthly_path}")

    evm = pd.read_csv(events_monthly_path)

    # Your pipeline schema: period_year + period_month
    evm_year_col = _pick_first_existing_col(evm, ["period_year", "year", "Year"])
    evm_month_col = _pick_first_existing_col(evm, ["period_month", "month", "Month"])
    evm_month_id_col = _pick_first_existing_col(evm, ["month_id", "published_month"])

    if evm_month_id_col is not None:
        evm["month_id"] = evm[evm_month_id_col].astype(str).str.slice(0, 7)
    elif evm_year_col is not None and evm_month_col is not None:
        evm["month_id"] = _build_month_id_from_year_month(evm, evm_year_col, evm_month_col)
    else:
        raise ValueError(
            "monthly_event_counts.csv must contain either month_id/published_month "
            "or (period_year/year and period_month/month)."
        )

    evm_exclude = {c for c in [evm_month_id_col, evm_year_col, evm_month_col] if c is not None}
    evm_exclude.add("month_id")

    event_cols = _numeric_event_columns(evm, exclude_cols=evm_exclude)
    if not event_cols:
        raise ValueError("monthly_event_counts.csv has no numeric event columns to sum.")

    evm["event_intensity_total"] = evm[event_cols].fillna(0).sum(axis=1)
    evm_out = evm[["month_id", "event_intensity_total"]].copy()

    # -------------------------
    # NAAMSA: coverage by month + missing totals flag
    # -------------------------
    if not naamsa_path.exists():
        raise FileNotFoundError(f"Missing input: {naamsa_path}")

    na = pd.read_csv(naamsa_path)

    # Your NAAMSA schema: period_year + period_month
    na_year_col = _pick_first_existing_col(na, ["period_year", "year", "Year"])
    na_month_col = _pick_first_existing_col(na, ["period_month", "month", "Month"])
    na_month_id_col = _pick_first_existing_col(na, ["month_id", "published_month"])

    if na_month_id_col is not None:
        na["month_id"] = na[na_month_id_col].astype(str).str.slice(0, 7)
    elif na_year_col is not None and na_month_col is not None:
        na["month_id"] = _build_month_id_from_year_month(na, na_year_col, na_month_col)
    else:
        raise ValueError(
            "naamsa_monthly_sales_headlines.csv must contain either month_id/published_month "
            "or (period_year/year and period_month/month)."
        )

    total_col = _pick_first_existing_col(
        na,
        [
            "total_sales_units",
            "total_sales",
            "industry_total_units",
            "industry_total",
            "sales_units",
            "sales_total_units",
        ],
    )
    if total_col is None:
        # Coverage still possible, but totals missingness cannot be computed
        na["naamsa_total_missing"] = pd.NA
    else:
        na["naamsa_total_missing"] = pd.to_numeric(na[total_col], errors="coerce").isna()

    na_cov = (
        na.groupby("month_id", as_index=False)
        .agg(
            naamsa_rows=("month_id", "size"),
            naamsa_total_missing=("naamsa_total_missing", "max"),
        )
    )
    na_cov["naamsa_available"] = 1

    # -------------------------
    # PANEL: coverage + missing sales months
    # -------------------------
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing input: {panel_path}")

    panel = pd.read_csv(panel_path)
    panel_month_col = _pick_first_existing_col(panel, ["month_id", "month", "published_month"])
    if panel_month_col is None:
        raise ValueError("panel_monthly_events_sales.csv must contain month_id or month column.")
    panel["month_id"] = panel[panel_month_col].astype(str).str.slice(0, 7)

    panel_sales_col = _pick_first_existing_col(
        panel,
        ["total_sales_units", "total_sales", "industry_total_units", "sales_units", "sales_total_units"],
    )
    if panel_sales_col is None:
        raise ValueError("panel_monthly_events_sales.csv must contain a sales total column such as total_sales_units.")

    panel_sales = pd.to_numeric(panel[panel_sales_col], errors="coerce")
    panel["panel_sales_missing"] = panel_sales.isna()

    # Build event intensity in the panel (sum numeric event columns excluding month + sales)
    exclude_panel = {panel_month_col, "month_id", panel_sales_col, "panel_sales_missing"}

    # Do NOT treat date fields as event counts
    for c in ["period_year", "period_month", "year", "month_num"]:
        if c in panel.columns:
            exclude_panel.add(c)

    panel_event_cols = _numeric_event_columns(panel, exclude_cols=exclude_panel)
    if panel_event_cols:
        panel["panel_event_intensity_total"] = panel[panel_event_cols].fillna(0).sum(axis=1)
    else:
        panel["panel_event_intensity_total"] = 0.0

    panel_cov = (
        panel.groupby("month_id", as_index=False)
        .agg(
            panel_rows=("month_id", "size"),
            panel_sales_missing=("panel_sales_missing", "max"),
            panel_event_intensity_total=("panel_event_intensity_total", "sum"),
        )
    )
    panel_cov["panel_available"] = 1

    missing_months = panel_cov.loc[panel_cov["panel_sales_missing"] == True, ["month_id"]].copy()
    missing_months["missing_field"] = "total_sales_units"
    missing_months["dataset"] = "panel_monthly_events_sales.csv"
    missing_months["reason"] = "NAAMSA total missing in merged panel for this month"

    # -------------------------
    # Coverage audit: merge everything by month_id
    # -------------------------
    cov = news_counts.merge(evm_out, on="month_id", how="outer")
    cov = cov.merge(
        na_cov[["month_id", "naamsa_available", "naamsa_total_missing"]],
        on="month_id",
        how="outer",
    )
    cov = cov.merge(
        panel_cov[["month_id", "panel_available", "panel_sales_missing", "panel_event_intensity_total"]],
        on="month_id",
        how="outer",
    )

    # Fill sensible defaults
    if "news_articles" in cov.columns:
        cov["news_articles"] = cov["news_articles"].fillna(0).astype(int)
    if "event_intensity_total" in cov.columns:
        cov["event_intensity_total"] = cov["event_intensity_total"].fillna(0)
    cov["naamsa_available"] = cov["naamsa_available"].fillna(0).astype(int)
    cov["panel_available"] = cov["panel_available"].fillna(0).astype(int)

    cov = cov.sort_values("month_id").reset_index(drop=True)
    cov.to_csv(out_cov, index=False)

    missing_months = missing_months.sort_values("month_id").reset_index(drop=True)
    missing_months.to_csv(out_miss, index=False)

    print(f"Saved: {out_cov} | rows={len(cov)}")
    print(f"Saved: {out_miss} | rows={len(missing_months)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
