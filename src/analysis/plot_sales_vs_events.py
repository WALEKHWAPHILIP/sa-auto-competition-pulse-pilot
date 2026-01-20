from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _numeric_event_columns(df: pd.DataFrame, exclude_cols: set[str]) -> list[str]:
    cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def main() -> int:
    root = _project_root()
    panel_path = root / "data" / "processed" / "panel_monthly_events_sales.csv"
    out_path = root / "reports" / "figures" / "sales_vs_event_intensity.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not panel_path.exists():
        raise FileNotFoundError(f"Missing input: {panel_path}")

    df = pd.read_csv(panel_path)

    month_col = _pick_first_existing_col(df, ["month_id", "month", "published_month"])
    if month_col is None:
        raise ValueError("panel_monthly_events_sales.csv must contain month_id or month column.")
    df["month_id"] = df[month_col].astype(str).str.slice(0, 7)

    # Convert month_id to a real datetime for clean axis formatting
    df["month_dt"] = pd.to_datetime(df["month_id"] + "-01", errors="coerce")

    sales_col = _pick_first_existing_col(
        df,
        ["total_sales_units", "total_sales", "industry_total_units", "sales_units", "sales_total_units"],
    )
    if sales_col is None:
        raise ValueError("panel_monthly_events_sales.csv must contain a sales total column such as total_sales_units.")
    df["total_sales_units"] = pd.to_numeric(df[sales_col], errors="coerce")

    exclude = {month_col, "month_id", "month_dt", sales_col, "total_sales_units"}

    # Do NOT treat date fields as event counts
    for c in ["period_year", "period_month", "year", "month_num"]:
        if c in df.columns:
            exclude.add(c)

    event_cols = _numeric_event_columns(df, exclude_cols=exclude)
    if event_cols:
        df["event_intensity_total"] = df[event_cols].fillna(0).sum(axis=1)
    else:
        df["event_intensity_total"] = 0.0

    df = df.sort_values("month_dt").reset_index(drop=True)
    df_plot = df.dropna(subset=["month_dt"]).copy()

    # Keep only months where we have at least one of (sales, events) present
    df_plot = df_plot[(~df_plot["total_sales_units"].isna()) | (df_plot["event_intensity_total"] != 0)]

    fig, ax1 = plt.subplots()

    ax1.plot(df_plot["month_dt"], df_plot["total_sales_units"])
    ax1.set_xlabel("month")
    ax1.set_ylabel("total_sales_units")
    ax1.grid(True, axis="y")

    ax2 = ax1.twinx()
    ax2.plot(df_plot["month_dt"], df_plot["event_intensity_total"])
    ax2.set_ylabel("event_intensity_total")

    # Clean x-axis: show fewer ticks, formatted YYYY-MM
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.DateFormatter("%Y-%m")
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {out_path} | rows={len(df_plot)} | event_cols={len(event_cols)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
