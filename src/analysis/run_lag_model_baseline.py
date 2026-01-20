from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


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


def _ols(X: np.ndarray, y: np.ndarray) -> dict:
    # OLS via least squares
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat

    n = X.shape[0]
    k = X.shape[1]
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (rss / tss) if tss > 0 else float("nan")

    dof = max(n - k, 1)
    sigma2 = rss / dof

    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    var_beta = sigma2 * XtX_inv
    se = np.sqrt(np.diag(var_beta))
    tvals = beta / se

    return {
        "beta": beta,
        "se": se,
        "t": tvals,
        "n": n,
        "k": k,
        "r2": r2,
    }


def main() -> int:
    root = _project_root()
    panel_path = root / "data" / "processed" / "panel_monthly_events_sales.csv"
    out_path = root / "reports" / "tables" / "lag_model_baseline.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not panel_path.exists():
        raise FileNotFoundError(f"Missing input: {panel_path}")

    df = pd.read_csv(panel_path)

    month_col = _pick_first_existing_col(df, ["month_id", "month", "published_month"])
    if month_col is None:
        raise ValueError("panel_monthly_events_sales.csv must contain month_id or month column.")
    df["month_id"] = df[month_col].astype(str).str.slice(0, 7)

    # Outcome
    sales_col = _pick_first_existing_col(
        df,
        ["total_sales_units", "total_sales", "industry_total_units", "sales_units", "sales_total_units"],
    )
    if sales_col is None:
        raise ValueError("panel_monthly_events_sales.csv must contain a sales total column such as total_sales_units.")
    df["total_sales_units"] = pd.to_numeric(df[sales_col], errors="coerce")

    # Event intensity: sum numeric event columns excluding IDs/sales and date fields
    exclude = {month_col, "month_id", sales_col, "total_sales_units"}
    for c in ["period_year", "period_month", "year", "month_num"]:
        if c in df.columns:
            exclude.add(c)

    event_cols = _numeric_event_columns(df, exclude_cols=exclude)
    if event_cols:
        df["event_intensity_total"] = df[event_cols].fillna(0).sum(axis=1)
    else:
        df["event_intensity_total"] = 0.0

    df = df.sort_values("month_id").reset_index(drop=True)
    df["event_intensity_lag1"] = df["event_intensity_total"].shift(1)

    # Fixed effects controls derived from month_id
    df["year"] = pd.to_numeric(df["month_id"].str.slice(0, 4), errors="coerce")
    df["month_num"] = pd.to_numeric(df["month_id"].str.slice(5, 7), errors="coerce")

    # Keep complete rows
    model_df = df.dropna(
        subset=["total_sales_units", "event_intensity_total", "event_intensity_lag1", "year", "month_num"]
    ).copy()

    if len(model_df) < 12:
        raise ValueError(
            f"Not enough complete rows to run regression (need at least 12). Have {len(model_df)}."
        )

    # Design: y ~ 1 + events_t + events_t-1 + month-of-year FE + year FE
    y = model_df["total_sales_units"].to_numpy(dtype=float)

    X_parts = []
    var_names = []

    # Intercept
    X_parts.append(np.ones((len(model_df), 1)))
    var_names.append("Intercept")

    # Events current + lag1
    X_parts.append(model_df["event_intensity_total"].to_numpy(dtype=float).reshape(-1, 1))
    var_names.append("events_t")

    X_parts.append(model_df["event_intensity_lag1"].to_numpy(dtype=float).reshape(-1, 1))
    var_names.append("events_t_minus_1")

    # Month-of-year FE (drop first)
    month_dummies = pd.get_dummies(model_df["month_num"].astype(int), prefix="month", drop_first=True)
    X_parts.append(month_dummies.to_numpy(dtype=float))
    var_names.extend(list(month_dummies.columns))

    # Year FE (drop first)
    year_dummies = pd.get_dummies(model_df["year"].astype(int), prefix="year", drop_first=True)
    X_parts.append(year_dummies.to_numpy(dtype=float))
    var_names.extend(list(year_dummies.columns))

    X = np.hstack(X_parts)

    res = _ols(X, y)

    out = pd.DataFrame(
        {
            "variable": var_names,
            "coef": res["beta"],
            "std_err": res["se"],
            "t_value": res["t"],
        }
    )
    out["n_obs"] = res["n"]
    out["r2"] = res["r2"]

    out.to_csv(out_path, index=False)

    print(
        f"Saved: {out_path} | rows={len(out)} | n_obs={res['n']} | r2={res['r2']:.4f} | event_cols={len(event_cols)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
