from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd


def die(msg: str) -> "NoReturn":
    print(f"ERROR: {msg}")
    raise SystemExit(1)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic QA gate (offline).")
    ap.add_argument("--min-months-news", type=int, default=6)
    ap.add_argument("--min-months-events", type=int, default=12)
    ap.add_argument("--check-snapshots", type=int, default=1, choices=[0, 1])
    ap.add_argument("--snapshot-sample", type=int, default=10)
    args = ap.parse_args()

    # ---------------- news_articles.csv ----------------
    news_path = Path(r"data\processed\news_articles.csv")
    if not news_path.exists():
        die(f"Missing {news_path}")

    news = pd.read_csv(news_path)

    req = ["article_id", "source_id", "url", "title", "text"]
    for c in req:
        if c not in news.columns:
            die(f"news_articles.csv missing column: {c}")

    time_cols = [c for c in ["published_at", "published_raw", "fetched_at_utc"] if c in news.columns]
    if not time_cols:
        die("news_articles.csv missing time columns (expected published_at/published_raw/fetched_at_utc)")

    if news["article_id"].duplicated().any():
        die("news_articles.csv has duplicate article_id")

    if news["url"].isna().any():
        die("news_articles.csv has NA url")

    dtcol = "published_at" if "published_at" in news.columns else time_cols[0]
    dt = pd.to_datetime(news[dtcol], errors="coerce", utc=True)
    if not dt.notna().any():
        die(f"news_articles.csv date column parses to all-NA: {dtcol}")
    # Avoid pandas warning: drop timezone explicitly before Period conversion
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert(None)
    m = dt.dt.to_period("M")
    months = int(m.nunique())
    print(f"news_articles: rows={len(news)} months={months} min={m.min()} max={m.max()}")

    if months < args.min_months_news:
        die(f"news_articles has < {args.min_months_news} months of coverage")

    # ---------------- monthly_event_counts.csv ----------------
    evm_path = Path(r"data\processed\monthly_event_counts.csv")
    if not evm_path.exists():
        die(f"Missing {evm_path}")

    evm = pd.read_csv(evm_path)

    for c in ["period_year", "period_month"]:
        if c not in evm.columns:
            die(f"monthly_event_counts.csv missing column: {c}")

    if evm.duplicated(["period_year", "period_month"]).any():
        die("monthly_event_counts.csv has duplicate (period_year, period_month)")

    event_cols = [c for c in evm.columns if c not in ["period_year", "period_month"]]
    if not event_cols:
        die("monthly_event_counts.csv has no event columns")

    for c in event_cols:
        s = pd.to_numeric(evm[c], errors="coerce")
        if s.isna().any():
            die(f"monthly_event_counts.csv has non-numeric values in {c}")
        if (s < 0).any():
            die(f"monthly_event_counts.csv has negative values in {c}")

    months_ev = int(len(evm))
    print(f"monthly_event_counts: rows={months_ev} event_cols={len(event_cols)}")
    if months_ev < args.min_months_events:
        die(f"monthly_event_counts has < {args.min_months_events} months")

    # ---------------- naamsa_monthly_sales_headlines.csv ----------------
    na_path = Path(r"data\processed\naamsa_monthly_sales_headlines.csv")
    if not na_path.exists():
        die(f"Missing {na_path}")

    na = pd.read_csv(na_path)

    for c in ["period_year", "period_month", "total_sales_units"]:
        if c not in na.columns:
            die(f"naamsa_monthly_sales_headlines.csv missing column: {c}")

    if na.duplicated(["period_year", "period_month"]).any():
        die("naamsa_monthly_sales_headlines.csv has duplicate (period_year, period_month)")

    tot = pd.to_numeric(na["total_sales_units"], errors="coerce")
    miss = int(tot.isna().sum())
    print(f"naamsa_monthly_sales_headlines: rows={len(na)} missing_totals={miss}")

    ok = tot.dropna()
    if len(ok) > 0:
        if (ok <= 0).any():
            die("naamsa total_sales_units has non-positive values")
        bad = int(((ok < 10000) | (ok > 200000)).sum())
        print(f"naamsa totals range_check(10000..200000): bad={bad}")

    # ---------------- panel_monthly_events_sales.csv ----------------
    pan_path = Path(r"data\processed\panel_monthly_events_sales.csv")
    if not pan_path.exists():
        die(f"Missing {pan_path}")

    pan = pd.read_csv(pan_path)

    for c in ["month_id", "total_sales_units"]:
        if c not in pan.columns:
            die(f"panel_monthly_events_sales.csv missing column: {c}")

    if pan["month_id"].duplicated().any():
        die("panel has duplicate month_id (should be 1 row per month)")

    if ("period_year" in pan.columns) and ("period_month" in pan.columns):
        exp = pan["period_year"].astype(int).astype(str) + "-" + pan["period_month"].astype(int).map(lambda x: f"{x:02d}")
        got = pan["month_id"].astype(str).str.slice(0, 7)
        if (exp != got).any():
            die("panel month_id does not match period_year/period_month for some rows")

    skip = {"month_id", "total_sales_units", "period_year", "period_month"}
    ev_cols = [c for c in pan.columns if c not in skip]

    for c in ev_cols:
        s = pd.to_numeric(pan[c], errors="coerce")
        if s.isna().any():
            die(f"panel has non-numeric values in {c}")
        if (s < 0).any():
            die(f"panel has negative values in {c}")

    msk = pd.to_numeric(pan["total_sales_units"], errors="coerce").isna()
    miss_months = sorted(set(pan.loc[msk, "month_id"].astype(str).str.slice(0, 7).tolist()))
    print(f"panel: rows={len(pan)} event_cols={len(ev_cols)}")
    print("panel missing sales months:", miss_months)

    # ---------------- missingness vs docs ----------------
    doc_path = Path(r"docs\definition_of_done.md")
    if doc_path.exists():
        doc = doc_path.read_text(encoding="utf-8", errors="ignore")
        doc_months = re.findall(r"(?m)^- (\d{4}-\d{2})\s*$", doc)
        if not doc_months:
            die("docs/definition_of_done.md exists but has no lines like '- YYYY-MM'")
        doc_months = sorted(set(doc_months))
        computed = sorted(set(miss_months))
        if computed != doc_months:
            print("ERROR: Missing months mismatch!")
            print(" documented=", doc_months)
            print(" computed  =", computed)
            raise SystemExit(1)
        print("OK: missing months match docs:", computed)
    else:
        print("SKIP: docs/definition_of_done.md not found (cannot enforce documented missingness).")

    # ---------------- snapshot hash spot-check ----------------
    if args.check_snapshots == 1:
        for c in ["snapshot_path", "snapshot_sha256"]:
            if c not in news.columns:
                die(f"missing {c} in news_articles.csv for snapshot check")

        sample = news.sort_values("article_id").head(int(args.snapshot_sample)).copy()
        for _, r in sample.iterrows():
            p = Path(str(r["snapshot_path"]))
            if not p.exists():
                die(f"snapshot file missing: {p}")
            h = sha256_bytes(p.read_bytes())
            exp = str(r["snapshot_sha256"])
            if h != exp:
                die(f"snapshot sha mismatch for {p}")
        print(f"OK: snapshot sample hash check passed ({len(sample)} files)")
    else:
        print("CHECK_SNAPSHOTS=0 so snapshot checks were skipped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
