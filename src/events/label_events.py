from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.utils.config import load_yaml, get_pipeline_paths


DATE_PATTERNS = [
    re.compile(r"/(?P<y>19\d{2}|20\d{2})/(?P<m>0[1-9]|1[0-2])/(?P<d>0[1-9]|[12]\d|3[01])/", re.I),
    re.compile(r"(?P<y>19\d{2}|20\d{2})[-_](?P<m>0[1-9]|1[0-2])[-_](?P<d>0[1-9]|[12]\d|3[01])", re.I),
]


def coerce_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def date_from_url(url: str) -> Optional[pd.Timestamp]:
    u = (url or "").strip()
    for rx in DATE_PATTERNS:
        m = rx.search(u)
        if m:
            y, mm, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
            try:
                return pd.Timestamp(year=y, month=mm, day=d, tz="UTC")
            except Exception:
                return None
    return None


def is_probable_listing_page(url: str, title: str, text: str, published_raw: str, published_at: str = "") -> bool:
    u = (url or "").rstrip("/").lower()
    t = (title or "").strip().lower()

    has_pub = bool(str(published_at or "").strip()) or bool(str(published_raw or "").strip())
    if has_pub:
        return False

    if re.fullmatch(r"https?://[^/]+/news", u):
        return True
    if re.search(r"/news/(category|latest|industry|motoring|cars|vehicle)s?$", u):
        return True
    if re.search(r"/category/[^/]+$", u):
        return True

    if any(k in t for k in ["motoring", "latest", "category", "news"]):
        if len((text or "")) > 2000:
            return True

    return False


@dataclass
class OntologyRule:
    code: str
    positive_patterns: List[str]
    negative_patterns: List[str]
    min_score: int


def load_ontology(schema_path: str) -> Tuple[Dict, List[OntologyRule]]:
    schema = load_yaml(schema_path)
    scoring = schema.get("scoring", {}) or {}
    onto = schema.get("ontology", {}) or {}

    rules: List[OntologyRule] = []
    for code, spec in onto.items():
        pos = spec.get("positive_patterns", []) or []
        neg = spec.get("negative_patterns", []) or []
        min_score = int(spec.get("min_score", 1))
        rules.append(OntologyRule(code=str(code), positive_patterns=pos, negative_patterns=neg, min_score=min_score))

    return scoring, rules


def score_text(title: str, body: str, rule: OntologyRule, title_weight: int, body_weight: int) -> int:
    """
    Simple interpretable scoring:
    - +title_weight for each positive pattern found in title
    - +body_weight  for each positive pattern found in body
    - If any negative pattern matches anywhere -> score = 0 (hard veto)
    """
    t = (title or "").lower()
    b = (body or "").lower()
    full = t + "\n" + b

    for pat in rule.negative_patterns:
        try:
            if re.search(pat, full, flags=re.IGNORECASE):
                return 0
        except re.error:
            # ignore broken regex in schema rather than crash
            continue

    score = 0
    for pat in rule.positive_patterns:
        try:
            if re.search(pat, t, flags=re.IGNORECASE):
                score += title_weight
            if re.search(pat, b, flags=re.IGNORECASE):
                score += body_weight
        except re.error:
            continue

    return score


def stable_article_id(url: str, text_sha: str) -> str:
    base = (text_sha or "") + "|" + (url or "")
    return hashlib.sha256(base.encode("utf-8", errors="ignore")).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description="Label news articles into event codes and aggregate monthly counts.")
    ap.add_argument("--pipeline", required=True, help="configs/pipeline.yaml")
    ap.add_argument("--schema", required=True, help="configs/labeling_schema.yaml")
    args = ap.parse_args()

    pipeline_cfg = load_yaml(args.pipeline)
    paths = get_pipeline_paths(pipeline_cfg)
    processed_dir = Path(paths["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    scoring_cfg, rules = load_ontology(args.schema)
    title_weight = int(scoring_cfg.get("title_weight", 2))
    body_weight = int(scoring_cfg.get("body_weight", 1))

    in_csv = processed_dir / "news_articles.csv"
    if not in_csv.exists():
        raise SystemExit(f"Missing input: {in_csv}")

    df = pd.read_csv(in_csv)

    if "article_id" not in df.columns:
        df["article_id"] = df.apply(lambda r: stable_article_id(r.get("url", ""), r.get("text_sha256", "")), axis=1)

    df["is_listing"] = df.apply(
        lambda r: is_probable_listing_page(
            r.get("url", ""), r.get("title", ""), r.get("text", ""), r.get("published_raw", ""), r.get("published_at", "")
        ),
        axis=1,
    )
    df = df[~df["is_listing"]].copy()

    # Build event_time_utc (published_at preferred)
    pub = coerce_dt(df["published_at"]) if "published_at" in df.columns else pd.Series([pd.NaT] * len(df), dtype="datetime64[ns, UTC]")
    missing = pub.isna()
    if missing.any() and "published_raw" in df.columns:
        pub2 = coerce_dt(df.loc[missing, "published_raw"])
        pub.loc[missing] = pub2

    url_dt = df["url"].fillna("").map(date_from_url)
    url_dt = pd.to_datetime(url_dt, errors="coerce", utc=True)
    fetched = coerce_dt(df["fetched_at_utc"]) if "fetched_at_utc" in df.columns else pd.Series([pd.NaT] * len(df), dtype="datetime64[ns, UTC]")

    df["event_time_utc"] = pub.fillna(url_dt).fillna(fetched)
    df["period_year"] = df["event_time_utc"].dt.year.astype("Int64")
    df["period_month"] = df["event_time_utc"].dt.month.astype("Int64")

    # Apply labeling
    hits_rows: List[Dict] = []
    matrix_rows: List[Dict] = []
    total_hits = 0

    for _, row in df.iterrows():
        article_id = row["article_id"]
        url = row.get("url")
        title = row.get("title", "")
        body = row.get("text", "")

        base = {
            "article_id": article_id,
            "source_id": row.get("source_id"),
            "url": url,
            "title": title,
            "fetched_at_utc": row.get("fetched_at_utc"),
            "published_raw": row.get("published_raw"),
            "published_at": row.get("published_at"),
            "event_time_utc": row.get("event_time_utc"),
            "period_year": int(row["period_year"]) if pd.notna(row["period_year"]) else None,
            "period_month": int(row["period_month"]) if pd.notna(row["period_month"]) else None,
        }

        row_hits: Dict[str, int] = {}
        hit_count = 0

        for rule in rules:
            s = score_text(title, body, rule, title_weight=title_weight, body_weight=body_weight)
            hit = 1 if s >= rule.min_score else 0
            row_hits[rule.code] = hit
            hit_count += hit

            if hit:
                hits_rows.append({**base, "event_code": rule.code, "score": s})

        total_hits += hit_count
        matrix_rows.append({**base, **row_hits})

    labeled_csv = processed_dir / "news_events_labeled.csv"
    matrix_csv = processed_dir / "news_events_matrix.csv"
    monthly_csv = processed_dir / "monthly_event_counts.csv"

    df_hits = pd.DataFrame(hits_rows)
    df_mat = pd.DataFrame(matrix_rows)

    df_hits.to_csv(labeled_csv, index=False, encoding="utf-8")
    df_mat.to_csv(matrix_csv, index=False, encoding="utf-8")

    event_cols = [r.code for r in rules]
    grp = df_mat.groupby(["period_year", "period_month"], dropna=False)[event_cols].sum().reset_index()
    grp.to_csv(monthly_csv, index=False, encoding="utf-8")

    print(f"Saved: {labeled_csv} | hit_rows={len(df_hits)} | total_hits={total_hits} | articles_kept={len(df_mat)}")
    print(f"Saved: {matrix_csv} | rows={len(df_mat)}")
    print(f"Saved: {monthly_csv} | rows={len(grp)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
