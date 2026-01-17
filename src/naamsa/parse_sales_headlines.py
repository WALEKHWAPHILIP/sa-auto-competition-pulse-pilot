from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pdfplumber

from src.utils.config import load_yaml, get_pipeline_paths


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

MIN_TOTAL_SALES = 10_000
MAX_TOTAL_SALES = 200_000

POS_WORDS = [
    "aggregate",
    "overall",
    "total",
    "reported industry",
    "industry",
    "domestic",
    "new vehicle sales",
    "industry sales",
    "new vehicle",
]
NEG_WORDS = [
    "passenger",
    "light commercial",
    "lcv",
    "truck",
    "trucks",
    "medium and heavy commercial",
    "mhcv",
    "bus",
    "buses",
    "rental",
    "fleet",
    "dealer",
    # exports often separate; mild negative to prefer domestic totals
    "export",
    "exports",
]

# number formats like 45,123 or 45 123 or 45123/123456
NUM = r"(\d{1,3}(?:[,\s]\d{3})+|\d{5,6})"

PATTERNS = [
    rf"(aggregate|overall|total)\s+(domestic\s+)?(industry\s+)?new\s+vehicle\s+sales[^0-9]{{0,160}}{NUM}",
    rf"(aggregate|overall|total)\s+(reported\s+)?industry\s+sales[^0-9]{{0,160}}{NUM}",
    rf"(aggregate|overall|total)[^0-9]{{0,140}}\bsales\b[^0-9]{{0,160}}{NUM}",
    rf"new\s+vehicle\s+sales[^0-9]{{0,140}}(aggregate|overall|total)[^0-9]{{0,160}}{NUM}",
]


def normalize_int(num_str: str) -> Optional[int]:
    s = re.sub(r"[^\d]", "", num_str)
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def clean_text(text: str) -> str:
    # lightweight cleanup for common mojibake + whitespace
    t = text
    t = t.replace("ΓÇÖ", "'")
    t = t.replace("ΓÇô", "-")
    t = t.replace("ΓÇ£", '"').replace("ΓÇ¥", '"')
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def parse_period_from_filename(name: str) -> Tuple[Optional[int], Optional[int]]:
    """
    FIX: Prefer the actual report month name in filename (April/May/June)
    rather than the downloader prefix YYYY_MM_ (upload month).
    """
    low = name.lower()

    # downloader prefix (upload month)
    pref_year = None
    pref_month = None
    m_pref = re.match(r"^(19\d{2}|20\d{2})_(0[1-9]|1[0-2])_", low)
    if m_pref:
        pref_year = int(m_pref.group(1))
        pref_month = int(m_pref.group(2))

    # Find report month by name (highest priority)
    report_month = None
    for mn, mi in MONTHS.items():
        if mn in low:
            report_month = mi
            break

    # Find year anywhere (prefer year near month name)
    # Usually filenames include "...April-2025..." etc.
    m_year = re.search(r"(19\d{2}|20\d{2})", low)
    report_year = int(m_year.group(1)) if m_year else None

    # If month name exists, use it; year from name if possible else fall back to prefix year
    if report_month is not None:
        if report_year is None:
            report_year = pref_year
        return report_year, report_month

    # Otherwise fall back to prefix
    if pref_year is not None and pref_month is not None:
        return pref_year, pref_month

    # Last resort: year only
    return report_year, None


def score_window(wlow: str) -> int:
    s = 0
    for p in POS_WORDS:
        if p in wlow:
            s += 2
    for n in NEG_WORDS:
        if n in wlow:
            s -= 2
    return s


def extract_text_plus_tables(pdf_path: Path, max_pages: int) -> str:
    """
    NEW: include table text, which is where Flash Report Summaries often hide totals.
    """
    parts: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        n = min(len(pdf.pages), max_pages)
        for i in range(n):
            page = pdf.pages[i]
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)

            # Try tables
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []

            for table in tables:
                # table is list of rows; row is list of cells
                for row in table:
                    if not row:
                        continue
                    row_txt = " ".join([c for c in row if isinstance(c, str) and c.strip()])
                    if row_txt.strip():
                        parts.append(row_txt)

    return "\n".join(parts)


def find_sales_number(text: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    text_clean = clean_text(text)
    low = text_clean.lower()

    candidates: List[Tuple[int, int, str, str]] = []  # (score, value, pattern, window)

    # 1) Pattern-based candidates
    for pat in PATTERNS:
        for m in re.finditer(pat, low, flags=re.IGNORECASE | re.DOTALL):
            raw_num = m.groups()[-1]  # last captured group is number
            val = normalize_int(raw_num)
            if val is None:
                continue
            if not (MIN_TOTAL_SALES <= val <= MAX_TOTAL_SALES):
                continue

            start = max(m.start(), 0)
            end = min(m.end(), len(text_clean))
            window = text_clean[max(0, start - 200) : min(len(text_clean), end + 200)]
            s = score_window(window.lower())
            candidates.append((s, val, pat, window.strip()))

    # 2) Fallback scan: if patterns fail, scan all plausible numbers and score context
    if not candidates:
        for m in re.finditer(NUM, low):
            raw_num = m.group(1)
            val = normalize_int(raw_num)
            if val is None:
                continue
            if not (MIN_TOTAL_SALES <= val <= MAX_TOTAL_SALES):
                continue

            start = max(m.start(), 0)
            end = min(m.end(), len(text_clean))
            window = text_clean[max(0, start - 220) : min(len(text_clean), end + 220)]
            wlow = window.lower()

            # Require some core context
            if "sales" not in wlow:
                continue
            if ("vehicle" not in wlow) and ("industry" not in wlow):
                continue

            s = score_window(wlow)
            candidates.append((s, val, "FALLBACK_NUMBER_SCAN", window.strip()))

    if not candidates:
        return None, None, None

    # Prefer highest score; if tie, larger number wins (totals often bigger)
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_score, best_val, best_pat, best_window = candidates[0]

    # Require at least some signal; patterns already enforce "total/aggregate" so 0 is OK
    if best_pat != "FALLBACK_NUMBER_SCAN" and best_score < 0:
        return None, None, None
    if best_pat == "FALLBACK_NUMBER_SCAN" and best_score < 2:
        return None, None, None

    return best_val, best_pat, best_window


def read_manifest_success(manifest_path: Path) -> List[Dict]:
    recs: List[Dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("status") == "success" and rec.get("saved_path"):
                recs.append(rec)
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse NAAMSA PDFs and extract headline total sales numbers.")
    ap.add_argument("--pipeline", required=True, help="Path to configs/pipeline.yaml")
    args = ap.parse_args()

    pipeline_cfg = load_yaml(args.pipeline)
    paths = get_pipeline_paths(pipeline_cfg)
    raw_naamsa_dir = Path(paths["raw_naamsa_dir"])
    processed_dir = Path(paths["processed_dir"])

    naamsa_cfg = pipeline_cfg.get("naamsa", {})
    max_pages = int(naamsa_cfg.get("max_parse_pages", 20))

    manifest_path = raw_naamsa_dir / "pdf_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run download first: python -m src.naamsa.download_pdfs ..."
        )

    recs = read_manifest_success(manifest_path)
    if not recs:
        print("No successful downloads recorded in manifest. Nothing to parse.")
        return 2

    rows: List[Dict] = []
    for rec in recs:
        pdf_path = Path(str(rec["saved_path"]))
        if not pdf_path.exists():
            continue

        y1, m1 = parse_period_from_filename(pdf_path.name)

        text = extract_text_plus_tables(pdf_path, max_pages=max_pages)
        if not text.strip():
            rows.append(
                {
                    "period_year": y1,
                    "period_month": m1,
                    "total_sales_units": None,
                    "match_pattern": None,
                    "evidence_window": None,
                    "pdf_path": str(pdf_path),
                    "pdf_sha256": rec.get("sha256"),
                    "pdf_url": rec.get("pdf_url"),
                }
            )
            continue

        val, pat, window = find_sales_number(text)
        rows.append(
            {
                "period_year": y1,
                "period_month": m1,
                "total_sales_units": val,
                "match_pattern": pat,
                "evidence_window": window,
                "pdf_path": str(pdf_path),
                "pdf_sha256": rec.get("sha256"),
                "pdf_url": rec.get("pdf_url"),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["period_year", "period_month", "pdf_path"],
        ascending=[False, False, True],
        na_position="last",
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "naamsa_monthly_sales_headlines.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    found = out["total_sales_units"].notna().sum()
    print(f"Saved: {out_path} | parsed={len(out)} | with_sales_number={found} | max_pages={max_pages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
