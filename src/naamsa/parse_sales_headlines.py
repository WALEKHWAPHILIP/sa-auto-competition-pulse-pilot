# src\naamsa\parse_sales_headlines.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.utils.config import load_yaml, get_pipeline_paths
from src.utils.snapshot import extract_text_plus_tables

# =============================================================================
# NAAMSA monthly total-sales extractor (research-grade, deterministic)
#
# Writes:
#   data\processed\naamsa_monthly_sales_headlines.csv
#
# Stable output columns:
#   period_year               int
#   period_month              int (1-12)
#   total_sales_units         int (nullable)
#   extraction_method         {HEADLINE_REGEX, ANCHORED_INDUSTRY_TOTAL, FALLBACK_SCAN}
#   extraction_confidence     {high, medium, low} (blank if total_sales_units blank)
#   warning_flag              blank or {MULTIPLE_TOTALS_NEARBY, OUT_OF_RANGE, AMBIGUOUS}
#   evidence_pattern          specific pattern tag used (audit)
#   evidence_window           local context used to extract (audit)
#   pdf_path                  str (audit)
#
# Deterministic extraction hierarchy:
#   1) HEADLINE_REGEX (high)            -> tight media-release phrases
#        1a) regex patterns (very high)
#        1b) anchored forward-scan from "new vehicle sales" (still headline-stage; deterministic)
#   2) ANCHORED_INDUSTRY_TOTAL (medium) -> "Industry Total" anchored extraction (Flash Reports)
#   3) FALLBACK_SCAN (low)              -> guarded scan in a limited evidence window
#
# De-duplication by month:
#   Keep <= 1 row per (period_year, period_month), preferring:
#     HEADLINE_REGEX > ANCHORED_INDUSTRY_TOTAL > FALLBACK_SCAN
#   Tie-breaker: strongest anchor presence in evidence_window (deterministic).
# =============================================================================


MONTHS: Dict[str, int] = {
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
MONTH_NAMES: List[str] = [m.title() for m in MONTHS.keys()]
MONTH_RX = "(" + "|".join(MONTH_NAMES) + ")"

# Numeric token supports: "55,956" or "55 956" or "55956"
NUM_TOKEN = r"(\d{1,3}(?:[,\s]\d{3})+|\d+)"

# Broad acceptance (headline/anchored can accept but flag OUT_OF_RANGE if outside sane)
BROAD_MIN_TOTAL = 10_000
BROAD_MAX_TOTAL = 160_000  # allow some drift; we still flag OUT_OF_RANGE vs sane_min/sane_max

# Sane monthly range (strict for FALLBACK_SCAN)
DEFAULT_SANE_MIN = 15_000
DEFAULT_SANE_MAX = 70_000

# Evidence windows
EVIDENCE_PAD_HEADLINE = 280
EVIDENCE_PAD_ANCHOR = 600
EVIDENCE_PAD_FALLBACK = 380

# Headline anchored-scan parameters (deterministic, bounded)
HEAD_ANCHOR_SCAN_AHEAD = 450  # chars ahead of anchor phrase
HEAD_ANCHOR_VERB_LOOKBACK = 80  # chars before number inside scan window
HEAD_TEXT_CHARS_FOR_HEADLINES = 50_000

# Text limits (performance / safety)
MAX_TEXT_CHARS = 220_000


# ----------------------------
# Text handling / coercion
# ----------------------------

def clean_text(text: str) -> str:
    t = (text or "").replace("\u00a0", " ")
    t = (
        t.replace("ΓÇô", "-")
         .replace("ΓÇÖ", "'")
         .replace("ΓÇ£", '"')
         .replace("ΓÇ¥", '"')
         .replace("╬ô├ç├û", "'")
         .replace("╬ô├ç├┤", "-")
         .replace("╬ô├ç┬ú", '"')
         .replace("╬ô├ç┬Ñ", '"')
    )
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > MAX_TEXT_CHARS:
        t = t[:MAX_TEXT_CHARS]
    return t


def normalize_int(s: str) -> Optional[int]:
    if s is None:
        return None
    ss = str(s).strip().replace(",", "").replace(" ", "")
    if not ss:
        return None
    try:
        return int(ss)
    except Exception:
        return None


def safe_extract_text_and_tables(extracted: Any) -> Tuple[str, Any]:
    """
    extract_text_plus_tables() can return:
      - str
      - (text, tables) tuple
      - (text, tables, ...) tuple/list
    We never assume str; we coerce safely and keep tables separately.
    """
    if isinstance(extracted, str):
        return extracted, None

    if isinstance(extracted, (tuple, list)):
        text_part = extracted[0] if len(extracted) > 0 else ""
        tables_part = extracted[1] if len(extracted) > 1 else None
        if not isinstance(text_part, str):
            try:
                text_part = str(text_part)
            except Exception:
                text_part = ""
        return text_part, tables_part

    try:
        return str(extracted), None
    except Exception:
        return "", None


def tables_to_searchable_text(tables: Any, max_chars: int = 60_000) -> str:
    """
    Convert extracted tables into a compact searchable string.
    Deterministic + cheap (no heavy parsing).
    """
    if tables is None:
        return ""

    chunks: List[str] = []

    # pandas DataFrame
    try:
        import pandas as _pd
        if isinstance(tables, _pd.DataFrame):
            s = tables.to_string(index=False)
            return s[:max_chars]
    except Exception:
        pass

    def _iter_rows(obj: Any) -> Iterable[Any]:
        if obj is None:
            return []
        if isinstance(obj, dict):
            return obj.values()
        if isinstance(obj, (list, tuple)):
            return obj
        return [obj]

    try:
        for tbl in _iter_rows(tables):
            for row in _iter_rows(tbl):
                if isinstance(row, (list, tuple)):
                    line = " | ".join([str(x) for x in row if x is not None])
                else:
                    line = str(row)
                line = re.sub(r"\s+", " ", line).strip()
                if line:
                    chunks.append(line)
            chunks.append("")
    except Exception:
        try:
            chunks.append(str(tables))
        except Exception:
            return ""

    s = "\n".join(chunks).strip()
    return s[:max_chars]


def _window(text_clean: str, start: int, end: int, pad: int) -> str:
    a = max(0, start - pad)
    b = min(len(text_clean), end + pad)
    return text_clean[a:b].strip()


# ----------------------------
# Period parsing
# ----------------------------

def parse_period_from_filename(name: str) -> Tuple[Optional[int], Optional[int]]:
    low = (name or "").lower()

    pref_year = None
    pref_month = None
    m_pref = re.match(r"^(19\d{2}|20\d{2})_(0[1-9]|1[0-2])_", low)
    if m_pref:
        pref_year = int(m_pref.group(1))
        pref_month = int(m_pref.group(2))

    report_month = None
    for mn, mi in MONTHS.items():
        if mn in low:
            report_month = mi
            break

    m_year = re.search(r"(19\d{2}|20\d{2})", low)
    report_year = int(m_year.group(1)) if m_year else None

    if report_month is not None:
        if report_year is None:
            report_year = pref_year
        return report_year, report_month

    if pref_year is not None and pref_month is not None:
        return pref_year, pref_month

    return report_year, None


def parse_period_from_text(text: str) -> Optional[Tuple[int, int]]:
    """
    Extract (year, month) from PDF text.
    Prioritize explicit "month of X YYYY" and "X YYYY ... sales" patterns.
    """
    if not text:
        return None

    head = " ".join((text or "").split())[:16000]
    head_low = head.lower()

    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    month_rx = "(" + "|".join(month_names) + ")"
    m_map = {name.lower(): i + 1 for i, name in enumerate(month_names)}

    looks_like_sales_report = any(
        k in head_low for k in ["new vehicle sales", "vehicle sales", "flash report", "industry total", "naamsa", "sales"]
    )

    patterns = [
        rf"(?:for|during)\s+(?:the\s+)?month\s+of\s+{month_rx}\s+(19\d{{2}}|20\d{{2}})",
        rf"\b{month_rx}\s+(19\d{{2}}|20\d{{2}})\s+(?:new\s+vehicle\s+)?sales\b",
        rf"\bIn\s+{month_rx}\s+(19\d{{2}}|20\d{{2}})\b",
    ]

    for pat in patterns:
        m = re.search(pat, head, flags=re.IGNORECASE)
        if not m:
            continue
        month_str = m.group(1)
        year = int(m.group(2))
        month = m_map.get(month_str.lower())
        if month and 1 <= month <= 12 and 1900 <= year <= 2100:
            return year, month

    # fallback only if sales-like AND "sales" near the Month Year mention
    if looks_like_sales_report:
        m2 = re.search(rf"\b{month_rx}\s+(19\d{{2}}|20\d{{2}})\b", head, flags=re.IGNORECASE)
        if m2:
            a = max(0, m2.start() - 80)
            b = min(len(head), m2.end() + 80)
            near = head[a:b].lower()
            if "sales" in near:
                month_str = m2.group(1)
                year = int(m2.group(2))
                month = m_map.get(month_str.lower())
                if month and 1 <= month <= 12 and 1900 <= year <= 2100:
                    return year, month

    return None


# ----------------------------
# Deterministic sales extraction
# ----------------------------

_BAD_NEAR_PATTERNS = [
    re.compile(r"%"),
    re.compile(r"\bkg\b"),
    re.compile(r">\s*8500\s*kg"),
    re.compile(r"\bytd\b|\byear\s*to\s*date\b|\byear-to-date\b"),
]

EXPORT_RX = re.compile(r"\bexport(s)?\b", flags=re.IGNORECASE)

# “full year” appears in December media release; do not reject it globally,
# but penalize candidates that are very close to "full year" if a better month-matched candidate exists.
FULL_YEAR_RX = re.compile(r"\bfull\s+year\b|\bcalendar\s+year\b|\bannual\b", flags=re.IGNORECASE)

# Anchor phrase variants (media releases)
NEW_VEHICLE_SALES_ANCHOR_RX = re.compile(
    r"\bnew\s*[- ]\s*vehicle\s+sales\b", flags=re.IGNORECASE
)

# Verbs that typically introduce the monthly total near the anchor
ANCHOR_TOTAL_VERB_RX = re.compile(
    r"\b(?:were|was|stood\s+at|came\s+in\s+at|recorded\s+at|totalled|totaled|amounted\s+to|at)\b",
    flags=re.IGNORECASE,
)


def _is_bad_context_near(text: str, idx: int, bad_patterns: List[re.Pattern], span: int = 20) -> bool:
    a = max(0, idx - span)
    b = min(len(text), idx + span)
    snippet = text[a:b].lower()
    return any(p.search(snippet) for p in bad_patterns)


def _export_too_close(text: str, idx: int, span: int = 10) -> bool:
    a = max(0, idx - span)
    b = min(len(text), idx + span)
    return bool(EXPORT_RX.search(text[a:b]))


def _full_year_close(text: str, idx: int, span: int = 25) -> bool:
    a = max(0, idx - span)
    b = min(len(text), idx + span)
    return bool(FULL_YEAR_RX.search(text[a:b]))


def _compute_warning_for_picked(val: int, sane_min: int, sane_max: int, current_warning: str = "") -> str:
    if val is None:
        return current_warning
    if val < sane_min or val > sane_max:
        return current_warning or "OUT_OF_RANGE"
    return current_warning


def _acceptable_broad(val: int) -> bool:
    return (val is not None) and (BROAD_MIN_TOTAL <= val <= BROAD_MAX_TOTAL)


def _month_name_from_int(m: Optional[int]) -> str:
    if not m or not (1 <= m <= 12):
        return ""
    return MONTH_NAMES[m - 1]


def _extract_number_group_last(m: re.Match) -> Optional[Tuple[str, int, int]]:
    """
    Deterministically return the LAST capture group that contains a digit.
    (Robust even when patterns capture Month/Year before the number.)
    Returns (raw, start, end).
    """
    last_idx = None
    for i in range(1, m.re.groups + 1):
        g = m.group(i)
        if g and re.search(r"\d", g):
            last_idx = i
    if last_idx is None:
        return None
    raw = m.group(last_idx)
    s, e = m.span(last_idx)
    return raw, s, e


# 1) HEADLINE_REGEX (high confidence)
# (Your updated list; kept exactly, plus we add anchored scan below when no regex match hits.)
HEADLINE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "AGGREGATE_NEW_VEHICLE_SALES_TO_UNITS",
        re.compile(
            r"\baggregate\b.{0,160}\bnew[- ]vehicle\b.{0,120}\bsales\b.{0,140}\b(?:rose|increased|climbed|improved|grew)\b.{0,120}\bto\b.{0,40}"
            + NUM_TOKEN
            + r".{0,40}\b(?:units|vehicles)\b",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "AGGREGATE_NEW_VEHICLE_SALES_AT_UNITS",
        re.compile(
            r"\baggregate\b.{0,220}\bnew[- ]vehicle\b.{0,160}\bsales\b.{0,120}\b(?:at|were\s+at|came\s+in\s+at|recorded\s+at)\b.{0,40}"
            + NUM_TOKEN
            + r".{0,40}\b(?:units|vehicles)\b",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "TOTAL_REPORTED_INDUSTRY_SALES_OF_VEHICLES",
        re.compile(
            r"\btotal\b.{0,60}\breported\b.{0,60}\bindustry\b.{0,60}\bsales\b.{0,60}\b(?:of|at|were)\b.{0,40}"
            + NUM_TOKEN
            + r".{0,40}\b(?:units|vehicles)\b",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "NEW_VEHICLE_SALES_TOTALLED_UNITS",
        re.compile(
            r"\bnew[- ]vehicle\b.{0,80}\bsales\b.{0,140}\b(?:totalled|totaled|amounted\s+to|were|stood\s+at|came\s+in\s+at|recorded)\b.{0,50}"
            + NUM_TOKEN
            + r".{0,50}\b(?:units|vehicles)\b",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    ),
    # --- Added: common NAAMSA media-release phrasing (still tight & anchored) ---
    (
        "INDUSTRY_NEW_VEHICLE_SALES_RECORDED_UNITS",
        re.compile(
            r"\bindustry\b.{0,60}\bnew[- ]vehicle\b.{0,60}\bsales\b.{0,120}\b(?:at|were\s+at|came\s+in\s+at|recorded\s+at|totalled|totaled)\b.{0,50}"
            + NUM_TOKEN
            + r".{0,50}\b(?:units|vehicles)\b",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "TOTAL_NEW_VEHICLE_SALES_WERE_UNITS",
        re.compile(
            r"\b(?:total|overall)\b.{0,80}\bnew[- ]vehicle\b.{0,60}\bsales\b.{0,120}\b(?:were|stood\s+at|came\s+in\s+at|recorded\s+at)\b.{0,50}"
            + NUM_TOKEN
            + r".{0,50}\b(?:units|vehicles)\b",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    ),
]


def _headline_sort_key(
    evidence: str,
    match_start: int,
    period_year: Optional[int],
    period_month: Optional[int],
    val: int,
    sane_min: int,
    sane_max: int,
    export_close: bool,
    full_year_close: bool,
    verb_present: bool,
) -> Tuple[int, int, int, int, int, int, int, int]:
    """
    Deterministic priority for headline candidates.
    Higher is better.
    """
    ev = (evidence or "").lower()
    month_name = _month_name_from_int(period_month).lower()
    year_str = str(period_year) if period_year else ""

    matches_period = 0
    if month_name and month_name in ev:
        matches_period += 1
    if year_str and year_str in evidence:
        matches_period += 1  # 0..2

    has_domestic = 1 if "domestic" in ev else 0
    has_aggregate = 1 if "aggregate" in ev else 0
    in_sane = 1 if (sane_min <= val <= sane_max) else 0

    # deterministic penalties encoded as binary boosts:
    not_export_close = 0 if export_close else 1
    not_full_year_close = 0 if full_year_close else 1
    has_verb = 1 if verb_present else 0

    # Earlier matches slightly preferred only after semantic signals:
    return (
        matches_period,          # most important
        has_domestic,
        has_aggregate,
        has_verb,                # helps media releases
        not_full_year_close,     # avoid annual figure when month matched exists
        not_export_close,        # avoid exports numbers
        in_sane,                 # prefer sane, but allow out-of-range with warning
        -match_start,            # deterministic tie-breaker
    )


def extract_headline_total(
    text_clean: str,
    sane_min: int,
    sane_max: int,
    period_year: Optional[int] = None,
    period_month: Optional[int] = None,
) -> Tuple[Optional[int], str, str, str, str, str]:
    """
    HEADLINE stage extraction:
      1) Try regex patterns (tight).
      2) If none, do deterministic anchored scan from "new vehicle sales" phrase:
         - search forward only up to HEAD_ANCHOR_SCAN_AHEAD chars
         - apply guardrails
         - score candidates deterministically
         - if ambiguous -> return None (no invention)
    """
    head = text_clean[:HEAD_TEXT_CHARS_FOR_HEADLINES]

    candidates: List[Tuple[Tuple[int, int, int, int, int, int, int, int], int, str, str, bool]] = []
    # tuple: (sort_key, value, pattern_tag, evidence, is_anchored_scan)

    # 1) regex patterns
    for tag, rx in HEADLINE_PATTERNS:
        for m in rx.finditer(head):
            ng = _extract_number_group_last(m)
            if not ng:
                continue
            raw, num_s, num_e = ng
            val = normalize_int(raw)
            if val is None or not _acceptable_broad(val):
                continue

            if _is_bad_context_near(head, num_s, _BAD_NEAR_PATTERNS, span=14):
                continue

            export_close = _export_too_close(head, num_s, span=10)
            full_year_close = _full_year_close(head, num_s, span=25)

            evidence = _window(text_clean, m.start(), m.end(), pad=EVIDENCE_PAD_HEADLINE)
            # verb presence inside evidence (helps score but does not change determinism)
            verb_present = bool(ANCHOR_TOTAL_VERB_RX.search(evidence))

            sk = _headline_sort_key(
                evidence=evidence,
                match_start=m.start(),
                period_year=period_year,
                period_month=period_month,
                val=val,
                sane_min=sane_min,
                sane_max=sane_max,
                export_close=export_close,
                full_year_close=full_year_close,
                verb_present=verb_present,
            )
            candidates.append((sk, val, tag, evidence, False))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_key, best_val, best_tag, best_ev, _ = candidates[0]
        warn = _compute_warning_for_picked(best_val, sane_min, sane_max, "")
        return best_val, "HEADLINE_REGEX", "high", best_tag, best_ev, warn

    # 2) anchored scan (still HEADLINE stage)
    anchor_matches = list(NEW_VEHICLE_SALES_ANCHOR_RX.finditer(head))
    if not anchor_matches:
        return None, "", "", "", "", ""

    scan_candidates: List[Tuple[Tuple[int, int, int, int, int, int, int, int], int, str, str, int, int]] = []
    # (sort_key, val, tag, evidence, num_abs_start, num_abs_end)

    for am in anchor_matches[:6]:  # deterministic bound
        a_start = am.start()
        a_end = am.end()
        scan_start = a_end
        scan_end = min(len(head), scan_start + HEAD_ANCHOR_SCAN_AHEAD)
        scan = head[scan_start:scan_end]

        # find numbers after the anchor phrase
        for nm in re.finditer(NUM_TOKEN, scan):
            raw = nm.group(1)
            val = normalize_int(raw)
            if val is None or not _acceptable_broad(val):
                continue

            num_abs_start = scan_start + nm.start(1)
            num_abs_end = scan_start + nm.end(1)

            if _is_bad_context_near(head, num_abs_start, _BAD_NEAR_PATTERNS, span=14):
                continue
            if _export_too_close(head, num_abs_start, span=10):
                continue

            # Determine if a total-verb is close before the number within the scan slice
            vb_a = max(0, nm.start(1) - HEAD_ANCHOR_VERB_LOOKBACK)
            vb_b = nm.start(1)
            verb_present = bool(ANCHOR_TOTAL_VERB_RX.search(scan[vb_a:vb_b]))

            full_year_close = _full_year_close(head, num_abs_start, span=25)

            evidence = _window(text_clean, num_abs_start, num_abs_end, pad=EVIDENCE_PAD_HEADLINE)

            sk = _headline_sort_key(
                evidence=evidence,
                match_start=a_start,
                period_year=period_year,
                period_month=period_month,
                val=val,
                sane_min=sane_min,
                sane_max=sane_max,
                export_close=False,  # already filtered exports close
                full_year_close=full_year_close,
                verb_present=verb_present,
            )

            scan_candidates.append((sk, val, "NEW_VEHICLE_SALES_ANCHORED_SCAN", evidence, num_abs_start, num_abs_end))

    if not scan_candidates:
        return None, "", "", "", "", ""

    scan_candidates.sort(key=lambda x: x[0], reverse=True)

    # Ambiguity rule (deterministic):
    # If the top-2 candidates have identical sort_key and different values -> ambiguous -> blank.
    if len(scan_candidates) >= 2:
        if scan_candidates[0][0] == scan_candidates[1][0] and scan_candidates[0][1] != scan_candidates[1][1]:
            return None, "", "", "", scan_candidates[0][3], "AMBIGUOUS"

    best_key, best_val, best_tag, best_ev, _, _ = scan_candidates[0]

    # confidence: high if verb_present contributed AND matches_period > 0, else medium (still HEADLINE stage)
    matches_period = best_key[0]
    has_verb = best_key[3]
    conf = "high" if (matches_period > 0 and has_verb > 0) else "medium"

    warn = _compute_warning_for_picked(best_val, sane_min, sane_max, "")
    return best_val, "HEADLINE_REGEX", conf, best_tag, best_ev, warn


# 2) ANCHORED_INDUSTRY_TOTAL
INDUSTRY_TOTAL_RX = re.compile(r"\bindustry\s+total\b", flags=re.IGNORECASE)


def _first_numbers_after_anchor(blob: str, anchor_match: re.Match, max_ahead: int) -> List[Tuple[int, int, int]]:
    start = anchor_match.end()
    end = min(len(blob), start + max_ahead)
    tail = blob[start:end]

    out: List[Tuple[int, int, int]] = []
    for nm in re.finditer(NUM_TOKEN, tail):
        val = normalize_int(nm.group(1))
        if val is None:
            continue
        abs_start = start + nm.start(1)
        abs_end = start + nm.end(1)
        out.append((val, abs_start, abs_end))
    return out


def extract_anchored_industry_total(
    text_clean: str, tables_text: str, sane_min: int, sane_max: int
) -> Tuple[Optional[int], str, str, str, str, str]:
    sources: List[Tuple[str, str]] = []
    if tables_text.strip():
        sources.append(("TABLES", tables_text))
    sources.append(("TEXT", text_clean))

    for src_tag, blob in sources:
        m = INDUSTRY_TOTAL_RX.search(blob)
        if not m:
            continue

        candidates = _first_numbers_after_anchor(blob, m, max_ahead=260)
        if not candidates:
            continue

        valid: List[Tuple[int, int, int]] = []
        for val, s, e in candidates:
            if not _acceptable_broad(val):
                continue
            if _is_bad_context_near(blob, s, _BAD_NEAR_PATTERNS, span=14):
                continue
            if _export_too_close(blob, s, span=8):
                continue
            valid.append((val, s, e))

        if not valid:
            continue

        picked_val, picked_s, picked_e = valid[0]

        warning_flag = _compute_warning_for_picked(picked_val, sane_min, sane_max, "")
        
        if not warning_flag:
            sane_valid = [t for t in valid if (sane_min <= t[0] <= sane_max)]
            if len(sane_valid) >= 2:
                first_val = sane_valid[0][0]
                second_val = sane_valid[1][0]
                if first_val > 0:
                    rel_gap = abs(first_val - second_val) / float(first_val)
                    if rel_gap <= 0.10:
                        warning_flag = "MULTIPLE_TOTALS_NEARBY"



        evidence_pattern = f"INDUSTRY_TOTAL_FIRST_NUMBER_{src_tag}"
        evidence = _window(blob, picked_s, picked_e, pad=EVIDENCE_PAD_ANCHOR)

        if "industry total" not in evidence.lower():
            anchor_win = _window(blob, m.start(), m.end(), pad=EVIDENCE_PAD_ANCHOR)
            if "industry total" in anchor_win.lower():
                evidence = anchor_win

        return picked_val, "ANCHORED_INDUSTRY_TOTAL", "medium", evidence_pattern, evidence, warning_flag

    return None, "", "", "", "", ""


# 3) FALLBACK_SCAN (strict sane bounds)
FALLBACK_ANCHORS: List[Tuple[str, re.Pattern]] = [
    ("INDUSTRY_TOTAL", re.compile(r"\bindustry\s+total\b", flags=re.IGNORECASE)),
    ("TOTAL_VEHICLE_SALES", re.compile(r"\btotal\s+vehicle\s+sales\b", flags=re.IGNORECASE)),
    ("NEW_VEHICLE_SALES", re.compile(r"\bnew\s*[- ]\s*vehicle\s+sales\b", flags=re.IGNORECASE)),
    ("DOMESTIC_NEW_VEHICLE_SALES", re.compile(r"\bdomestic\b.{0,20}\bnew\s*[- ]\s*vehicle\s+sales\b", flags=re.IGNORECASE)),
]


def extract_fallback_scan(text_clean: str, sane_min: int, sane_max: int) -> Tuple[Optional[int], str, str, str, str, str]:
    low = text_clean.lower()

    anchor_name = ""
    anchor_match: Optional[re.Match] = None
    for name, rx in FALLBACK_ANCHORS:
        m = rx.search(low)
        if m:
            anchor_name = name
            anchor_match = m
            break

    if anchor_match:
        win = _window(text_clean, anchor_match.start(), anchor_match.end(), pad=EVIDENCE_PAD_FALLBACK)
        win_offset = max(0, anchor_match.start() - EVIDENCE_PAD_FALLBACK)
        anchor_pos = anchor_match.start()
    else:
        win = text_clean[:9000]
        win_offset = 0
        anchor_pos = None

    candidates: List[Tuple[int, int]] = []
    for nm in re.finditer(NUM_TOKEN, win):
        val = normalize_int(nm.group(1))
        if val is None:
            continue

        if val < sane_min or val > sane_max:
            continue

        if _is_bad_context_near(win, nm.start(1), _BAD_NEAR_PATTERNS, span=14):
            continue

        if _export_too_close(win, nm.start(1), span=10):
            continue

        abs_pos = win_offset + nm.start(1)
        candidates.append((val, abs_pos))

    if not candidates:
        return None, "", "", "", "", ""

    if anchor_pos is not None:
        scored = sorted(candidates, key=lambda x: (abs(x[1] - anchor_pos), x[0]))
        best = scored[0]

        if len(scored) >= 2:
            d0 = abs(scored[0][1] - anchor_pos)
            d1 = abs(scored[1][1] - anchor_pos)
            if (d1 - d0) <= 10 and scored[1][0] != scored[0][0]:
                return None, "", "", "", win, "AMBIGUOUS"

        val = best[0]
        evidence = _window(text_clean, best[1], best[1] + 1, pad=EVIDENCE_PAD_FALLBACK)
        pattern = f"FALLBACK_SCAN_NEAR_{anchor_name or 'NO_ANCHOR'}"
        return val, "FALLBACK_SCAN", "low", pattern, evidence, ""

    unique_vals = sorted({v for v, _ in candidates})
    if len(unique_vals) >= 2:
        return None, "", "", "", win, "AMBIGUOUS"

    val = unique_vals[0]
    return val, "FALLBACK_SCAN", "low", "FALLBACK_SCAN_NO_ANCHOR", win, ""


def extract_monthly_total_sales(
    text: str,
    tables: Any,
    sane_min: int,
    sane_max: int,
    period_year: Optional[int] = None,
    period_month: Optional[int] = None,
) -> Dict[str, Any]:
    text_clean = clean_text(text)
    tables_text = clean_text(tables_to_searchable_text(tables))

    total, method, conf, pattern, evidence, warn = extract_headline_total(
        text_clean, sane_min, sane_max, period_year=period_year, period_month=period_month
    )
    if total is not None:
        return {
            "total_sales_units": total,
            "extraction_method": method,
            "extraction_confidence": conf,
            "warning_flag": warn or "",
            "evidence_pattern": pattern,
            "evidence_window": evidence,
        }

    total, method, conf, pattern, evidence, warn = extract_anchored_industry_total(
        text_clean, tables_text, sane_min, sane_max
    )
    if total is not None:
        return {
            "total_sales_units": total,
            "extraction_method": method,
            "extraction_confidence": conf,
            "warning_flag": warn or "",
            "evidence_pattern": pattern,
            "evidence_window": evidence,
        }

    total, method, conf, pattern, evidence, warn = extract_fallback_scan(text_clean, sane_min, sane_max)
    if total is not None:
        return {
            "total_sales_units": total,
            "extraction_method": method,
            "extraction_confidence": conf,
            "warning_flag": warn or "",
            "evidence_pattern": pattern,
            "evidence_window": evidence,
        }

    warning_flag = "AMBIGUOUS" if warn == "AMBIGUOUS" else ""
    return {
        "total_sales_units": None,
        "extraction_method": "",
        "extraction_confidence": "",
        "warning_flag": warning_flag,
        "evidence_pattern": "",
        "evidence_window": evidence or "",
    }


# ----------------------------
# Manifest helpers
# ----------------------------

def read_manifest_success(manifest_path: Path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("status") == "success" and rec.get("saved_path"):
                recs.append(rec)
    return recs


# ----------------------------
# De-duplication by month
# ----------------------------

_METHOD_RANK = {"HEADLINE_REGEX": 3, "ANCHORED_INDUSTRY_TOTAL": 2, "FALLBACK_SCAN": 1, "": 0, None: 0}
_CONF_RANK = {"high": 3, "medium": 2, "low": 1, "": 0, None: 0}

_STRONG_ANCHOR_PHRASES = [
    "aggregate new vehicle sales",
    "total reported industry sales",
    "industry total",
    "total vehicle sales",
    "domestic new vehicle sales",
    "new vehicle sales",
]


def _anchor_strength(evidence_window: str) -> int:
    w = (evidence_window or "").lower()
    s = 0
    for p in _STRONG_ANCHOR_PHRASES:
        if p in w:
            s += 2
    if "units" in w or "vehicles" in w:
        s += 1
    return s


def dedupe_by_month(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    work = df.copy()
    work["_method_rank"] = work["extraction_method"].map(_METHOD_RANK).fillna(0).astype(int)
    work["_conf_rank"] = work["extraction_confidence"].map(_CONF_RANK).fillna(0).astype(int)
    work["_anchor_strength"] = work["evidence_window"].apply(_anchor_strength).fillna(0).astype(int)
    work["_has_total"] = work["total_sales_units"].notna().astype(int)
    work["_pdf_path_key"] = work["pdf_path"].fillna("").astype(str)

    work = work.sort_values(
        by=["period_year", "period_month", "_method_rank", "_conf_rank", "_anchor_strength", "_has_total", "_pdf_path_key"],
        ascending=[True, True, False, False, False, False, True],
    )

    out = work.groupby(["period_year", "period_month"], as_index=False).head(1).copy()
    out = out.drop(columns=["_method_rank", "_conf_rank", "_anchor_strength", "_has_total", "_pdf_path_key"], errors="ignore")
    out = out.drop_duplicates(subset=["period_year", "period_month"], keep="first").copy()
    return out


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Parse NAAMSA PDFs and extract monthly total sales (deterministic).")
    ap.add_argument("--pipeline", required=True, help="Path to configs\\pipeline.yaml")
    args = ap.parse_args()

    pipeline_cfg = load_yaml(args.pipeline)
    paths = get_pipeline_paths(pipeline_cfg)

    raw_naamsa_dir = Path(paths["raw_naamsa_dir"])
    processed_dir = Path(paths["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    naamsa_cfg = pipeline_cfg.get("naamsa", {}) or {}
    max_pages = int(naamsa_cfg.get("max_parse_pages", 20))

    sane_min = int(naamsa_cfg.get("sane_min_total_sales", DEFAULT_SANE_MIN))
    sane_max = int(naamsa_cfg.get("sane_max_total_sales", DEFAULT_SANE_MAX))
    if sane_min >= sane_max:
        sane_min, sane_max = DEFAULT_SANE_MIN, DEFAULT_SANE_MAX

    manifest_path = raw_naamsa_dir / "pdf_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run download first:\n"
            f"  python -m src.naamsa.download_pdfs --pipeline configs\\pipeline.yaml --in_links data\\processed\\naamsa_links_to_fetch.csv"
        )

    recs = read_manifest_success(manifest_path)
    if not recs:
        print("No successful downloads recorded in manifest. Nothing to parse.")
        return 2

    rows: List[Dict[str, Any]] = []

    for rec in recs:
        pdf_path = Path(str(rec.get("saved_path", "")))
        if not pdf_path.exists():
            continue

        extracted = extract_text_plus_tables(pdf_path, max_pages=max_pages)
        text_part, tables_part = safe_extract_text_and_tables(extracted)

        period = parse_period_from_text(text_part)
        if period is not None:
            period_year, period_month = period
        else:
            period_year, period_month = parse_period_from_filename(pdf_path.name)

        sales = extract_monthly_total_sales(
            text_part,
            tables_part,
            sane_min=sane_min,
            sane_max=sane_max,
            period_year=period_year,
            period_month=period_month,
        )

        rows.append(
            {
                "period_year": period_year,
                "period_month": period_month,
                "total_sales_units": sales.get("total_sales_units"),
                "extraction_method": sales.get("extraction_method") or "",
                "extraction_confidence": sales.get("extraction_confidence") or "",
                "warning_flag": sales.get("warning_flag") or "",
                "evidence_pattern": sales.get("evidence_pattern") or "",
                "evidence_window": sales.get("evidence_window") or "",
                "pdf_path": str(pdf_path),
            }
        )

    if not rows:
        print("No PDFs parsed (rows=0).")
        return 0

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["period_year", "period_month"]).copy()
    if df.empty:
        out_path = processed_dir / "naamsa_monthly_sales_headlines.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Saved: {out_path} | rows=0")
        return 0

    df["period_year"] = df["period_year"].astype(int)
    df["period_month"] = df["period_month"].astype(int)

    df = dedupe_by_month(df)

    final_cols = [
        "period_year",
        "period_month",
        "total_sales_units",
        "extraction_method",
        "extraction_confidence",
        "warning_flag",
        "evidence_pattern",
        "evidence_window",
        "pdf_path",
    ]
    for c in final_cols:
        if c not in df.columns:
            df[c] = ""

    df = df[final_cols].copy()

    out_path = processed_dir / "naamsa_monthly_sales_headlines.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    missing = int(df["total_sales_units"].isna().sum())
    print(f"Saved: {out_path} | rows={len(df)} | missing_totals={missing} | sane_range={sane_min}-{sane_max}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
