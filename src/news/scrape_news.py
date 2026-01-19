from __future__ import annotations

import argparse
import json
import re
import time
import hashlib
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.config import load_yaml, get_pipeline_paths


# =============================================================================
# GOAL (why this file exists)
# =============================================================================
# This module scrapes publisher article pages into:
#   data/processed/news_articles.csv
#
# Key design requirements (research-grade / no guessing):
#   1) Dates must be stable and machine-parseable:
#      - published_at stored as: YYYY-MM-DDTHH:MM:SSZ (no microseconds)
#      - undated pages can be dropped deterministically via pipeline.yaml
#
#   2) Link hubs (aggregators) must not pollute the dataset:
#      - A hub page like news.co.za/motoring/ is NOT an article.
#      - We only use hubs to DISCOVER outbound publisher URLs.
#      - Those discovered URLs must be ROUTED into the correct publisher source_id.
#
#   3) Pagination must not bias the dataset when caps apply:
#      - If you cap max_articles_per_source_per_run, you should not accidentally
#        keep only “old” pages or only “new” pages.
#      - We therefore crawl pagination in a deterministic "balanced" order:
#        newest page, oldest page, next newest, next oldest, ...
# =============================================================================


# ----------------------------
# Helpers (hashing, filenames, timestamps)
# ----------------------------

INVALID_FS_CHARS = r'[<>:"/\\|?*\x00-\x1F]'


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes((s or "").encode("utf-8", errors="ignore"))


def safe_filename(s: str, max_len: int = 180) -> str:
    s = (s or "").strip()
    s = re.sub(INVALID_FS_CHARS, "_", s)
    s = re.sub(r"\s+", "_", s)
    if len(s) > max_len:
        s = s[:max_len]
    return s or "snapshot"


# ----------------------------
# Text normalization + extraction
# ----------------------------

def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\u00a0", " ")
    # light mojibake cleanup + whitespace normalization
    t = t.replace("ΓÇô", "-").replace("ΓÇÖ", "'").replace("ΓÇ£", '"').replace("ΓÇ¥", '"')
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def visible_text_from_node(node) -> str:
    for tag in node.find_all(["script", "style", "noscript"]):
        tag.decompose()
    txt = node.get_text(" ", strip=True)
    return normalize_text(txt)


def choose_main_text(soup: BeautifulSoup) -> str:
    # Try common article containers first
    for sel in [
        "article",
        "div.article",
        "div.article-content",
        "div.article__content",
        "div.post-content",
        "div.entry-content",
        "div.content",
        "main",
    ]:
        node = soup.select_one(sel)
        if node:
            txt = visible_text_from_node(node)
            if len(txt) >= 200:
                return txt

    # Fallback to body
    body = soup.body or soup
    return visible_text_from_node(body)


def extract_title(soup: BeautifulSoup) -> str:
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return normalize_text(og["content"])
    if soup.title and soup.title.string:
        return normalize_text(soup.title.string)
    h1 = soup.find("h1")
    if h1:
        return normalize_text(h1.get_text(" ", strip=True))
    return ""


def extract_author(soup: BeautifulSoup) -> str:
    m = soup.find("meta", attrs={"name": "author"})
    if m and m.get("content"):
        return normalize_text(m["content"])
    for cls in ["author", "byline", "article-author", "post-author"]:
        node = soup.find(class_=re.compile(cls, re.IGNORECASE))
        if node:
            txt = normalize_text(node.get_text(" ", strip=True))
            if 2 <= len(txt) <= 80:
                return txt
    return ""


# ----------------------------
# Date parsing (stable UTC ISO output)
# ----------------------------

def _ts_to_iso_utc(ts: pd.Timestamp) -> str:
    """
    Always output: YYYY-MM-DDTHH:MM:SSZ (NO microseconds).
    Avoids pandas parsing issues with fractional seconds + 'Z' on some builds.
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    dt = ts.to_pydatetime().replace(tzinfo=timezone.utc, microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


def _try_parse_dt(value: str) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        ts = pd.to_datetime(str(value).strip(), errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def _parse_date_from_text(text: str) -> Optional[pd.Timestamp]:
    """
    Last-resort: try to detect a human-readable publish date embedded in text.
    Keep this conservative to avoid false positives.
    """
    if not text:
        return None
    t = normalize_text(text)

    month_rx = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    m = re.search(
        r"\b(?P<d>[0-3]?\d)\s+(?P<m>" + month_rx + r")\s+(?P<y>19\d{2}|20\d{2})\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        d = int(m.group("d"))
        y = int(m.group("y"))
        mn = m.group("m").lower()
        key = mn[:3]
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        mm = month_map.get(key)
        if mm:
            try:
                return pd.Timestamp(year=y, month=mm, day=d, tz="UTC")
            except Exception:
                return None

    m2 = re.search(r"\b(?P<y>19\d{2}|20\d{2})[-/](?P<m>0[1-9]|1[0-2])[-/](?P<d>[0-3]\d)\b", t)
    if m2:
        try:
            return pd.Timestamp(
                year=int(m2.group("y")),
                month=int(m2.group("m")),
                day=int(m2.group("d")),
                tz="UTC",
            )
        except Exception:
            return None

    return None


def extract_published_info(soup: BeautifulSoup, main_text: str = "") -> Tuple[str, str]:
    """
    Returns:
      (published_at_iso_utc, published_raw)
    published_at_iso_utc is "" if not found/parseable.
    """
    raw = ""

    # 1) OpenGraph / article meta
    for attr in ["article:published_time", "og:published_time"]:
        m = soup.find("meta", attrs={"property": attr})
        if m and m.get("content"):
            raw = normalize_text(m["content"])
            break

    # 2) common meta name variants
    if not raw:
        for name in ["pubdate", "publishdate", "date", "dc.date", "dc.date.issued", "parsely-pub-date", "sailthru.date"]:
            m2 = soup.find("meta", attrs={"name": name})
            if m2 and m2.get("content"):
                raw = normalize_text(m2["content"])
                break

    # 3) <time datetime="...">
    if not raw:
        t = soup.find("time")
        if t:
            if t.get("datetime"):
                raw = normalize_text(t.get("datetime", ""))
            else:
                raw = normalize_text(t.get_text(" ", strip=True))

    # 4) JSON-LD datePublished
    if not raw:
        for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
            try:
                payload = json.loads(s.get_text() or "")
            except Exception:
                continue

            objs = payload if isinstance(payload, list) else [payload]
            found = None
            for obj in objs:
                if not isinstance(obj, dict):
                    continue

                if "@graph" in obj and isinstance(obj["@graph"], list):
                    for g in obj["@graph"]:
                        if isinstance(g, dict) and g.get("datePublished"):
                            found = g.get("datePublished")
                            break
                if found:
                    break

                if obj.get("datePublished"):
                    found = obj.get("datePublished")
                    break

            if found:
                raw = normalize_text(str(found))
                break

    ts = _try_parse_dt(raw)
    if ts is None:
        # conservative text sniffing (avoid scanning entire article)
        head = (main_text or "")[:700]
        ts = _parse_date_from_text(head) or _parse_date_from_text(main_text)

    if ts is None:
        return "", raw

    return _ts_to_iso_utc(ts), raw


# ----------------------------
# Simhash (64-bit) for deduplication
# ----------------------------

def simhash64(text: str) -> int:
    text = (text or "").lower()
    if not text:
        return 0
    toks = re.findall(r"[a-z0-9]{2,}", text)
    if not toks:
        return 0

    freq: Dict[str, int] = {}
    for tok in toks:
        freq[tok] = freq.get(tok, 0) + 1

    v = [0] * 64
    for tok, w in freq.items():
        h = hashlib.md5(tok.encode("utf-8")).hexdigest()
        bits = int(h[:16], 16)
        for i in range(64):
            v[i] += w if ((bits >> i) & 1) else -w

    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= (1 << i)
    return out


# ----------------------------
# URL allow/deny filtering
# ----------------------------

def allowed_url(url: str, allow_patterns: List[str], deny_substrings: List[str]) -> bool:
    u = (url or "").strip()
    if not u:
        return False

    # deny substrings first (cheap and deterministic)
    for d in deny_substrings or []:
        if d and d in u:
            return False

    # allow patterns (regex)
    if not allow_patterns:
        return True

    for pat in allow_patterns:
        try:
            if re.search(pat, u, flags=re.IGNORECASE):
                return True
        except re.error:
            continue

    return False


# ----------------------------
# HTTP fetch helpers
# ----------------------------

def fetch_page(session: requests.Session, url: str, headers: Dict[str, str], timeout: int) -> Tuple[int, str]:
    """Returns (status_code, text). Never raises."""
    try:
        r = session.get(url, headers=headers, timeout=timeout)
        return int(r.status_code), (r.text or "")
    except Exception:
        return 0, ""


# ----------------------------
# Discovery (pagination + RSS)
# ----------------------------

def _balanced_page_order(low: int, high: int) -> List[int]:
    """
    Deterministic balanced order to avoid bias under caps:
      1, high, 2, high-1, 3, high-2, ...
    """
    out: List[int] = []
    a, b = low, high
    while a <= b:
        out.append(a)
        if b != a:
            out.append(b)
        a += 1
        b -= 1
    return out


def _find_highest_existing_page(
    session: requests.Session,
    page_pattern: str,
    start_page: int,
    max_pages: int,
    headers: Dict[str, str],
    timeout: int,
    rate: float,
) -> Optional[int]:
    """
    Find the highest page number (within the configured window) that returns HTTP 200.
    Deterministic: scan downward from (start_page + max_pages - 1) to start_page.
    """
    hi = start_page + max_pages - 1
    for p in range(hi, start_page - 1, -1):
        page_url = page_pattern.format(page=p)
        status, html = fetch_page(session, page_url, headers, timeout)
        if status == 200 and html:
            return p
        time.sleep(rate)
    return None


def discover_links_section_pagination(session: requests.Session, source_cfg: dict, http_cfg: dict) -> List[str]:
    """
    For sources with page_pattern:
      - find highest existing page within window
      - crawl pages in balanced order to avoid cap bias

    For sources without page_pattern:
      - crawl entry_points and follow "next" links (best-effort)
    """
    entry_points = source_cfg.get("discovery", {}).get("entry_points", []) or source_cfg.get("entry_points", []) or []
    pagination = source_cfg.get("pagination", {}) or {}
    page_pattern = pagination.get("page_pattern")
    start_page = int(pagination.get("start_page", 1))
    max_pages = int(pagination.get("max_pages", 40))

    allow_patterns = source_cfg.get("allow", []) or []
    deny_substrings = source_cfg.get("deny", []) or []

    headers = {"User-Agent": str(http_cfg.get("user_agent", "Academic research bot"))}
    timeout = int(http_cfg.get("timeout_seconds", 25))
    rate = float(http_cfg.get("rate_limit_seconds", 2))

    urls: List[str] = []
    seen: Set[str] = set()

    if page_pattern:
        hi_page = _find_highest_existing_page(session, page_pattern, start_page, max_pages, headers, timeout, rate)
        if hi_page is None:
            return []

        for p in _balanced_page_order(start_page, hi_page):
            page_url = page_pattern.format(page=p)
            status, html = fetch_page(session, page_url, headers, timeout)
            if status != 200 or not html:
                time.sleep(rate)
                continue

            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                full = urljoin(page_url, href)
                if full in seen:
                    continue
                if allowed_url(full, allow_patterns, deny_substrings):
                    seen.add(full)
                    urls.append(full)

            time.sleep(rate)

        return urls

    # fallback: crawl entry_points with next links
    for entry in entry_points:
        current = entry
        for _ in range(max_pages):
            status, html = fetch_page(session, current, headers, timeout)
            if status in (404, 410) or not html:
                break

            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                full = urljoin(current, href)
                if full in seen:
                    continue
                if allowed_url(full, allow_patterns, deny_substrings):
                    seen.add(full)
                    urls.append(full)

            nxt = None
            for a in soup.find_all("a", href=True):
                txt = (a.get_text(" ", strip=True) or "").lower()
                if txt in ("next", "older", "next page") or "next" in txt:
                    nxt = urljoin(current, a["href"].strip())
                    break
            if not nxt:
                break
            current = nxt
            time.sleep(rate)

    return urls


def discover_links_rss_feed(session: requests.Session, source_cfg: dict, http_cfg: dict) -> List[str]:
    entry_points = source_cfg.get("discovery", {}).get("entry_points", []) or source_cfg.get("entry_points", []) or []
    allow_patterns = source_cfg.get("allow", []) or []
    deny_substrings = source_cfg.get("deny", []) or []

    headers = {"User-Agent": str(http_cfg.get("user_agent", "Academic research bot"))}
    timeout = int(http_cfg.get("timeout_seconds", 25))

    urls: List[str] = []
    seen: Set[str] = set()

    for feed_url in entry_points:
        status, xml = fetch_page(session, feed_url, headers, timeout)
        if status in (404, 410) or not xml:
            continue

        soup = BeautifulSoup(xml, "xml")

        # RSS <item><link>...
        for item in soup.find_all("item"):
            link = item.find("link")
            href = (link.get_text() if link else "").strip()
            if href and href not in seen and allowed_url(href, allow_patterns, deny_substrings):
                seen.add(href)
                urls.append(href)

        # Atom <entry><link href="...">
        for entry in soup.find_all("entry"):
            link = entry.find("link", href=True)
            href = (link.get("href") if link else "").strip()
            if href and href not in seen and allowed_url(href, allow_patterns, deny_substrings):
                seen.add(href)
                urls.append(href)

    return urls


# ----------------------------
# Fetch snapshot to disk
# ----------------------------

def fetch_snapshot(session: requests.Session, url: str, out_dir: Path, http_cfg: dict) -> Tuple[Optional[Path], Optional[str], Optional[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": str(http_cfg.get("user_agent", "Academic research bot"))}
    timeout = int(http_cfg.get("timeout_seconds", 25))
    rate = float(http_cfg.get("rate_limit_seconds", 2))
    max_retries = int(http_cfg.get("max_retries", 3))

    parsed = urlparse(url)
    basename = safe_filename(parsed.path.strip("/").replace("/", "_") or "index")
    snap_path = out_dir / f"{basename}.html"

    # cache hit: reuse snapshot
    if snap_path.exists() and snap_path.stat().st_size > 500:
        b = snap_path.read_bytes()
        return snap_path, sha256_bytes(b), utc_now_iso()

    html = ""
    for _ in range(max_retries):
        status, html = fetch_page(session, url, headers, timeout)
        if status == 200 and html:
            break
        time.sleep(rate)

    if not html:
        return None, None, None

    b = html.encode("utf-8", errors="ignore")
    snap_sha = sha256_bytes(b)
    fetched_at = utc_now_iso()

    snap_path.write_bytes(b)
    time.sleep(rate)
    return snap_path, snap_sha, fetched_at


# ----------------------------
# Run window filters
# ----------------------------

def parse_date_ymd(s: object) -> Optional[date]:
    if s is None:
        return None
    ss = str(s).strip()
    if not ss or ss.lower() in {"none", "null", "nan"}:
        return None
    d = pd.to_datetime(ss, errors="coerce", utc=True)
    if pd.isna(d):
        return None
    return d.date()


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--sources", required=True)
    args = ap.parse_args()

    pipeline_cfg = load_yaml(args.pipeline)
    sources_cfg = load_yaml(args.sources)

    paths = get_pipeline_paths(pipeline_cfg)
    raw_news_dir = Path(paths["raw_news_dir"])
    processed_dir = Path(paths["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    http_cfg = pipeline_cfg.get("http", {}) or {}
    run_cfg = pipeline_cfg.get("run", {}) or {}

    max_per_source = int(run_cfg.get("max_articles_per_source_per_run", 250))
    start_date = parse_date_ymd(run_cfg.get("start_date"))
    end_date = parse_date_ymd(run_cfg.get("end_date"))
    min_year = int(run_cfg.get("min_published_year", 0) or 0)
    drop_undated = bool(run_cfg.get("drop_undated_articles", False))

    text_cfg = pipeline_cfg.get("text", {}) or {}
    min_chars = int(text_cfg.get("min_article_chars", 800))

    session = requests.Session()

    # -------------------------------------------------------------------------
    # Step 1: DISCOVERY
    #
    # We split sources into:
    #   A) discovery-only (hub) sources => contribute URLs to global pool
    #   B) real publisher sources (news_site/rss_feed) => scrape into dataset
    #
    # This solves the exact issue you observed:
    # - links discovered from news.co.za were being saved under source_id=news_sa_motoring
    # - now they will be routed into the correct publisher source_id
    # -------------------------------------------------------------------------
    hub_url_pool: List[str] = []
    discovered_by_source: Dict[str, List[str]] = {}

    for source in (sources_cfg.get("sources", []) or []):
        kind = str(source.get("kind") or "")
        if kind not in ("news_site", "rss_feed"):
            continue

        source_id = str(source.get("source_id"))
        discovery_only = bool(source.get("discovery_only", False))
        max_discovered_urls = int(source.get("max_discovered_urls", 0) or 0)

        discovery_method = (source.get("discovery") or {}).get("method")
        if kind == "rss_feed" or discovery_method == "rss_feed":
            urls = discover_links_rss_feed(session, source, http_cfg)
        else:
            urls = discover_links_section_pagination(session, source, http_cfg)

        # stable uniqueness, keep original order
        dedup: List[str] = []
        seen: Set[str] = set()
        for u in urls:
            if u not in seen:
                seen.add(u)
                dedup.append(u)

        if max_discovered_urls and len(dedup) > max_discovered_urls:
            dedup = dedup[:max_discovered_urls]

        discovered_by_source[source_id] = dedup

        if discovery_only:
            hub_url_pool.extend(dedup)

    # stable unique hub pool
    hub_seen: Set[str] = set()
    hub_unique: List[str] = []
    for u in hub_url_pool:
        if u not in hub_seen:
            hub_seen.add(u)
            hub_unique.append(u)

    # -------------------------------------------------------------------------
    # Step 2: SCRAPING
    #
    # For each real publisher source:
    #   candidate_urls = own_discovered + hub_discovered_that_match_this_source
    #   then fetch + parse + filter + save.
    # -------------------------------------------------------------------------
    all_rows: List[Dict] = []

    for source in (sources_cfg.get("sources", []) or []):
        kind = str(source.get("kind") or "")
        if kind not in ("news_site", "rss_feed"):
            continue

        source_id = str(source.get("source_id"))
        if bool(source.get("discovery_only", False)):
            # Never scrape articles for discovery-only hubs
            continue

        out_dir = raw_news_dir / source_id

        allow_patterns = source.get("allow", []) or []
        deny_substrings = source.get("deny", []) or []

        # Own URLs first (publisher’s canonical discovery)
        own_urls = discovered_by_source.get(source_id, []) or []

        # Then add hub URLs that match this publisher’s allow/deny rules
        routed_from_hubs: List[str] = []
        for u in hub_unique:
            if allowed_url(u, allow_patterns, deny_substrings):
                routed_from_hubs.append(u)

        # stable union: own first, then routed, no duplicates
        candidate: List[str] = []
        cand_seen: Set[str] = set()
        for u in own_urls + routed_from_hubs:
            if u not in cand_seen:
                cand_seen.add(u)
                candidate.append(u)

        # enforce per-source cap (deterministic)
        candidate = candidate[:max_per_source]

        for url in candidate:
            snap_path, snap_sha, fetched_at = fetch_snapshot(session, url, out_dir, http_cfg)
            if not snap_path:
                continue

            html = snap_path.read_text(encoding="utf-8", errors="ignore")
            parser = "xml" if html.lstrip().startswith("<?xml") else "lxml"
            soup = BeautifulSoup(html, parser)

            title = extract_title(soup)
            author = extract_author(soup)
            text = choose_main_text(soup)
            published_at, published_raw = extract_published_info(soup, text)

            text_norm = normalize_text(text)
            if len(text_norm) < min_chars:
                continue

            ts = _try_parse_dt(published_at)

            # optional strict rule: drop pages without a real publish date
            if drop_undated and ts is None:
                continue

            # date window filters
            if ts is not None:
                if min_year and int(ts.year) < min_year:
                    continue
                pub_d = ts.date()
                if start_date and pub_d < start_date:
                    continue
                if end_date and pub_d > end_date:
                    continue

            row = {
                "source_id": source_id,
                "url": url,
                "fetched_at_utc": fetched_at,
                "published_raw": published_raw,
                "published_at": (_ts_to_iso_utc(ts) if ts is not None else ""),
                "title": title,
                "author": author,
                "text": text_norm,
                "text_sha256": sha256_text((title or "") + "\n" + text_norm),
                "simhash64": simhash64((title or "") + " " + text_norm),
                "snapshot_path": str(snap_path).replace("/", "\\"),
                "snapshot_sha256": snap_sha,
            }
            all_rows.append(row)

    if not all_rows:
        print("No articles scraped.")
        return 0

    df = pd.DataFrame(all_rows)

    # stable article id derived from content hash
    df["article_id"] = df["text_sha256"].astype(str).str.slice(0, 16)

    # dedupe across all sources by content identity
    df = df.drop_duplicates(subset=["article_id"], keep="first").copy()

    out_csv = processed_dir / "news_articles.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved: {out_csv} | rows={len(df)} | sources={df['source_id'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
