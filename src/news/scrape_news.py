from __future__ import annotations

import argparse
import json
import re
import time
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.config import load_yaml, get_pipeline_paths


# ----------------------------
# Helpers
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def safe_filename(s: str, max_len: int = 160) -> str:
    s = s.strip()
    s = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", s)
    s = re.sub(r"\s+", "_", s)
    return s[:max_len] if len(s) > max_len else s


def normalize_text(t: str) -> str:
    if not t:
        return ""
    # light mojibake cleanup + whitespace normalization
    t = t.replace("ΓÇÖ", "'").replace("ΓÇô", "-")
    t = t.replace("ΓÇ£", '"').replace("ΓÇ¥", '"')
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def visible_text_from_node(node) -> str:
    # remove scripts/styles inside node
    for tag in node.find_all(["script", "style", "noscript"]):
        tag.decompose()
    txt = node.get_text(" ", strip=True)
    return normalize_text(txt)


def choose_main_text(soup: BeautifulSoup) -> str:
    # candidate containers (site-agnostic heuristics)
    candidates = []
    selectors = [
        "article",
        "div[itemprop='articleBody']",
        "div.entry-content",
        "div.article-body",
        "div#article-body",
        "div.article__content",
        "div.article-content",
        "section.article",
        "main",
    ]
    for sel in selectors:
        for node in soup.select(sel):
            txt = visible_text_from_node(node)
            if txt and len(txt) >= 200:
                candidates.append(txt)

    if candidates:
        # choose the longest candidate (usually the actual article)
        candidates.sort(key=len, reverse=True)
        return candidates[0]

    # fallback: body text
    if soup.body:
        return visible_text_from_node(soup.body)
    return ""


def extract_title(soup: BeautifulSoup) -> str:
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return normalize_text(og["content"])
    tw = soup.find("meta", attrs={"name": "twitter:title"})
    if tw and tw.get("content"):
        return normalize_text(tw["content"])
    if soup.title and soup.title.string:
        return normalize_text(soup.title.string)
    h1 = soup.find("h1")
    if h1:
        return normalize_text(h1.get_text(" ", strip=True))
    return ""


def extract_author(soup: BeautifulSoup) -> str:
    # common meta + page patterns
    ma = soup.find("meta", attrs={"name": "author"})
    if ma and ma.get("content"):
        return normalize_text(ma["content"])

    # try common byline classes
    for sel in ["span.author", "a.author", "div.author", "p.author", "span.byline", "div.byline"]:
        node = soup.select_one(sel)
        if node:
            txt = normalize_text(node.get_text(" ", strip=True))
            if 2 <= len(txt) <= 80:
                return txt

    # "By X"
    text = soup.get_text(" ", strip=True)
    m = re.search(r"\bBy\s+([A-Z][A-Za-z\.\- ]{2,50})\b", text)
    if m:
        return normalize_text(m.group(1))
    return ""


def extract_published(soup: BeautifulSoup) -> str:
    # prefer machine-readable times
    for attr in ["article:published_time", "og:published_time"]:
        m = soup.find("meta", attrs={"property": attr})
        if m and m.get("content"):
            return normalize_text(m["content"])

    m2 = soup.find("meta", attrs={"name": "pubdate"})
    if m2 and m2.get("content"):
        return normalize_text(m2["content"])

    t = soup.find("time")
    if t and t.get("datetime"):
        return normalize_text(t["datetime"])

    # fallback: none
    return ""


# ----------------------------
# Simhash (64-bit) + hamming
# ----------------------------

def simhash64(text: str) -> int:
    """
    Simple 64-bit simhash over tokens.
    Deterministic, no external deps.
    """
    text = normalize_text(text.lower())
    if not text:
        return 0

    # tokens: words >= 2 chars
    toks = re.findall(r"[a-z0-9]{2,}", text)
    if not toks:
        return 0

    # term frequency weights
    freq: Dict[str, int] = {}
    for tok in toks:
        freq[tok] = freq.get(tok, 0) + 1

    v = [0] * 64
    for tok, w in freq.items():
        h = hashlib.md5(tok.encode("utf-8")).hexdigest()  # stable
        x = int(h[:16], 16)  # 64-bit from first 16 hex chars
        for i in range(64):
            bit = (x >> i) & 1
            v[i] += w if bit else -w

    out = 0
    for i in range(64):
        if v[i] >= 0:
            out |= (1 << i)
    return out


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ----------------------------
# Snapshot manifest
# ----------------------------

@dataclass
class SnapshotRecord:
    fetched_at_utc: str
    source_id: str
    url: str
    path: str
    sha256: str
    bytes: int
    content_type: str
    http_status: int
    status: str   # success|failed
    error: str | None


def append_manifest(manifest_path: Path, rec: SnapshotRecord) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


# ----------------------------
# Discovery
# ----------------------------

def allowed_url(url: str, allow_patterns: List[str], deny_substrings: List[str]) -> bool:
    for d in deny_substrings:
        if d and d in url:
            return False
    for pat in allow_patterns:
        if re.search(pat, url):
            return True
    return False


def discover_links_section_pagination(
    session: requests.Session,
    source_cfg: dict,
    http_cfg: dict,
) -> List[str]:
    entry_points = source_cfg.get("discovery", {}).get("entry_points", []) or source_cfg.get("entry_points", []) or []
    pagination = source_cfg.get("pagination", {}) or {}
    page_pattern = pagination.get("page_pattern")
    start_page = int(pagination.get("start_page", 1))
    max_pages = int(pagination.get("max_pages", 40))

    allow_patterns = source_cfg.get("allow", []) or []
    deny_substrings = source_cfg.get("deny", []) or []

    headers = {"User-Agent": str(http_cfg.get("user_agent", "Academic research bot"))}
    timeout = int(http_cfg.get("timeout_seconds", 25))
    rate = int(http_cfg.get("rate_limit_seconds", 3))

    urls: List[str] = []
    seen: Set[str] = set()

    # if page_pattern exists, use it
    if page_pattern:
        for p in range(start_page, start_page + max_pages):
            page_url = page_pattern.format(page=p)
            try:
                r = session.get(page_url, headers=headers, timeout=timeout)
                html = r.text if r.ok else ""
            except Exception:
                html = ""
            if not html:
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

    # otherwise: crawl entry_points and follow "next" up to max_pages
    for entry in entry_points:
        current = entry
        for _ in range(max_pages):
            try:
                r = session.get(current, headers=headers, timeout=timeout)
                html = r.text if r.ok else ""
            except Exception:
                html = ""
            if not html:
                time.sleep(rate)
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

            # try find next link
            nxt = None
            link_rel_next = soup.find("link", attrs={"rel": "next"})
            if link_rel_next and link_rel_next.get("href"):
                nxt = urljoin(current, link_rel_next["href"])

            if not nxt:
                a_next = soup.find("a", string=re.compile(r"next", re.IGNORECASE))
                if a_next and a_next.get("href"):
                    nxt = urljoin(current, a_next["href"])

            if not nxt or nxt == current:
                time.sleep(rate)
                break

            current = nxt
            time.sleep(rate)

    return urls


# ----------------------------
# Fetch + extract
# ----------------------------

def fetch_snapshot(
    session: requests.Session,
    url: str,
    out_dir: Path,
    http_cfg: dict,
    source_id: str,
    manifest_path: Path,
) -> Tuple[Optional[Path], Optional[str], Optional[str]]:
    headers = {"User-Agent": str(http_cfg.get("user_agent", "Academic research bot"))}
    timeout = int(http_cfg.get("timeout_seconds", 25))
    rate = int(http_cfg.get("rate_limit_seconds", 3))
    max_retries = int(http_cfg.get("max_retries", 3))

    out_dir.mkdir(parents=True, exist_ok=True)
    fetched_at = utc_now_iso()

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, headers=headers, timeout=timeout)
            ct = r.headers.get("Content-Type", "")
            b = r.content or b""
            if not r.ok or not b:
                raise RuntimeError(f"HTTP {r.status_code}")

            sha = sha256_bytes(b)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            host = urlparse(url).netloc.replace(":", "_")
            fn = safe_filename(f"{ts}_{host}_{sha[:12]}.html")
            path = out_dir / fn
            path.write_bytes(b)

            append_manifest(
                manifest_path,
                SnapshotRecord(
                    fetched_at_utc=fetched_at,
                    source_id=source_id,
                    url=url,
                    path=str(path).replace("/", "\\"),
                    sha256=sha,
                    bytes=len(b),
                    content_type=ct,
                    http_status=r.status_code,
                    status="success",
                    error=None,
                ),
            )
            time.sleep(rate)
            return path, sha, fetched_at

        except Exception as e:
            last_err = e
            time.sleep(2 * attempt)

    append_manifest(
        manifest_path,
        SnapshotRecord(
            fetched_at_utc=fetched_at,
            source_id=source_id,
            url=url,
            path="",
            sha256="",
            bytes=0,
            content_type="",
            http_status=0,
            status="failed",
            error=str(last_err),
        ),
    )
    return None, None, None


def extract_article_from_html(html: str) -> Tuple[str, str, str, str]:
    soup = BeautifulSoup(html, "lxml")
    title = extract_title(soup)
    published = extract_published(soup)
    author = extract_author(soup)
    text = choose_main_text(soup)
    return title, published, author, text


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Scrape news sources into raw HTML snapshots + processed article table.")
    ap.add_argument("--pipeline", required=True, help="configs/pipeline.yaml")
    ap.add_argument("--sources", required=True, help="configs/sources.yaml")
    args = ap.parse_args()

    pipeline_cfg = load_yaml(args.pipeline)
    sources_cfg = load_yaml(args.sources)

    paths = get_pipeline_paths(pipeline_cfg)
    raw_news_dir = Path(paths["raw_news_dir"])
    processed_dir = Path(paths["processed_dir"])

    http_cfg = pipeline_cfg.get("http", {})
    run_cfg = pipeline_cfg.get("run", {})
    max_per_source = int(run_cfg.get("max_articles_per_source_per_run", 250))

    text_cfg = pipeline_cfg.get("text", {})
    min_chars = int(text_cfg.get("min_article_chars", 800))
    dedupe_cfg = (text_cfg.get("dedupe", {}) or {})
    method = str(dedupe_cfg.get("method", "simhash")).lower()
    ham_thr = int(dedupe_cfg.get("hamming_threshold", 3))

    session = requests.Session()

    all_rows: List[Dict] = []

    sources_list = sources_cfg.get("sources", []) or []
    for source in sources_list:
        if source.get("kind") != "news_site":
            continue

        source_id = str(source.get("source_id"))
        out_dir = raw_news_dir / source_id
        manifest_path = raw_news_dir / "manifest.jsonl"

        # discover article URLs
        urls = discover_links_section_pagination(session, source, http_cfg)
        # stable uniqueness
        dedup_urls = []
        seen = set()
        for u in urls:
            if u not in seen:
                seen.add(u)
                dedup_urls.append(u)

        # cap per run
        dedup_urls = dedup_urls[:max_per_source]

        for url in dedup_urls:
            snap_path, snap_sha, fetched_at = fetch_snapshot(
                session=session,
                url=url,
                out_dir=out_dir,
                http_cfg=http_cfg,
                source_id=source_id,
                manifest_path=manifest_path,
            )
            if not snap_path:
                continue

            html = snap_path.read_text(encoding="utf-8", errors="ignore")
            title, published, author, text = extract_article_from_html(html)

            text_norm = normalize_text(text)
            if len(text_norm) < min_chars:
                continue

            row = {
                "source_id": source_id,
                "url": url,
                "fetched_at_utc": fetched_at,
                "published_raw": published,
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
        print("No articles collected (all filtered or fetch failures).")
        return 2

    df = pd.DataFrame(all_rows)

    # Deduplication (simhash or exact)
    if method == "simhash":
        kept = []
        seen_hashes: List[int] = []
        for _, r in df.iterrows():
            h = int(r["simhash64"])
            is_dup = False
            for prev in seen_hashes:
                if hamming64(h, prev) <= ham_thr:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(r)
                seen_hashes.append(h)
        df = pd.DataFrame(kept)
    else:
        df = df.drop_duplicates(subset=["text_sha256"]).copy()

    # Save processed
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "news_articles.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Saved: {out_path} | articles={len(df)} | dedupe_method={method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
