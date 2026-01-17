from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from src.utils.config import load_yaml, get_source_by_id, get_pipeline_paths
from src.utils.http_client import fetch_text
from src.utils.snapshot import save_text_snapshot, append_manifest_jsonl


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]


def extract_links(html: str, base_url: str) -> List[Tuple[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    out: List[Tuple[str, str]] = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        abs_url = urljoin(base_url, href.strip())
        text = " ".join(a.get_text(" ", strip=True).split())
        out.append((abs_url, text))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Discover NAAMSA PDF links from press releases index page.")
    ap.add_argument("--pipeline", required=True, help="Path to configs/pipeline.yaml")
    ap.add_argument("--sources", required=True, help="Path to configs/sources.yaml")
    ap.add_argument("--out", required=True, help="Output CSV path for discovered PDF links")
    args = ap.parse_args()

    pipeline_cfg = load_yaml(args.pipeline)
    sources_cfg = load_yaml(args.sources)

    paths = get_pipeline_paths(pipeline_cfg)
    raw_naamsa_dir = paths["raw_naamsa_dir"]
    processed_dir = paths["processed_dir"]

    http_cfg = pipeline_cfg.get("http", {})
    headers = {"User-Agent": str(http_cfg.get("user_agent", "Academic research bot"))}
    timeout_seconds = int(http_cfg.get("timeout_seconds", 25))
    rate_limit_seconds = int(http_cfg.get("rate_limit_seconds", 3))
    max_retries = int(http_cfg.get("max_retries", 3))

    press_src = get_source_by_id(sources_cfg, "naamsa_press_releases")
    entry_points = press_src.get("discovery", {}).get("entry_points", [])
    if not entry_points:
        raise ValueError("naamsa_press_releases has no discovery.entry_points in sources.yaml")

    pdf_src = get_source_by_id(sources_cfg, "naamsa_pdfs")
    pdf_allow = pdf_src.get("discovery", {}).get("pdf_allow_regex", [])
    if not pdf_allow:
        raise ValueError("naamsa_pdfs has no discovery.pdf_allow_regex in sources.yaml")
    pdf_patterns = compile_patterns(pdf_allow)

    discovered_rows: List[Dict[str, Any]] = []

    manifest_path = Path(raw_naamsa_dir) / "manifest.jsonl"

    for idx, url in enumerate(entry_points, start=1):
        if idx > 1:
            time.sleep(rate_limit_seconds)

        html = fetch_text(url, headers=headers, timeout_seconds=timeout_seconds, max_retries=max_retries)

        snap = save_text_snapshot(
            text=html,
            out_dir=raw_naamsa_dir,
            prefix="press_releases_index",
            url=url,
            content_type="text/html",
        )
        append_manifest_jsonl(manifest_path, snap)

        links = extract_links(html, base_url=url)

        for link_url, link_text in links:
            # keep only PDFs that match the allow-regex patterns
            if any(p.search(link_url) for p in pdf_patterns):
                filename_guess = link_url.split("/")[-1].split("?")[0]
                discovered_rows.append(
                    {
                        "source_id": "naamsa_pdfs",
                        "page_url": url,
                        "pdf_url": link_url,
                        "link_text": link_text,
                        "discovered_at_utc": utc_now_iso(),
                        "filename_guess": filename_guess,
                        "index_snapshot_path": snap.path,
                        "index_snapshot_sha256": snap.sha256,
                    }
                )

    if not discovered_rows:
        print("No matching PDF links found. Check sources.yaml pdf_allow_regex patterns.")
        return 2

    df = pd.DataFrame(discovered_rows)

    # Deduplicate by pdf_url
    df = df.drop_duplicates(subset=["pdf_url"]).sort_values(["pdf_url"]).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Saved: {out_path} ({len(df)} unique PDF links)")
    print(f"Manifest appended: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


