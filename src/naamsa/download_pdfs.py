from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd
import requests

from src.utils.config import load_yaml, get_pipeline_paths


@dataclass
class PdfManifestRecord:
    downloaded_at_utc: str
    pdf_url: str
    saved_path: str | None
    sha256: str | None
    bytes: int | None
    http_status: int | None
    status: str  # "success" | "failed" | "skipped"
    error: str | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_streaming_download(url: str, out_path: Path, headers: Dict[str, str], timeout: int) -> Tuple[str, int, int]:
    import hashlib

    h = hashlib.sha256()
    total = 0

    with requests.get(url, headers=headers, timeout=timeout, stream=True) as r:
        status = r.status_code
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                h.update(chunk)
                total += len(chunk)

    return h.hexdigest(), total, status


def extract_upload_yyyy_mm(url: str) -> Tuple[Optional[int], Optional[int]]:
    # matches /wp-content/uploads/YYYY/MM/
    m = re.search(r"/uploads/(\d{4})/(\d{2})/", url)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def safe_filename(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", name)
    name = re.sub(r"\s+", "_", name)
    if not name.lower().endswith(".pdf"):
        name = name + ".pdf"
    return name[:200] if len(name) > 200 else name


def load_success_urls(manifest_path: Path) -> Set[str]:
    """Read existing manifest and return set of pdf_url that were already downloaded successfully."""
    success: Set[str] = set()
    if not manifest_path.exists():
        return success
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("status") == "success" and rec.get("pdf_url"):
                    success.add(str(rec["pdf_url"]))
            except Exception:
                continue
    return success


def append_manifest(manifest_path: Path, rec: PdfManifestRecord) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download NAAMSA PDFs listed in naamsa_links_to_fetch.csv.")
    ap.add_argument("--pipeline", required=True, help="Path to configs/pipeline.yaml")
    ap.add_argument("--in_links", required=True, help="CSV path: data/processed/naamsa_links_to_fetch.csv")
    args = ap.parse_args()

    pipeline_cfg = load_yaml(args.pipeline)
    paths = get_pipeline_paths(pipeline_cfg)
    raw_naamsa_dir = Path(paths["raw_naamsa_dir"])

    http_cfg = pipeline_cfg.get("http", {})
    headers = {"User-Agent": str(http_cfg.get("user_agent", "Academic research bot"))}
    timeout_seconds = int(http_cfg.get("timeout_seconds", 25))
    rate_limit_seconds = int(http_cfg.get("rate_limit_seconds", 3))
    max_retries = int(http_cfg.get("max_retries", 3))

    naamsa_cfg = pipeline_cfg.get("naamsa", {})
    download_limit = int(naamsa_cfg.get("download_limit", 40))

    links_path = Path(args.in_links)
    if not links_path.exists():
        raise FileNotFoundError(f"Links CSV not found: {links_path}")

    df = pd.read_csv(links_path)

    if "pdf_url" not in df.columns or "filename_guess" not in df.columns:
        raise ValueError("Input CSV must contain columns: pdf_url, filename_guess")

    yy_mm = df["pdf_url"].apply(extract_upload_yyyy_mm)
    df["upload_year"] = [t[0] for t in yy_mm]
    df["upload_month"] = [t[1] for t in yy_mm]

    df_sorted = df.sort_values(
        by=["upload_year", "upload_month", "pdf_url"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    df_pick = df_sorted.head(download_limit).copy()

    manifest_path = raw_naamsa_dir / "pdf_manifest.jsonl"
    success_urls = load_success_urls(manifest_path)

    downloaded = 0
    skipped = 0
    failed = 0

    for _, row in df_pick.iterrows():
        url = str(row["pdf_url"])

        # Incremental skip: already successfully downloaded in manifest
        if url in success_urls:
            skipped += 1
            continue

        guess = safe_filename(str(row.get("filename_guess", "naamsa.pdf")) or "naamsa.pdf")
        uy = row.get("upload_year")
        um = row.get("upload_month")

        prefix = ""
        if pd.notna(uy) and pd.notna(um):
            prefix = f"{int(uy):04d}_{int(um):02d}_"

        out_path = raw_naamsa_dir / f"{prefix}{guess}"

        # Also skip if file exists (extra safety)
        if out_path.exists() and out_path.stat().st_size > 0:
            append_manifest(
                manifest_path,
                PdfManifestRecord(
                    downloaded_at_utc=utc_now_iso(),
                    pdf_url=url,
                    saved_path=str(out_path),
                    sha256=None,
                    bytes=int(out_path.stat().st_size),
                    http_status=None,
                    status="skipped",
                    error="file_already_exists",
                ),
            )
            skipped += 1
            continue

        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                sha, nbytes, status_code = sha256_streaming_download(
                    url=url,
                    out_path=out_path,
                    headers=headers,
                    timeout=timeout_seconds,
                )
                append_manifest(
                    manifest_path,
                    PdfManifestRecord(
                        downloaded_at_utc=utc_now_iso(),
                        pdf_url=url,
                        saved_path=str(out_path),
                        sha256=sha,
                        bytes=nbytes,
                        http_status=status_code,
                        status="success",
                        error=None,
                    ),
                )
                downloaded += 1
                time.sleep(rate_limit_seconds)
                break
            except Exception as e:
                last_err = e
                time.sleep(2 * attempt)
        else:
            # Failed after retries
            append_manifest(
                manifest_path,
                PdfManifestRecord(
                    downloaded_at_utc=utc_now_iso(),
                    pdf_url=url,
                    saved_path=str(out_path),
                    sha256=None,
                    bytes=None,
                    http_status=None,
                    status="failed",
                    error=str(last_err),
                ),
            )
            failed += 1

    print(f"Downloaded: {downloaded} | Skipped: {skipped} | Failed: {failed} | Limit={download_limit}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
