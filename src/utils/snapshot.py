from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .hashing import sha256_bytes


@dataclass
class SnapshotRecord:
    fetched_at_utc: str
    url: str
    path: str
    sha256: str
    bytes: int
    content_type: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save_text_snapshot(
    *,
    text: str,
    out_dir: str | Path,
    prefix: str,
    url: str,
    content_type: str = "text/html",
) -> SnapshotRecord:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{ts}.html"
    path = out / filename

    data = text.encode("utf-8", errors="replace")
    path.write_bytes(data)

    rec = SnapshotRecord(
        fetched_at_utc=utc_now_iso(),
        url=url,
        path=str(path).replace("\\\\", "/"),
        sha256=sha256_bytes(data),
        bytes=len(data),
        content_type=content_type,
    )
    return rec


def append_manifest_jsonl(manifest_path: str | Path, record: SnapshotRecord) -> None:
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(record), ensure_ascii=False)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def extract_text_plus_tables(pdf_path: str, max_pages: int = 2):
    """
    Extract text (and best-effort tables) from a PDF.
    Returns: (text, tables)
    tables is a list (possibly empty). This keeps the API stable for NAAMSA parsing.
    """
    text_parts = []
    tables = []

    # 1) Try pdfplumber first (best for text + tables)
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages if not max_pages else pdf.pages[:max_pages]
            for p in pages:
                try:
                    text_parts.append(p.extract_text() or "")
                except Exception:
                    text_parts.append("")
                try:
                    for t in (p.extract_tables() or []):
                        tables.append(t)
                except Exception:
                    pass
        return "\n".join(text_parts), tables
    except Exception:
        pass

    # 2) Fallback: pypdf (text only)
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(pdf_path)
        n = len(reader.pages) if not max_pages else min(len(reader.pages), max_pages)
        for i in range(n):
            try:
                text_parts.append(reader.pages[i].extract_text() or "")
            except Exception:
                text_parts.append("")
        return "\n".join(text_parts), tables
    except Exception as e:
        raise RuntimeError(f"PDF text extraction failed for {pdf_path}: {e}")


