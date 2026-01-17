from __future__ import annotations

import time
from typing import Dict, Optional

import requests


def fetch_text(url: str, headers: Dict[str, str], timeout_seconds: int, max_retries: int) -> str:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout_seconds)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or r.encoding
            return r.text
        except Exception as e:
            last_err = e
            # linear backoff: 2, 4, 6 ...
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to fetch after {max_retries} retries: {url}. Last error: {last_err}")

