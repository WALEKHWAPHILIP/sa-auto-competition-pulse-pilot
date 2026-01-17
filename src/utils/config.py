from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {p}")
    return data


def get_source_by_id(sources_cfg: Dict[str, Any], source_id: str) -> Dict[str, Any]:
    sources = sources_cfg.get("sources", [])
    for s in sources:
        if s.get("source_id") == source_id:
            return s
    raise KeyError(f"source_id not found in configs/sources.yaml: {source_id}")


def get_pipeline_paths(pipeline_cfg: Dict[str, Any]) -> Dict[str, str]:
    storage = pipeline_cfg.get("storage", {})
    raw_news_dir = storage.get("raw_news_dir")
    raw_naamsa_dir = storage.get("raw_naamsa_dir")
    processed_dir = storage.get("processed_dir")
    if not raw_naamsa_dir or not processed_dir or not raw_news_dir:
        raise ValueError("pipeline.yaml missing storage.raw_news_dir/raw_naamsa_dir/processed_dir")
    return {
        "raw_news_dir": str(raw_news_dir),
        "raw_naamsa_dir": str(raw_naamsa_dir),
        "processed_dir": str(processed_dir),
    }