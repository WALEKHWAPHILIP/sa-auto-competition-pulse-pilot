from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _profile(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        missing = int(s.isna().sum())
        missing_pct = (missing / n * 100.0) if n else 0.0
        dtype = str(s.dtype)
        example = ""
        try:
            non_null = s.dropna()
            if len(non_null) > 0:
                example = str(non_null.iloc[0])
        except Exception:
            example = ""
        out.append(
            {
                "column": c,
                "dtype": dtype,
                "missing_count": missing,
                "missing_pct": round(missing_pct, 2),
                "example_value": example,
            }
        )
    return pd.DataFrame(out)


def main() -> int:
    root = _project_root()

    inputs = [
        ("news_articles.csv", root / "data" / "processed" / "news_articles.csv"),
        ("monthly_event_counts.csv", root / "data" / "processed" / "monthly_event_counts.csv"),
        ("naamsa_monthly_sales_headlines.csv", root / "data" / "processed" / "naamsa_monthly_sales_headlines.csv"),
        ("panel_monthly_events_sales.csv", root / "data" / "processed" / "panel_monthly_events_sales.csv"),
    ]

    out_path = root / "docs" / "data_dictionary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Data Dictionary (Auto-Generated Schema Snapshot)")
    lines.append("")
    lines.append("This file is generated from the *actual current* datasets in `data\\processed\\`.")
    lines.append("Regenerate anytime with:")
    lines.append("`python -m src.analysis.generate_data_dictionary`")
    lines.append("")

    for name, path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

        df = pd.read_csv(path)
        lines.append(f"## {name}")
        lines.append(f"- Path: `{path.as_posix()}`")
        lines.append(f"- Rows: {len(df)}")
        lines.append(f"- Columns: {len(df.columns)}")
        lines.append("")

        prof = _profile(df)

        # Write markdown table (deterministic ordering)
        prof = prof.sort_values("column").reset_index(drop=True)
        lines.append("| column | dtype | missing_count | missing_pct | example_value |")
        lines.append("|---|---:|---:|---:|---|")
        for _, r in prof.iterrows():
            col = str(r["column"]).replace("|", "\\|")
            dtype = str(r["dtype"]).replace("|", "\\|")
            mc = int(r["missing_count"])
            mp = float(r["missing_pct"])
            ex = str(r["example_value"]).replace("\n", " ").replace("\r", " ")
            ex = ex.replace("|", "\\|")
            if len(ex) > 80:
                ex = ex[:77] + "..."
            lines.append(f"| {col} | {dtype} | {mc} | {mp} | {ex} |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
