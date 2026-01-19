import pandas as pd
import re
import yaml

df = pd.read_csv("data/processed/news_articles.csv")
text = (df["title"].fillna("") + "\n" + df["text"].fillna("")).str.lower()

sch = yaml.safe_load(open(r"configs/labeling_schema.yaml", "r", encoding="utf-8"))
ont = sch.get("ontology", {}) or {}

out = {}
for code, spec in ont.items():
    pos = spec.get("positive_patterns", []) or []
    c = 0
    for p in pos:
        try:
            c += int(text.str.contains(p, regex=True).sum())
        except re.error:
            # ignore invalid regex patterns
            pass
    out[code] = c

print(out)
