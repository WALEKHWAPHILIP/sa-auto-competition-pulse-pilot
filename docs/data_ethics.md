\# Data Ethics and Collection Policy



\## Purpose

This project collects public South African automotive news and public market-side information to measure competition-relevant events and relate them to market outcomes.



\## Sources

All sources are listed in `configs/sources.yaml`. Sources may change over time; any change must be documented in the weekly PI update.



\## Collection principles

\- Respect robots.txt and site terms when applicable.

\- Throttle requests and use retry/backoff.

\- Collect only what is necessary for research measurement (articles and public market data).

\- Avoid collecting personal data (user comments, private profiles, emails, phone numbers).

\- Store raw snapshots for auditability.



\## Storage policy

\- Raw data is immutable and stored under `data/raw/`.

\- Derived datasets are reproducible and stored under `data/interim/` and `data/processed/`.

\- Retrieval metadata and hashes are recorded to support audit and reproducibility.



\## Limitations

Scraping policies vary by site. The project aims to minimize load and store minimal content necessary for research. If a site disallows automated access, we will remove it or switch to permitted alternatives (e.g., RSS feeds).



