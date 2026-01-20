# Data Ethics and Responsible Use

## Purpose and scope
This repository collects and processes publicly available South African automotive news and NAAMSA monthly sales releases in order to create reproducible, auditable research datasets linking *news-derived competition events* to *market outcomes*.

This project is research-oriented. It is not designed to:
- collect personal data,
- identify private individuals,
- bypass paywalls or access controls,
- or reproduce copyrighted articles beyond what is necessary for measurement and verification.

## Data categories used
1) **News articles (public web pages)**  
   - Stored fields: title, publication date, URL, source, and extracted text needed for event measurement.
   - The pipeline uses deduplication and audit metadata to reduce duplicate measurement.

2) **NAAMSA releases (public PDFs / press releases)**  
   - Stored fields: month, year, and industry total sales units (and associated parsing evidence fields).

3) **Derived research datasets (processed tables)**  
   - Aggregations (monthly event counts), merged panels, and audit tables/figures.

## Ethical collection rules (non-negotiable)
- Respect robots.txt and site terms where applicable.
- Use throttling and retry/backoff to avoid undue load on publishers.
- Store only the minimum content needed for research measurement and auditability.
- Do not collect personal information beyond what is already present in the article as part of publication metadata.
- Do not attempt to defeat paywalls, logins, or access restrictions.

## Data minimization and privacy
- This project focuses on competitive actions (launches, recalls, pricing actions, etc.) and market outcomes, not individuals.
- If an article contains incidental personal information, it is not used as a modeling target.
- Do not enrich with private datasets or attempt re-identification.

## Copyright and use of text
- News text is used for measurement (event labeling) and quality assurance.
- Do not redistribute large verbatim portions of articles.
- When presenting examples in memos, keep excerpts short and attribute the source URL.

## Known limitations and bias
- Source selection may bias what events are observed (coverage bias).
- Event labels are measurement proxies, not ground truth.
- Missing months in NAAMSA totals are documented and treated as a limitation.

## Contact
If a publisher or data owner has concerns, open a GitHub issue in this repository describing the specific content and requested change.
