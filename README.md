powershell -NoProfile -Command @'

# SA Auto Competition Pulse (Pilot)



# -------------------------------

# README.md for GitHub / Markdown

# -------------------------------



# SA Auto Competition Pulse (Pilot)



\*\*Goal:\*\* Build a reproducible pipeline that converts South African automotive news into monthly competition-event time series and links them to monthly new-vehicle sales (NAAMSA), producing figures, tables, and a short research memo.



## Research question

Do competition-relevant news events predict or coincide with changes in monthly new-vehicle sales in South Africa?



## Data sources (v1)

- NAAMSA press releases index + monthly sales PDFs (downloaded and snapshotted)

- BusinessTech Motoring

- IOL Motoring Industry News



All source rules live in `configs/sources.yaml`. Global pipeline settings live in `configs/pipeline.yaml`.



## Event ontology (v1)

Multi-label event types defined in `configs/labeling\_schema.yaml`:

`ENTRY\_EXIT`, `LAUNCH\_PRODUCT`, `PRICE\_ACTION`, `RECALL\_QUALITY`, `CAPACITY\_SUPPLY`, `DEALER\_NETWORK`, `EV\_POLICY\_INFRA`.



## Repo structure

- `configs/` — configuration registry (sources, pipeline, labeling)

- `data/raw/` — immutable snapshots (HTML/PDF)

- `data/processed/` — analysis-ready datasets

- `reports/` — figures, tables, memo

- `src/` — pipeline code

- `tests/` — minimal tests



## Quickstart (environment)

1. Create a virtual environment (venv)

2. Install dependencies: `pip install -r requirements.txt`

3. Run pipeline scripts in `src/`

'@ | Set-Content -Encoding UTF8 README.md



