# Hospital Inventory Intelligence & Waste Reduction System

Decision-support prototype built on **synthetic data only**. It is not connected to a hospital, and no figure
here is a real saving. Critical-stock policies would need hospital approval.

## Quick start (Mac)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_data.py   # rebuild the reconciled synthetic data (seed 42)
python run_audit.py               # data-quality + stock reconciliation report
pytest -q                         # run the tests
streamlit run app/streamlit_app.py
```

## Project layout
| Path | Purpose |
|---|---|
| `data/raw/` | Current reconciled synthetic CSVs (generated) |
| `data/raw_original_unreconciled/` | Original starter files, kept for reference and as generator input |
| `src/data_generator.py` | Day-by-day store-room simulation that produces consistent data |
| `src/validation.py` | Data-quality checks |
| `src/reconciliation.py` | Stock ledger checks |
| `src/analytics.py`, `app/` | Starter analytics and dashboard (to be rebuilt on later days) |
| `reports/` | Audit outputs |

See `docs/BEGINNER_GUIDE.md` for a step-by-step walkthrough.
