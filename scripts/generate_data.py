"""Regenerate the reconciled synthetic dataset.   Run:  python scripts/generate_data.py"""
import shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.data_generator import generate, SEED

raw = ROOT / "data" / "raw"
backup = ROOT / "data" / "raw_original_unreconciled"
if not backup.exists():                       # keep the original files, never overwrite them
    shutil.copytree(raw, backup)
    print(f"Backed up original files to {backup.relative_to(ROOT)}")
products = pd.read_csv(backup / "product_master.csv")
demand = pd.read_csv(backup / "consumption_transactions.csv")
out = generate(demand, products, seed=SEED)
for name, fname in [("batches", "inventory_batches.csv"), ("movements", "stock_movements.csv"),
                    ("procurement", "procurement_transactions.csv"), ("consumption", "consumption_transactions.csv")]:
    out[name].to_csv(raw / fname, index=False)
    print(f"wrote data/raw/{fname}: {len(out[name])} rows")
