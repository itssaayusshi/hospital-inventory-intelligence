"""Run the data audit + reconciliation and save reports.   Usage: python run_audit.py"""
from pathlib import Path
import pandas as pd
from src.data_loader import load_data, load_movements
from src.validation import validate
from src.reconciliation import reconcile_movements

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
d = load_data(); mv = load_movements()
rep = validate(d["products"], d["batches"], d["consumption"], d["procurement"])
prod_rec, batch_rec = reconcile_movements(d["products"], d["batches"], mv)
out = Path("reports"); out.mkdir(exist_ok=True)
rep.to_csv(out / "data_quality_report.csv", index=False)
prod_rec.to_csv(out / "reconciliation_by_product.csv", index=False)
batch_rec.to_csv(out / "reconciliation_by_batch.csv", index=False)
print("=== DATA QUALITY (non-zero findings) ===\n", rep[rep["count"] > 0].to_string(index=False))
print("\n=== STOCK RECONCILIATION BY PRODUCT ===\n", prod_rec.drop(columns="product_name").to_string(index=False))
print(f"\nBatches reconciled: {(batch_rec.status_check == 'Reconciled').sum()} of {len(batch_rec)}")
print(f"Products reconciled: {(prod_rec.status_check == 'Reconciled').sum()} of {len(prod_rec)}")
