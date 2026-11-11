"""Day 5-6: ABC-VED, expiry monitoring, FEFO demo and inventory KPIs.  Usage: python run_inventory.py"""
from pathlib import Path
import pandas as pd
from src.data_loader import load_data, load_movements
from src.eda import demand_summary
from src.abc_ved import abc_classification, abc_ved_matrix
from src.expiry import expiry_status, exposure_by_window, fefo_issue_plan
from src.kpis import product_kpis, store_kpis

pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40)
d = load_data(); mv = load_movements(); out = Path("reports")
AS_OF = d["consumption"].transaction_date.max()          # end of the data = "today" for the analysis
ds = demand_summary(d["consumption"], d["products"])
demand = ds.set_index("product_id").avg_daily_demand

abc = abc_classification(d["products"], demand)
exp = expiry_status(d["batches"], AS_OF, demand)
kpi = product_kpis(d["products"], d["batches"], mv, d["procurement"], ds, exp)
store = store_kpis(kpi, d["consumption"])

abc.to_csv(out / "abc_ved.csv", index=False); exp.to_csv(out / "expiry_status.csv", index=False); kpi.to_csv(out / "product_kpis.csv", index=False)
print(f"As-of date: {AS_OF.date()}  (all data synthetic)\n")
print("=== ABC-VED ===\n", abc[["product_id", "product_name", "annual_value_inr", "value_share", "cum_share_after", "abc", "ved", "abc_ved", "priority_label"]].round(3).to_string(index=False))
print("\nMatrix (count of products)\n", abc_ved_matrix(abc))
print("\n=== STOCK EXPIRING WITHIN WINDOWS ===\n", exposure_by_window(exp).round(0).to_string(index=False))
print("\n=== 6 BATCHES CLOSEST TO EXPIRY ===\n", exp.sort_values("days_to_expiry")
      [["batch_id", "product_id", "remaining_qty", "expiry_date", "days_to_expiry", "bucket", "at_risk_qty", "at_risk_value_inr"]].head(6).to_string(index=False))
print("\n=== PRODUCT KPIs ===\n", kpi[["product_id", "product_name", "stock_qty", "stock_value_inr", "avg_daily_demand", "days_of_cover", "on_order_qty",
      "lead_time_days", "excess_value_inr", "annualised_turnover", "stock_status"]].round(1).to_string(index=False))
print("\n=== STORE KPIs ===")
for k_, v in store.items(): print(f"  {k_:34s} {v:,.2f}")
print("\n=== FEFO DEMO: issue 100 units of P008 ===\n", fefo_issue_plan(d["batches"], "P008", 100, AS_OF).to_string(index=False))
