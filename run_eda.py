"""Day 2-3: run the demand + procurement analysis. Prints tables and saves CSVs to reports/.  Charts: python make_charts.py"""
from pathlib import Path
import pandas as pd
from src.data_loader import load_data
from src.eda import (demand_summary, demand_trend, day_of_week_pattern, department_mix,
                     vendor_performance, price_comparison, spend_trend)

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
d = load_data(); out = Path("reports")

ds = demand_summary(d["consumption"], d["products"])
vp = vendor_performance(d["procurement"])
pc = price_comparison(d["procurement"], d["products"])
dow = day_of_week_pattern(d["consumption"])
wk = demand_trend(d["consumption"], "W"); mo = demand_trend(d["consumption"], "M")
for name, df in [("demand_summary", ds), ("vendor_performance", vp), ("price_comparison", pc), ("day_of_week", dow),
                 ("department_mix", department_mix(d["consumption"])), ("weekly_trend", wk), ("monthly_trend", mo)]:
    df.to_csv(out / f"{name}.csv", index=False)

print("=== DEMAND SUMMARY ===")
print(ds[["product_id", "product_name", "criticality", "avg_daily_demand", "std_daily_demand", "cv", "demand_pattern", "stockout_days", "fill_rate"]].round(2).to_string(index=False))
print("\n=== VENDOR PERFORMANCE ===")
print(vp.round(2).to_string(index=False))
print("\n=== DAY OF WEEK ===")
print(dow.round(2).to_string(index=False))
print("\n=== PRICE PAID vs LIST ===")
print(pc[["product_id", "product_name", "orders", "list_price", "weighted_avg_price", "avg_vs_list_pct"]].round(3).to_string(index=False))

print("\nSaved CSV tables in reports/. Next: python make_charts.py")
