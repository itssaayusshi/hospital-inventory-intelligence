"""Day 7-8: forecast backtest, safety stock, reorder point, order recommendations.  Usage: python run_planning.py"""
from pathlib import Path
import pandas as pd
from src.data_loader import load_data
from src.eda import demand_summary
from src.forecasting import backtest, best_method, forecast, DEFAULT_METHOD
from src.expiry import expiry_status
from src.reorder import lead_time_stats, reorder_table

pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 90)
d = load_data(); out = Path("reports"); as_of = d["consumption"].transaction_date.max()
ds = demand_summary(d["consumption"], d["products"]).set_index("product_id")

bt = backtest(d["consumption"]); ranking = best_method(bt)
bt.to_csv(out / "forecast_backtest.csv", index=False)
print("=== FORECAST BACKTEST (7-day-ahead, rolling origin; lower WAPE is better) ===\n", ranking.round(3).to_string(index=False))
print(f"\nMethod used below: {DEFAULT_METHOD} (best: {ranking.method.iloc[0]})")

# forecast of the next 28 days -> mean daily demand used for the policy
mu = pd.Series({pid: forecast(g.sort_values("transaction_date").quantity_requested, g.sort_values("transaction_date").transaction_date, 28).mean()
                for pid, g in d["consumption"].groupby("product_id")})
lt = lead_time_stats(d["procurement"], d["products"])
exp = expiry_status(d["batches"], as_of, ds.avg_daily_demand)
rt = reorder_table(d["products"], mu, ds.std_daily_demand, lt, d["batches"], d["procurement"], as_of, exp)
rt.to_csv(out / "reorder_recommendations.csv", index=False)

print("\n=== LEAD TIMES USED ===\n", lt.round(2).to_string(index=False))
print("\n=== POLICY NUMBERS ===\n", rt[["product_id", "product_name", "criticality", "service_level", "mu", "sigma", "lead_time_mean", "lead_time_std",
      "safety_stock", "reorder_point", "order_up_to"]].round(2).to_string(index=False))
print(f"\n=== RECOMMENDATIONS as of {as_of.date()} (synthetic data) ===\n", rt[["product_id", "product_name", "on_hand", "on_order", "inventory_position",
      "reorder_point", "recommended_qty", "order_value_inr", "status"]].round(1).to_string(index=False))
print("\n=== WHY ===")
for r in rt.itertuples(): print(f"  {r.product_id} {r.product_name:24s} {r.status:34s} {r.reason}")
