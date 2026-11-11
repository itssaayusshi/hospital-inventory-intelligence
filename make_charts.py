"""Day 2-3: build interactive HTML charts from the analysis tables. Needs plotly (pip install -r requirements.txt).
Usage: python make_charts.py   then open the files in reports/charts/ in your browser."""
from pathlib import Path
import plotly.express as px
from src.data_loader import load_data
from src.eda import demand_summary, demand_trend, day_of_week_pattern, vendor_performance, spend_trend

d = load_data(); charts = Path("reports") / "charts"; charts.mkdir(parents=True, exist_ok=True)
ds = demand_summary(d["consumption"], d["products"]); vp = vendor_performance(d["procurement"])
dow = day_of_week_pattern(d["consumption"]); wk = demand_trend(d["consumption"], "W")
# ---- charts (open the .html files in your browser)
wk_full = wk[~wk.is_partial].merge(d["products"][["product_id", "product_name"]], on="product_id")
px.line(wk_full, x="period_start", y="avg_daily_demand", color="product_name", title="Average daily demand by week (full weeks only)").write_html(charts / "weekly_demand.html")
px.bar(dow, x="weekday", y="index_vs_average", title="Demand by weekday (1.0 = average day)").write_html(charts / "weekday_pattern.html")
px.bar(ds.sort_values("cv"), x="product_name", y="cv", color="demand_pattern", title="Demand variability (CV = std / mean)").write_html(charts / "demand_variability.html")
px.bar(vp, x="vendor", y="on_time_rate", title="On-time delivery rate by vendor").write_html(charts / "vendor_on_time.html")
sp = spend_trend(d["procurement"], "vendor").melt(id_vars="month", var_name="vendor", value_name="spend_inr")
px.bar(sp, x="month", y="spend_inr", color="vendor", title="Procurement spend by month (INR, by order date)").write_html(charts / "spend_by_month.html")
print("Saved 5 charts in reports/charts/ - open them in your browser")
