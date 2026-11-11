"""Plotly figures. Each function takes tables and returns a figure; no calculations beyond layout."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BUCKET_ORDER = ["Expired", "0-30 days", "31-60 days", "61-90 days", ">90 days"]


def stock_value_bar(kpi):
    d = kpi.sort_values("stock_value_inr")
    return px.bar(d, x="stock_value_inr", y="product_name", orientation="h", title="Stock value by product (INR)",
                  labels={"stock_value_inr": "Stock value (INR)", "product_name": ""})


def status_pie(reorder):
    d = reorder.groupby("status").size().rename("products").reset_index()
    return px.pie(d, names="status", values="products", hole=0.5, title="Products by status")


def inventory_timeline(timeline):
    d = timeline.sum(axis=1).rename("value").rename_axis("date").reset_index()
    return px.line(d, x="date", y="value", title="Total inventory value over time (INR)", labels={"value": "INR", "date": ""})


def cover_chart(inv):
    d = inv.sort_values("days_cover_on_hand")
    fig = px.bar(d, x="product_name", y="days_cover_on_hand", title="Days of cover on hand vs supplier lead time",
                 labels={"days_cover_on_hand": "Days of cover", "product_name": ""})
    fig.add_trace(go.Scatter(x=d.product_name, y=d.lead_time_mean, mode="markers", name="Lead time (days)", marker=dict(size=11, symbol="diamond")))
    return fig


def expiry_buckets(exp):
    d = exp.groupby("bucket").stock_value_inr.sum().reindex(BUCKET_ORDER).fillna(0).rename("value").reset_index()
    return px.bar(d, x="bucket", y="value", title="Stock value by time to expiry (INR)", labels={"value": "INR", "bucket": ""})


def demand_forecast(hist_dates, hist_y, fc, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(hist_dates), y=list(hist_y), mode="lines", name="Actual daily demand"))
    fig.add_trace(go.Scatter(x=list(fc.index), y=list(fc.values), mode="lines+markers", name="Forecast"))
    fig.update_layout(title=title, yaxis_title="Units per day")
    return fig


def weekday_bar(dow):
    return px.bar(dow, x="weekday", y="index_vs_average", title="Demand by weekday (1.0 = average day)", labels={"index_vs_average": "Index", "weekday": ""})


def pareto(abc):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=abc.product_name, y=abc.annual_value_inr, name="Annual value (INR)"))
    fig.add_trace(go.Scatter(x=abc.product_name, y=abc.cum_share_after, name="Cumulative share", yaxis="y2", mode="lines+markers"))
    fig.update_layout(title="Pareto: annual consumption value", yaxis2=dict(overlaying="y", side="right", range=[0, 1.05], tickformat=".0%"))
    return fig


def expired_bar(kpi):
    d = kpi[kpi.expired_value_inr > 0].sort_values("expired_value_inr")
    return px.bar(d, x="expired_value_inr", y="product_name", orientation="h", title="Expired stock value by product (INR)",
                  labels={"expired_value_inr": "INR", "product_name": ""})


def vendor_on_time(vendors):
    return px.bar(vendors, x="vendor", y="on_time_rate", title="On-time delivery rate by vendor", labels={"on_time_rate": "On time", "vendor": ""})


def vendor_lead(vendors):
    d = vendors.melt(id_vars="vendor", value_vars=["avg_promised_lead_days", "avg_actual_lead_days"], var_name="type", value_name="days")
    return px.bar(d, x="vendor", y="days", color="type", barmode="group", title="Promised vs actual lead time (days)")


def spend_month(spend_wide):
    d = spend_wide.melt(id_vars="month", var_name="vendor", value_name="spend_inr")
    return px.bar(d, x="month", y="spend_inr", color="vendor", title="Procurement spend by month (INR, by order date)")
