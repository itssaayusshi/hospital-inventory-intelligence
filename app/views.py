"""The eight dashboard pages. Pages only display what src.pipeline computed."""
import pandas as pd
import streamlit as st
from src.eda import day_of_week_pattern, spend_trend
from src.expiry import exposure_by_window, fefo_issue_plan
from src.forecasting import METHODS, forecast
from src.abc_ved import abc_ved_matrix
from src.pipeline import (check_tables, expired_history, inventory_frame, open_orders, text_filter, unmet_demand, waste_scenarios)
from app import charts
from app.ui import csv_bytes, inr, pretty

SYNTH = "Synthetic data. Decision-support prototype, not a hospital system. Every rupee figure is calculated from this dataset."


def _names(m):
    return m.tables["products"].set_index("product_id").product_name


# ------------------------------------------------------------------ 1
def executive(m):
    st.header("Executive dashboard")
    st.caption(f"As of {m.as_of:%d %b %Y}. {SYNTH}")
    s = m.store
    r1 = st.columns(4)
    r1[0].metric("Stock value", inr(s["stock_value_inr"]))
    r1[1].metric("Expired in period", inr(s["expired_value_inr_in_period"]))
    r1[2].metric("Expiring within 90 days", inr(s["expiring_within_90d_value_inr"]))
    r1[3].metric("Fill rate", f"{s['fill_rate']:.1%}")
    r2 = st.columns(4)
    r2[0].metric("Stockout days", f"{s['stockout_days']}")
    r2[1].metric("Inventory turnover (per year)", f"{s['store_turnover']:.1f}x")
    r2[2].metric("High-priority alerts", int((m.alerts.severity == "High").sum()) if len(m.alerts) else 0)
    r2[3].metric("Overdue purchase orders", int((open_orders(m).days_overdue > 0).sum()))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("What needs attention")
        if m.alerts.empty:
            st.success("No alerts. Stock is within policy.")
        else:
            st.dataframe(pretty(m.alerts, {"product_id": "ID"}, icon_cols=["severity", "issue"]))
            st.download_button("Download alerts (CSV)", csv_bytes(m.alerts), "alerts.csv", "text/csv")
    with right:
        st.plotly_chart(charts.status_pie(m.reorder))
    a, b = st.columns(2)
    a.plotly_chart(charts.stock_value_bar(m.kpi))
    b.plotly_chart(charts.inventory_timeline(m.inv_timeline))
    with st.expander("Data quality and reconciliation checks"):
        rep = check_tables(m.tables)
        st.dataframe(rep[rep["count"] > 0] if (rep["count"] > 0).any() else rep)
        st.caption("INFO rows are normal observations. ERROR rows would block the analysis.")


# ------------------------------------------------------------------ 2
def inventory(m):
    st.header("Inventory overview")
    st.caption(SYNTH)
    inv = inventory_frame(m)
    f = st.columns(4)
    cats = f[0].multiselect("Category", sorted(inv.category.unique()), default=sorted(inv.category.unique()))
    crit = f[1].multiselect("Criticality", ["Vital", "Essential", "Desirable"], default=["Vital", "Essential", "Desirable"])
    stat = f[2].multiselect("Status", sorted(inv.status.unique()), default=sorted(inv.status.unique()))
    q = f[3].text_input("Search", "")
    x = text_filter(inv[inv.category.isin(cats) & inv.criticality.isin(crit) & inv.status.isin(stat)], q)
    k = st.columns(3)
    k[0].metric("Products shown", len(x))
    k[1].metric("Stock value (shown)", inr(x.stock_value_inr.sum()))
    k[2].metric("Units on hand (shown)", f"{int(x.on_hand.sum()):,}")
    st.dataframe(pretty(x.drop(columns=["lead_time_mean"]), {"days_cover_on_hand": "days_cover", "stock_value_inr": "stock_value_inr"}, icon_cols=["status"]))
    st.plotly_chart(charts.cover_chart(x if len(x) else inv))
    st.subheader("Batches holding stock (FEFO order)")
    b = m.expiry.merge(_names(m).rename("product_name"), left_on="product_id", right_index=True)
    st.dataframe(pretty(b[["batch_id", "product_id", "product_name", "remaining_qty", "received_date", "expiry_date", "days_to_expiry", "bucket", "stock_value_inr", "fefo_rank"]]))


# ------------------------------------------------------------------ 3
def expiry(m):
    st.header("Expiry monitoring")
    st.caption(SYNTH)
    ex = exposure_by_window(m.expiry)
    c = st.columns(3)
    for col, row in zip(c, ex.itertuples()):
        col.metric(f"Expiring within {row.window_days} days", inr(row.value_inr), f"{row.units:,} units, {row.batches} batches", delta_color="off")
    st.plotly_chart(charts.expiry_buckets(m.expiry))
    st.subheader("FEFO-aware risk and batch list")
    window = st.selectbox("Show batches expiring within (days)", [30, 60, 90, 180, 400], index=2)
    b = m.expiry[m.expiry.days_to_expiry <= window]
    st.dataframe(pretty(b[["batch_id", "product_id", "remaining_qty", "expiry_date", "days_to_expiry", "bucket", "stock_value_inr", "at_risk_qty", "at_risk_value_inr"]]))
    st.download_button("Download expiry list (CSV)", csv_bytes(m.expiry), "expiry_status.csv", "text/csv")
    st.caption("At-risk = units demand is not expected to use before the batch expires, assuming no further deliveries.")
    st.subheader("FEFO pick planner")
    names = _names(m)
    pid = st.selectbox("Product", list(names.index), format_func=lambda p: f"{p} - {names[p]}")
    qty = st.number_input("Quantity to issue", min_value=1, value=100)
    plan = fefo_issue_plan(m.tables["batches"], pid, int(qty), m.as_of)
    if plan.empty:
        st.warning("No usable stock for this product.")
    else:
        st.dataframe(pretty(plan))
        if plan.shortfall.iloc[0] > 0:
            st.warning(f"Short by {int(plan.shortfall.iloc[0])} units.")
    st.subheader("Past expiry write-offs")
    w = expired_history(m)
    st.metric("Total written off", inr(w.value_inr.sum()))
    st.dataframe(pretty(w) if len(w) else w)


# ------------------------------------------------------------------ 4
def demand(m):
    st.header("Demand forecasting")
    st.caption(SYNTH)
    names = _names(m); cons = m.tables["consumption"]
    c = st.columns(3)
    pid = c[0].selectbox("Product", list(names.index), format_func=lambda p: f"{p} - {names[p]}")
    method = c[1].selectbox("Method", list(METHODS), index=list(METHODS).index(m.params.method))
    horizon = c[2].slider("Forecast days", 7, 28, 14)
    g = cons[cons.product_id == pid].sort_values("transaction_date")
    fc = forecast(g.quantity_requested, g.transaction_date, horizon, method)
    row = m.demand[m.demand.product_id == pid].iloc[0]
    k = st.columns(4)
    k[0].metric("Average daily demand", f"{row.avg_daily_demand:.1f}")
    k[1].metric("Variability (CV)", f"{row.cv:.2f}", row.demand_pattern, delta_color="off")
    k[2].metric(f"Forecast next {horizon} days", f"{fc.sum():,.0f} units")
    k[3].metric("Stockout days", int(row.stockout_days))
    recent = g.tail(90)
    st.plotly_chart(charts.demand_forecast(recent.transaction_date, recent.quantity_requested, fc, f"{names[pid]}: last 90 days and forecast ({method})"))
    a, b = st.columns(2)
    a.plotly_chart(charts.weekday_bar(day_of_week_pattern(cons)))
    with b:
        st.subheader("Which method is most accurate?")
        st.dataframe(pretty(m.ranking, {"avg_wape": "avg_error_(WAPE)", "avg_bias": "avg_bias"}, decimals=3))
        st.caption("Rolling 7-day-ahead backtest using only past data. Lower error is better.")
    st.subheader("Demand summary (all products)")
    st.dataframe(pretty(m.demand[["product_id", "product_name", "criticality", "avg_daily_demand", "std_daily_demand", "cv", "demand_pattern", "p95_daily_demand", "stockout_days", "fill_rate"]], decimals=2))


# ------------------------------------------------------------------ 5
def reorder(m):
    st.header("Reorder recommendations")
    st.caption(f"{SYNTH} Service levels and review period are assumptions set in the sidebar.")
    r = m.reorder
    k = st.columns(4)
    k[0].metric("Products to order now", int((r.recommended_qty > 0).sum()))
    k[1].metric("Total order value", inr(r.order_value_inr.sum()))
    k[2].metric("Stocked out", int(r.status.str.startswith("Stocked out").sum()))
    k[3].metric("Relying on incoming orders", int(r.status.str.startswith("Watch").sum()))
    stat = st.multiselect("Status", sorted(r.status.unique()), default=sorted(r.status.unique()))
    q = st.text_input("Search products", "")
    x = text_filter(r[r.status.isin(stat)], q)
    cols = ["product_id", "product_name", "abc_ved", "on_hand", "on_order", "inventory_position", "safety_stock", "reorder_point", "order_up_to",
            "recommended_qty", "order_value_inr", "status", "reason"]
    st.dataframe(pretty(x[cols], icon_cols=["status"]))
    st.download_button("Download recommendations (CSV)", csv_bytes(r), "reorder_recommendations.csv", "text/csv")
    with st.expander("How these numbers are calculated"):
        st.markdown("- **Safety stock** = z x sqrt(lead time x demand variance + demand squared x lead-time variance)\n"
                    "- **Reorder point** = average daily demand x lead time + safety stock\n"
                    "- **Inventory position** = usable stock on hand + orders on the way\n"
                    "- **Order quantity** = order-up-to level minus position, rounded up to a pack of 10, only when position is at or below the reorder point")
    st.subheader("Policy inputs per product")
    st.dataframe(pretty(r[["product_id", "criticality", "service_level", "mu", "sigma", "lead_time_mean", "lead_time_std", "lead_time_source"]], {"mu": "forecast_daily_demand", "sigma": "demand_std"}, decimals=2))


# ------------------------------------------------------------------ 6
def abc_ved(m):
    st.header("ABC-VED analysis")
    st.caption(SYNTH)
    a = m.abc
    st.plotly_chart(charts.pareto(a))
    c = st.columns(2)
    with c[0]:
        st.subheader("Matrix (number of products)")
        st.dataframe(abc_ved_matrix(a))
    with c[1]:
        st.subheader("Priority groups")
        st.dataframe(a.groupby("priority_label").product_id.apply(", ".join).rename("products").reset_index())
    st.dataframe(pretty(a[["product_id", "product_name", "annual_value_inr", "value_share", "cum_share_after", "abc", "ved", "abc_ved", "priority_label"]], decimals=3))
    with st.expander("Methodology and how to read it"):
        st.markdown("**ABC** ranks products by annual consumption value (daily demand x 365 x unit cost): A = first 70% of value, B = next 20%, C = the rest.\n\n"
                    "**VED** is clinical criticality: Vital, Essential, Desirable.\n\n"
                    "**Combined**: AV, BV and AE get tight control (never run out, review often). CV, BE and AD get regular review. CE, BD and CD can use simple bulk rules. "
                    "This grouping is a common convention, not a rule of nature.")


# ------------------------------------------------------------------ 7
def waste(m):
    st.header("Waste and cost analytics")
    st.caption(SYNTH)
    s = m.store; um = unmet_demand(m)
    k = st.columns(4)
    k[0].metric("Expired in period", inr(s["expired_value_inr_in_period"]))
    k[1].metric("Excess stock now", inr(s["excess_stock_value_inr"]))
    k[2].metric("Unmet demand (value)", inr(um.unmet_demand_value_inr.sum()))
    k[3].metric("Inventory turnover", f"{s['store_turnover']:.1f}x per year")
    a, b = st.columns(2)
    if (m.kpi.expired_value_inr > 0).any():
        a.plotly_chart(charts.expired_bar(m.kpi))
    else:
        a.info("No expiry write-offs in this period.")
    b.plotly_chart(charts.stock_value_bar(m.kpi))
    st.subheader("By product")
    t = m.kpi[["product_id", "product_name", "expired_value_inr", "excess_value_inr", "at_risk_expiry_value_inr", "annualised_turnover", "days_inventory_outstanding"]] \
        .merge(um[["product_id", "units_short", "unmet_demand_value_inr"]], on="product_id")
    st.dataframe(pretty(t))
    st.download_button("Download waste table (CSV)", csv_bytes(t), "waste_by_product.csv", "text/csv")
    st.subheader("What-if scenarios (SIMULATED)")
    st.warning("These are simulated figures built from user-chosen assumptions on synthetic data. They are not real savings.")
    c = st.columns(3)
    wc = c[0].slider("Reduction in expiry waste (%)", 0, 100, 25) / 100
    ic = c[1].slider("Reduction in average inventory (%)", 0, 50, 10) / 100
    hr = c[2].slider("Annual holding cost (% of inventory value)", 5, 40, 20) / 100
    st.dataframe(pretty(waste_scenarios(m, wc, ic, hr), decimals=0))


# ------------------------------------------------------------------ 8
def procurement(m):
    st.header("Procurement and vendor performance")
    st.caption(f"{SYNTH} Each vendor has few orders, so treat on-time rates as illustration.")
    v = m.vendors; po = m.tables["procurement"]
    k = st.columns(3)
    k[0].metric("Total spend", inr(po.total_invoice_value_inr.sum()))
    k[1].metric("Purchase orders", len(po))
    k[2].metric("Open orders", int((po.status == "In transit").sum()))
    st.dataframe(pretty(v, decimals=2))
    a, b = st.columns(2)
    a.plotly_chart(charts.vendor_on_time(v))
    b.plotly_chart(charts.vendor_lead(v))
    st.plotly_chart(charts.spend_month(spend_trend(po, "vendor")))
    st.subheader("Price paid vs list price")
    st.dataframe(pretty(m.prices, decimals=3))
    st.subheader("Open purchase orders")
    o = open_orders(m)
    st.dataframe(pretty(o) if len(o) else o)
    if len(o) and (o.days_overdue > 0).any():
        st.warning(f"{int((o.days_overdue > 0).sum())} order(s) are past their promised date. Follow up with the vendor.")


PAGES = {"Executive Dashboard": executive, "Inventory Overview": inventory, "Expiry Monitoring": expiry, "Demand Forecasting": demand,
         "Reorder Recommendations": reorder, "ABC-VED Analysis": abc_ved, "Waste & Cost Analytics": waste, "Procurement & Vendors": procurement}
