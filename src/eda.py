"""Day 2-3: demand and procurement analysis.

Every function takes plain DataFrames and returns a DataFrame, so each one
can be tested on its own and reused by the Streamlit app later.

Key idea: for DEMAND we use `quantity_requested` (what staff wanted), not
`quantity_issued` (what stock allowed). On a stockout day the issued number
is too low and would understate true demand.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def demand_summary(consumption: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """One row per product: average daily demand, variability and service level."""
    c = consumption.copy()
    c["short"] = c.quantity_requested - c.quantity_issued
    g = c.groupby("product_id")
    out = pd.DataFrame({
        "days_observed": g.transaction_date.nunique(),
        "total_requested": g.quantity_requested.sum(),
        "total_issued": g.quantity_issued.sum(),
        "avg_daily_demand": g.quantity_requested.mean(),
        "std_daily_demand": g.quantity_requested.std(),      # sample std (ddof=1)
        "p95_daily_demand": g.quantity_requested.quantile(0.95),
        "max_daily_demand": g.quantity_requested.max(),
        "stockout_days": g.short.apply(lambda s: (s > 0).sum()),
        "units_short": g.short.sum(),
    })
    # Coefficient of variation = std / mean: how bumpy demand is, comparable across products
    out["cv"] = out.std_daily_demand / out.avg_daily_demand
    out["demand_pattern"] = pd.cut(out.cv, [-np.inf, 0.25, 0.40, np.inf], labels=["Stable", "Moderate", "Volatile"]).astype(str)
    out["fill_rate"] = out.total_issued / out.total_requested
    out = out.reset_index().merge(products[["product_id", "product_name", "category", "criticality", "unit_cost_inr"]], on="product_id")
    out["annual_demand_value_inr"] = out.avg_daily_demand * 365 * out.unit_cost_inr
    return out


def demand_trend(consumption: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Demand per product per period. freq 'W' = weeks, 'M' = months.

    Partial periods at the edges would look like false drops, so we also give
    `avg_daily_demand` (total / days with data) which is safe to compare.
    """
    c = consumption.copy()
    c["period"] = c.transaction_date.dt.to_period(freq)
    t = (c.groupby(["product_id", "period"])
          .agg(total_requested=("quantity_requested", "sum"), days=("transaction_date", "nunique"))
          .reset_index())
    t["avg_daily_demand"] = t.total_requested / t.days
    full = 7 if freq == "W" else None
    t["is_partial"] = (t.days < full) if full else (t.days < t.period.dt.days_in_month)
    t["period_start"] = t.period.dt.start_time
    return t


def day_of_week_pattern(consumption: pd.DataFrame) -> pd.DataFrame:
    """Average total daily demand (all products) by weekday."""
    daily = consumption.groupby("transaction_date").quantity_requested.sum().rename("total").reset_index()
    daily["weekday"] = daily.transaction_date.dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    out = daily.groupby("weekday").total.mean().reindex(order).rename("avg_total_units").reset_index()
    out["index_vs_average"] = out.avg_total_units / daily.total.mean()
    return out


def department_mix(consumption: pd.DataFrame) -> pd.DataFrame:
    x = consumption.groupby("department").quantity_requested.sum().rename("units").reset_index()
    x["share"] = x.units / x.units.sum()
    return x.sort_values("units", ascending=False)


def add_delivery_metrics(procurement: pd.DataFrame) -> pd.DataFrame:
    """Add lead time and delay to delivered orders. In-transit orders have no actual delivery yet."""
    po = procurement.copy()
    done = po.status == "Delivered"
    po["actual_lead_time_days"] = np.where(done, (po.delivery_date - po.order_date).dt.days, np.nan)
    po["promised_lead_time_days"] = (po.promised_delivery_date - po.order_date).dt.days
    po["delay_days"] = np.where(done, (po.delivery_date - po.promised_delivery_date).dt.days, np.nan)
    po["on_time"] = np.where(done, po.delay_days <= 0, np.nan)
    return po


def vendor_performance(procurement: pd.DataFrame) -> pd.DataFrame:
    """Per-vendor spend, speed and reliability (delivered orders only for timing)."""
    po = add_delivery_metrics(procurement)
    done = po[po.status == "Delivered"]
    spend = po.groupby("vendor").agg(orders=("purchase_order_id", "count"),
                                     total_spend_inr=("total_invoice_value_inr", "sum"))
    perf = done.groupby("vendor").agg(
        delivered_orders=("purchase_order_id", "count"),
        avg_promised_lead_days=("promised_lead_time_days", "mean"),
        avg_actual_lead_days=("actual_lead_time_days", "mean"),
        lead_time_std=("actual_lead_time_days", "std"),
        avg_delay_days=("delay_days", "mean"),
        max_delay_days=("delay_days", "max"),
        on_time_rate=("on_time", "mean"),
    )
    out = spend.join(perf).reset_index()
    out["spend_share"] = out.total_spend_inr / out.total_spend_inr.sum()
    return out.sort_values("total_spend_inr", ascending=False)


def price_comparison(procurement: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Paid price vs master list price, per product."""
    g = procurement.groupby("product_id").unit_cost_inr
    out = pd.DataFrame({"orders": g.size(), "min_price": g.min(), "avg_price": g.mean(), "max_price": g.max()}).reset_index()
    out = out.merge(products[["product_id", "product_name", "unit_cost_inr"]].rename(columns={"unit_cost_inr": "list_price"}), on="product_id")
    out["avg_vs_list_pct"] = (out.avg_price / out.list_price - 1) * 100
    # quantity-weighted average price is what you really paid per unit
    w = procurement.groupby("product_id").apply(lambda x: x.total_invoice_value_inr.sum() / x.order_qty.sum(), include_groups=False)
    out["weighted_avg_price"] = out.product_id.map(w)
    return out


def spend_trend(procurement: pd.DataFrame, by: str = "vendor") -> pd.DataFrame:
    """Monthly procurement spend (by order date) split by vendor or product."""
    po = procurement.copy()
    po["month"] = po.order_date.dt.to_period("M").astype(str)
    return po.pivot_table(index="month", columns=by, values="total_invoice_value_inr", aggfunc="sum", fill_value=0).reset_index()
