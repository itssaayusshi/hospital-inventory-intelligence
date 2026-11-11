"""Inventory KPIs per product and for the whole store."""
from __future__ import annotations
import numpy as np
import pandas as pd

EXCESS_COVER_DAYS = 45  # stock above this many days of demand counts as excess (adjustable assumption)


def inventory_value_timeline(movements: pd.DataFrame, batches: pd.DataFrame) -> pd.DataFrame:
    """End-of-day stock value (INR) per product: running sum of movement quantity x batch cost."""
    mv = movements.merge(batches[["batch_id", "unit_cost_inr"]], on="batch_id")
    mv["value"] = mv.quantity * mv.unit_cost_inr
    daily = mv.groupby(["product_id", "movement_date"]).value.sum().unstack(0).fillna(0)
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()), fill_value=0)
    return daily.cumsum()


def inventory_turnover(movements: pd.DataFrame, batches: pd.DataFrame) -> pd.DataFrame:
    """Annualised turnover = (value issued per year) / (average inventory value). Higher = stock moves faster."""
    timeline = inventory_value_timeline(movements, batches)
    mv = movements.merge(batches[["batch_id", "unit_cost_inr"]], on="batch_id")
    issued = mv[mv.movement_type == "ISSUE"].assign(v=lambda d: -d.quantity * d.unit_cost_inr).groupby("product_id").v.sum()
    days = len(timeline)
    out = pd.DataFrame({"issued_value_inr": issued, "avg_inventory_value_inr": timeline.mean()})
    out["annualised_turnover"] = (out.issued_value_inr * 365 / days) / out.avg_inventory_value_inr.replace(0, np.nan)
    out["days_inventory_outstanding"] = 365 / out.annualised_turnover
    return out.reset_index()


def product_kpis(products, batches, movements, procurement, demand_summary, expiry, excess_days=EXCESS_COVER_DAYS):
    """One row per product combining stock, cover, excess, expiry risk and incoming orders."""
    stock = batches.groupby("product_id").apply(lambda g: pd.Series({
        "stock_qty": g.remaining_qty.sum(), "stock_value_inr": (g.remaining_qty * g.unit_cost_inr).sum()}), include_groups=False)
    on_order = procurement[procurement.status == "In transit"].groupby("product_id").order_qty.sum().rename("on_order_qty")
    risk = expiry.groupby("product_id").agg(at_risk_expiry_units=("at_risk_qty", "sum"), at_risk_expiry_value_inr=("at_risk_value_inr", "sum"))
    near = expiry[expiry.days_to_expiry <= 90].groupby("product_id").stock_value_inr.sum().rename("expiring_90d_value_inr")
    wasted = movements[movements.movement_type == "EXPIRY_WRITE_OFF"].merge(batches[["batch_id", "unit_cost_inr"]], on="batch_id")
    wasted = (-wasted.quantity * wasted.unit_cost_inr).groupby(wasted.product_id).sum().rename("expired_value_inr")

    k = products[["product_id", "product_name", "criticality", "unit_cost_inr", "lead_time_days"]].set_index("product_id")
    k = k.join(stock).join(on_order).join(risk).join(near).join(wasted)
    k = k.join(demand_summary.set_index("product_id")[["avg_daily_demand", "cv", "stockout_days", "fill_rate"]])
    k[["on_order_qty", "at_risk_expiry_units", "at_risk_expiry_value_inr", "expiring_90d_value_inr", "expired_value_inr"]] = \
        k[["on_order_qty", "at_risk_expiry_units", "at_risk_expiry_value_inr", "expiring_90d_value_inr", "expired_value_inr"]].fillna(0)
    k["days_of_cover"] = k.stock_qty / k.avg_daily_demand
    k["days_of_cover_incl_on_order"] = (k.stock_qty + k.on_order_qty) / k.avg_daily_demand
    k["excess_units"] = (k.stock_qty - excess_days * k.avg_daily_demand).clip(lower=0).round()
    k["excess_value_inr"] = k.excess_units * k.unit_cost_inr
    k["stock_status"] = "OK"
    k.loc[k.days_of_cover > excess_days, "stock_status"] = "Overstocked"
    k.loc[k.days_of_cover_incl_on_order < k.lead_time_days, "stock_status"] = "Stockout risk"
    k.loc[k.stock_qty == 0, "stock_status"] = "Out of stock"
    return k.reset_index().merge(inventory_turnover(movements, batches), on="product_id")


def store_kpis(k: pd.DataFrame, consumption: pd.DataFrame) -> dict:
    """Headline numbers for the whole store (all values computed, none typed in)."""
    issued_units, requested_units = consumption.quantity_issued.sum(), consumption.quantity_requested.sum()
    period_days = consumption.transaction_date.nunique()   # length of the observed period, not typed in
    return {
        "stock_value_inr": float(k.stock_value_inr.sum()),
        "expired_value_inr_in_period": float(k.expired_value_inr.sum()),
        "expiring_within_90d_value_inr": float(k.expiring_90d_value_inr.sum()),
        "at_risk_of_expiry_value_inr": float(k.at_risk_expiry_value_inr.sum()),
        "excess_stock_value_inr": float(k.excess_value_inr.sum()),
        "fill_rate": float(issued_units / requested_units),
        "stockout_days": int((consumption.quantity_issued < consumption.quantity_requested).sum()),
        "products_overstocked": int((k.stock_status == "Overstocked").sum()),
        "products_stockout_risk": int(k.stock_status.isin(["Stockout risk", "Out of stock"]).sum()),
        "store_turnover": float(k.issued_value_inr.sum() * 365 / period_days / k.avg_inventory_value_inr.sum()),
    }
