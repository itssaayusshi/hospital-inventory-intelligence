"""Expiry monitoring and FEFO (First Expiry, First Out).

Only batches that still hold stock matter, so everything starts by dropping remaining_qty == 0.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

BUCKETS = [(-np.inf, 0, "Expired"), (0, 30, "0-30 days"), (30, 60, "31-60 days"), (60, 90, "61-90 days"), (90, np.inf, ">90 days")]


def expiry_status(batches: pd.DataFrame, as_of, avg_daily_demand: pd.Series | None = None) -> pd.DataFrame:
    """One row per batch WITH stock: days to expiry, bucket, value, and (optional) units expected to expire.

    FEFO-aware risk: for each product, line batches up by expiry date. Demand will eat the earliest batch
    first. Units that demand cannot reach before a batch expires are 'at risk':
        at_risk = clip(cumulative_stock_through_this_batch - avg_daily_demand x days_to_expiry, 0, batch_qty)
    This ignores future deliveries (it assumes nothing more arrives), so treat it as a warning, not a forecast.
    """
    as_of = pd.Timestamp(as_of)
    x = batches[batches.remaining_qty > 0].copy()
    x["days_to_expiry"] = (x.expiry_date - as_of).dt.days
    x["bucket"] = pd.cut(x.days_to_expiry, [b[0] for b in BUCKETS] + [np.inf], labels=[b[2] for b in BUCKETS], right=True).astype(str)
    x["stock_value_inr"] = x.remaining_qty * x.unit_cost_inr
    x = x.sort_values(["product_id", "expiry_date", "batch_id"]).reset_index(drop=True)
    x["fefo_rank"] = x.groupby("product_id").cumcount() + 1
    if avg_daily_demand is not None:
        x["avg_daily_demand"] = x.product_id.map(avg_daily_demand).fillna(0)
        x["cum_stock"] = x.groupby("product_id").remaining_qty.cumsum()
        x["demand_before_expiry"] = x.avg_daily_demand * x.days_to_expiry.clip(lower=0)
        x["at_risk_qty"] = (x.cum_stock - x.demand_before_expiry).clip(lower=0).clip(upper=x.remaining_qty).round().astype(int)
        x["at_risk_value_inr"] = x.at_risk_qty * x.unit_cost_inr
    return x


def exposure_by_window(status: pd.DataFrame, windows=(30, 60, 90)) -> pd.DataFrame:
    """Value of stock expiring within each window (cumulative: the 90-day figure includes the 30-day one)."""
    rows = [{"window_days": w, "batches": int((status.days_to_expiry <= w).sum()),
             "units": int(status.loc[status.days_to_expiry <= w, "remaining_qty"].sum()),
             "value_inr": float(status.loc[status.days_to_expiry <= w, "stock_value_inr"].sum())} for w in windows]
    return pd.DataFrame(rows)


def fefo_issue_plan(batches: pd.DataFrame, product_id: str, quantity: int, as_of) -> pd.DataFrame:
    """Which batches to pick from, to issue `quantity` units today.

    Earliest expiry first. Expired batches are never picked. If stock is short, the plan covers what exists and
    column `shortfall` on every row shows what is missing.
    """
    if quantity < 0:
        raise ValueError("quantity must be >= 0")
    as_of = pd.Timestamp(as_of)
    pool = batches[(batches.product_id == product_id) & (batches.remaining_qty > 0) & (batches.expiry_date > as_of)]
    pool = pool.sort_values(["expiry_date", "batch_id"])
    left, rows = quantity, []
    for _, b in pool.iterrows():
        if left == 0:
            break
        take = int(min(b.remaining_qty, left))
        rows.append({"batch_id": b.batch_id, "expiry_date": b.expiry_date, "pick_qty": take})
        left -= take
    plan = pd.DataFrame(rows, columns=["batch_id", "expiry_date", "pick_qty"])
    plan["shortfall"] = left
    return plan
