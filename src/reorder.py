"""Safety stock, reorder point and order quantity.

Formulas (L = lead time in days, mu/sigma = mean/std of DAILY demand, sigma_L = std of lead time)
  Safety stock     SS  = z * sqrt( L * sigma^2  +  mu^2 * sigma_L^2 )
                          demand uncertainty      supplier-delay uncertainty
  Reorder point    ROP = mu * L + SS
  Order-up-to      S   = ROP + mu * R            (R = review period: how often we place orders)
  Inventory pos.   IP  = usable on-hand + on-order     (what we really "have" counting incoming POs)
  Order quantity   Q   = S - IP, rounded UP to the pack size, only when IP <= ROP

z comes from the service level (probability of NOT stocking out in a cycle): Vital 99%, Essential 97.5%, Desirable 95%.
These service levels are ASSUMPTIONS; a real hospital would set them with clinicians.
"""
from __future__ import annotations
import math
from statistics import NormalDist
import numpy as np
import pandas as pd

SERVICE_LEVEL = {"Vital": 0.99, "Essential": 0.975, "Desirable": 0.95}
REVIEW_DAYS = 7
MAX_COVER_DAYS = 45
PACK = 10


def z_score(service_level: float) -> float:
    if not 0.5 < service_level < 1:
        raise ValueError("service level must be between 0.5 and 1")
    return NormalDist().inv_cdf(service_level)


def safety_stock(mu, sigma, lead_days, lead_std, z):
    if min(mu, sigma, lead_days, lead_std) < 0:
        raise ValueError("inputs must be non-negative")
    return z * math.sqrt(lead_days * sigma ** 2 + (mu ** 2) * lead_std ** 2)


def order_quantity(order_up_to, inventory_position, reorder_point, pack=PACK):
    """0 unless inventory position has fallen to/below the reorder point."""
    if inventory_position > reorder_point:
        return 0
    need = max(0.0, order_up_to - inventory_position)
    return int(math.ceil(need / pack) * pack)


def lead_time_stats(procurement: pd.DataFrame, products: pd.DataFrame, min_orders: int = 3) -> pd.DataFrame:
    """Actual lead time (order -> delivery) per product from DELIVERED orders.
    With fewer than `min_orders` orders the average is unreliable, so we fall back to the planned lead time
    from the product master and the lead-time spread pooled over all products."""
    done = procurement[procurement.status == "Delivered"].copy()
    done["lt"] = (done.delivery_date - done.order_date).dt.days
    pooled_std = done.groupby("product_id").lt.std().mean()
    g = done.groupby("product_id").lt.agg(["count", "mean", "std"])
    out = products[["product_id", "lead_time_days"]].set_index("product_id").join(g)
    ok = out["count"].fillna(0) >= min_orders
    out["lead_time_mean"] = np.where(ok, out["mean"], out.lead_time_days)
    out["lead_time_std"] = np.where(ok, out["std"], pooled_std)
    out["lead_time_source"] = np.where(ok, f"actual (n>={min_orders})", "planned (too few orders)")
    return out[["lead_time_mean", "lead_time_std", "lead_time_source"]].reset_index()


def reorder_table(products, demand_daily_mean, demand_daily_std, lead_stats, batches, procurement, as_of, expiry=None,
                  review_days=REVIEW_DAYS, max_cover_days=MAX_COVER_DAYS, pack=PACK, service_levels=SERVICE_LEVEL):
    """One row per product: policy numbers, current position, recommendation and a plain-English reason.

    demand_daily_mean: Series (product_id -> forecast mean daily demand); demand_daily_std: Series (product_id -> std).
    """
    t = products[["product_id", "product_name", "criticality", "unit_cost_inr"]].copy()
    t["mu"] = t.product_id.map(demand_daily_mean); t["sigma"] = t.product_id.map(demand_daily_std)
    t = t.merge(lead_stats, on="product_id")
    t["service_level"] = t.criticality.map(service_levels); t["z"] = t.service_level.map(z_score)
    t["safety_stock"] = [safety_stock(r.mu, r.sigma, r.lead_time_mean, r.lead_time_std, r.z) for r in t.itertuples()]
    t["reorder_point"] = t.mu * t.lead_time_mean + t.safety_stock
    t["order_up_to"] = np.maximum(t.reorder_point, np.minimum(t.reorder_point + t.mu * review_days, t.mu * max_cover_days))

    on_hand = batches.groupby("product_id").remaining_qty.sum()
    at_risk = expiry.groupby("product_id").at_risk_qty.sum() if expiry is not None else pd.Series(dtype=float)
    transit = procurement[procurement.status == "In transit"]
    t["on_hand"] = t.product_id.map(on_hand).fillna(0)
    t["at_risk_expiry"] = t.product_id.map(at_risk).fillna(0)
    t["usable_on_hand"] = (t.on_hand - t.at_risk_expiry).clip(lower=0)
    t["on_order"] = t.product_id.map(transit.groupby("product_id").order_qty.sum()).fillna(0)
    first_due = transit.groupby("product_id").delivery_date.min().reindex(t.product_id)   # reindex keeps the datetime type even when empty
    t["next_delivery"] = pd.to_datetime(first_due.to_numpy())                                # NaT when nothing is in transit
    as_of = pd.Timestamp(as_of)
    t["days_overdue"] = (as_of - t.next_delivery).dt.days.clip(lower=0).fillna(0).astype(int)   # promised date already passed
    t["inventory_position"] = t.usable_on_hand + t.on_order
    t["days_cover_on_hand"] = t.on_hand / t.mu
    t["days_cover_with_orders"] = t.inventory_position / t.mu
    t["recommended_qty"] = [order_quantity(r.order_up_to, r.inventory_position, r.reorder_point, pack) for r in t.itertuples()]
    t["order_value_inr"] = t.recommended_qty * t.unit_cost_inr

    def classify(r):
        if r.on_order > 0:
            due = f"{int(r.on_order)} units due {r.next_delivery:%d %b}" + (f" ({r.days_overdue} days overdue)" if r.days_overdue > 0 else "")
        else:
            due = "nothing on order"
        if r.on_hand == 0:
            if r.recommended_qty > 0:
                return "Stocked out - order now", f"No stock and {due}."
            if r.days_overdue > 0:
                return "Stocked out - PO overdue", f"No stock; {due}. Chase the supplier."
            return "Stocked out - order in transit", f"No stock, but {due}."
        if r.recommended_qty > 0:
            return "Reorder now", f"Position {r.inventory_position:.0f} <= reorder point {r.reorder_point:.0f} ({due})."
        if r.days_cover_on_hand < r.lead_time_mean and r.on_order > 0:
            return "Watch: relies on incoming order", f"{r.days_cover_on_hand:.1f} days of stock vs {r.lead_time_mean:.1f}-day lead time; {due}."
        if r.days_cover_with_orders > max_cover_days:
            return "Overstocked", f"{r.days_cover_with_orders:.0f} days of cover incl. orders (limit {max_cover_days})."
        return "OK", f"{r.days_cover_on_hand:.0f} days of cover; position above reorder point."
    t[["status", "reason"]] = [classify(r) for r in t.itertuples()]
    return t
