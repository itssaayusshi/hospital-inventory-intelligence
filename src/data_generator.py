"""Seeded generator for an internally consistent SYNTHETIC hospital inventory dataset.

All data produced here is synthetic. Nothing is real hospital data.

How consistency is guaranteed
-----------------------------
We do not invent stock numbers. We *simulate* a store room day by day:

  1. The store starts with opening batches.
  2. Every day, staff request items (the demand file we already had).
  3. Stock is issued FEFO (First Expiry, First Out). If stock is short,
     only the available quantity is issued (the rest is a stockout).
  4. Batches whose expiry date arrives are written off.
  5. On review days, a simple "hospital style" rule places purchase orders.
  6. Orders arrive after a (noisy) lead time and become new batches.

Because every unit moves through a `stock_movements` ledger, the final
`remaining_qty` of every batch is *derived* from movements, so it reconciles
by construction.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

SEED = 42
PACK = 10  # orders are rounded up to multiples of this
SHORT_LIFE_PROB = 0.35     # share of deliveries that arrive with little shelf life left
SHORT_LIFE_DAYS = (25, 75)

# product_id: (reorder_trigger_days_of_cover, order_up_to_days_of_cover, review_every_days,
#              received_shelf_life_min, received_shelf_life_max)
POLICY = {
    "P001": (12, 30, 7, 150, 400),
    "P002": (14, 32, 7, 200, 500),
    "P003": (12, 30, 7, 180, 400),
    "P004": (10, 28, 7, 120, 300),
    "P005": (17, 34, 7, 180, 500),
    "P006": (24, 60, 14, 150, 450),
    "P007": (45, 130, 14, 150, 365),  # expensive, slow mover, deliberately over-ordered
    "P008": (14, 32, 7, 100, 240),
    "P009": (18, 45, 14, 150, 400),
    "P010": (17, 55, 14, 150, 400),
}


def generate(demand: pd.DataFrame, products: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Build batches, movements, procurement and consumption from demand + product master."""
    rng = np.random.default_rng(seed)
    demand = demand.copy()
    demand["transaction_date"] = pd.to_datetime(demand["transaction_date"])
    start, end = demand.transaction_date.min(), demand.transaction_date.max()
    days = pd.date_range(start, end, freq="D")

    movements, batches, pos, consumption = [], [], [], []
    mv_n = [0]
    b_n = [0]
    po_n = [0]

    def add_move(date, pid, batch, mtype, qty, ref):
        mv_n[0] += 1
        movements.append(dict(movement_id=f"M{mv_n[0]:06d}", movement_date=date, product_id=pid,
                              batch_id=batch, movement_type=mtype, quantity=int(qty), reference_id=ref))

    for _, p in products.iterrows():
        pid = p.product_id
        trig, up_to, review, shelf_lo, shelf_hi = POLICY[pid]
        d = demand[demand.product_id == pid].set_index("transaction_date").sort_index()
        d = d.reindex(days)  # every product has every day
        mean_demand = d.quantity_issued.mean()
        qty_series = d.quantity_issued.to_numpy()
        stock: list[dict] = []      # live batches
        open_orders: list[dict] = []
        review_offset = int(rng.integers(0, review))
        planning_avg = mean_demand * rng.uniform(0.9, 1.1)

        def new_batch(received, expiry, qty, price, vendor, po_id, order_date):
            b_n[0] += 1
            b = dict(batch_id=f"B{b_n[0]:04d}", product_id=pid, purchase_order_id=po_id,
                     purchase_date=order_date, received_date=received, expiry_date=expiry,
                     received_qty=int(qty), remaining_qty=int(qty), vendor=vendor, unit_cost_inr=price)
            batches.append(b); stock.append(b)
            add_move(received, pid, b["batch_id"], "OPENING" if po_id == "OPENING" else "RECEIPT", qty, po_id)

        # --- opening stock: an older batch (short residual life) and a newer one
        open_total = max(PACK, round(mean_demand * rng.uniform(20, 35)))
        old_share = rng.uniform(0.35, 0.6)
        for share, (lo, hi) in [(old_share, (15, 90)), (1 - old_share, (90, 240))]:
            qty = max(1, round(open_total * share))
            new_batch(start, start + pd.Timedelta(days=int(rng.integers(lo, hi + 1))), qty,
                      float(p.unit_cost_inr), p.preferred_vendor, "OPENING", start - pd.Timedelta(days=int(rng.integers(30, 90))))

        for i, day in enumerate(days):
            # 1) receipts arriving today
            for o in [o for o in open_orders if o["actual_delivery"] == day]:
                lo, hi = SHORT_LIFE_DAYS if rng.random() < SHORT_LIFE_PROB else (shelf_lo, shelf_hi)
                expiry = day + pd.Timedelta(days=int(rng.integers(lo, hi + 1)))
                new_batch(day, expiry, o["qty"], o["price"], p.preferred_vendor, o["po_id"], o["order_date"])
                open_orders.remove(o)
            # 2) expiry write-offs (usable up to the day BEFORE expiry)
            for b in stock:
                if b["expiry_date"] <= day and b["remaining_qty"] > 0:
                    add_move(day, pid, b["batch_id"], "EXPIRY_WRITE_OFF", -b["remaining_qty"], "EXPIRY")
                    b["remaining_qty"] = 0
            stock = [b for b in stock if b["remaining_qty"] > 0]
            # 3) issue FEFO
            requested = int(qty_series[i])
            tx = d.iloc[i]
            remaining_to_issue = requested
            for b in sorted(stock, key=lambda x: x["expiry_date"]):
                if remaining_to_issue == 0:
                    break
                take = min(b["remaining_qty"], remaining_to_issue)
                b["remaining_qty"] -= take
                remaining_to_issue -= take
                add_move(day, pid, b["batch_id"], "ISSUE", -take, tx.transaction_id)
            stock = [b for b in stock if b["remaining_qty"] > 0]
            consumption.append(dict(transaction_id=tx.transaction_id, product_id=pid, transaction_date=day,
                                    quantity_requested=requested, quantity_issued=requested - remaining_to_issue,
                                    department=tx.department))
            # 4) end-of-day review
            if i % review == review_offset:
                avg = qty_series[max(0, i - 27): i + 1].mean() if i >= 7 else planning_avg
                position = sum(b["remaining_qty"] for b in stock) + sum(o["qty"] for o in open_orders)
                if position <= trig * avg:
                    qty = math.ceil((up_to * avg - position) / PACK) * PACK
                    promised = day + pd.Timedelta(days=int(p.lead_time_days))
                    actual = day + pd.Timedelta(days=max(1, int(p.lead_time_days) + int(rng.integers(-1, 4))))
                    price = round(float(p.unit_cost_inr) * float(np.clip(1 + rng.normal(0, 0.02), 0.95, 1.05)), 3)
                    po_n[0] += 1
                    po_id = f"PO{po_n[0]:04d}"
                    open_orders.append(dict(po_id=po_id, qty=qty, price=price, order_date=day, actual_delivery=actual))
                    pos.append(dict(purchase_order_id=po_id, product_id=pid, vendor=p.preferred_vendor,
                                    order_date=day, promised_delivery_date=promised, delivery_date=actual,
                                    order_qty=qty, total_invoice_value_inr=round(qty * price, 2),
                                    unit_cost_inr=price,
                                    status="Delivered" if actual <= end else "In transit"))
        # an order still in transit: show its PROMISED date (actual arrival is unknown to the hospital)
    po_df = pd.DataFrame(pos)
    in_transit = po_df.status == "In transit"
    po_df.loc[in_transit, "delivery_date"] = po_df.loc[in_transit, "promised_delivery_date"]

    batch_df = pd.DataFrame(batches)
    batch_df["status"] = "Active"
    batch_df.loc[batch_df.remaining_qty == 0, "status"] = "Depleted"
    mv_df = pd.DataFrame(movements)
    expired_ids = mv_df[mv_df.movement_type == "EXPIRY_WRITE_OFF"].batch_id
    batch_df.loc[batch_df.batch_id.isin(expired_ids), "status"] = "Expired"
    batch_df = batch_df.sort_values(["product_id", "received_date", "batch_id"]).reset_index(drop=True)
    cons_df = pd.DataFrame(consumption).sort_values("transaction_id").reset_index(drop=True)
    po_df = po_df.sort_values(["product_id", "order_date"]).reset_index(drop=True)
    return {"batches": batch_df, "movements": mv_df, "procurement": po_df, "consumption": cons_df}
