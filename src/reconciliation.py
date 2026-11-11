"""Stock reconciliation.

Idea: a trustworthy stock balance must be explainable by movements:

    closing = opening + receipts - issues (+/- adjustments)

The supplied batch `remaining_qty` is only a snapshot, so we never trust it on
its own. We rebuild the balance from the movement tables and measure how far
the snapshot is from it.
"""
from __future__ import annotations
import pandas as pd


def build_ledger(products, consumption, procurement, as_of=None):
    """Daily ledger per product: receipts (on delivery_date) and issues (on transaction_date)."""
    start = consumption.transaction_date.min()
    end = pd.Timestamp(as_of) if as_of is not None else consumption.transaction_date.max()
    days = pd.date_range(start, end, freq="D")
    idx = pd.MultiIndex.from_product([products["product_id"], days], names=["product_id", "date"])
    rec = (procurement[procurement.delivery_date.between(start, end)]
           .groupby(["product_id", "delivery_date"]).order_qty.sum().rename("receipts"))
    rec.index.names = ["product_id", "date"]
    iss = consumption.groupby(["product_id", "transaction_date"]).quantity_issued.sum().rename("issues")
    iss.index.names = ["product_id", "date"]
    led = pd.DataFrame(index=idx).join(rec).join(iss).fillna(0).reset_index()
    led["net_movement"] = led.receipts - led.issues
    led["cum_net"] = led.groupby("product_id").net_movement.cumsum()
    return led


def reconcile(products, batches, consumption, procurement, as_of=None):
    """One row per product comparing movement-based stock with the batch snapshot."""
    led = build_ledger(products, consumption, procurement, as_of)
    g = led.groupby("product_id")
    out = pd.DataFrame({
        "total_receipts": g.receipts.sum(),
        "total_issues": g.issues.sum(),
        "ledger_closing_zero_opening": g.net_movement.sum(),
        "lowest_balance_zero_opening": g.cum_net.min(),
    })
    out["first_negative_date"] = led[led.cum_net < 0].groupby("product_id").date.min()
    # Opening stock that would have been needed so the balance never drops below zero
    out["required_opening_stock"] = (-out.lowest_balance_zero_opening).clip(lower=0)
    out["implied_closing_stock"] = out.required_opening_stock + out.ledger_closing_zero_opening
    snap = batches.groupby("product_id").agg(batch_snapshot_qty=("remaining_qty", "sum"),
                                              batch_received_qty=("received_qty", "sum"))
    out = out.join(snap)
    # Shortfall if snapshot batches are treated as extra receipts (an optimistic reading)
    out["issues_not_covered_by_PO"] = (out.total_issues - out.total_receipts).clip(lower=0)
    out["issues_not_covered_by_PO_or_batches"] = (out.total_issues - out.total_receipts - out.batch_received_qty).clip(lower=0)
    out["snapshot_minus_implied_closing"] = out.batch_snapshot_qty - out.implied_closing_stock
    out["status"] = "Reconciled"
    out.loc[out.required_opening_stock > 0, "status"] = "Needs opening stock"
    out.loc[out.issues_not_covered_by_PO_or_batches > 0, "status"] = "Infeasible: issues exceed all recorded stock"
    return out.reset_index().merge(products[["product_id", "product_name"]], on="product_id")


def reconcile_movements(products, batches, movements, as_of=None):
    """Check that every batch balance equals the sum of its stock movements.

    Returns (product_level, batch_level). `difference` must be 0 everywhere.
    """
    mv = movements if as_of is None else movements[movements.movement_date <= pd.Timestamp(as_of)]
    ledger = mv.groupby("batch_id").quantity.sum().rename("ledger_qty")
    b = batches.set_index("batch_id").join(ledger).fillna({"ledger_qty": 0})
    b["difference"] = b.remaining_qty - b.ledger_qty
    b["status_check"] = b["difference"].map(lambda x: "Reconciled" if x == 0 else "MISMATCH")

    p = mv.pivot_table(index="product_id", columns="movement_type", values="quantity", aggfunc="sum", fill_value=0)
    for col in ["OPENING", "RECEIPT", "ISSUE", "EXPIRY_WRITE_OFF"]:
        if col not in p:
            p[col] = 0
    p = p[["OPENING", "RECEIPT", "ISSUE", "EXPIRY_WRITE_OFF"]]
    p["ledger_closing"] = p.sum(axis=1)
    p["batch_snapshot"] = batches.groupby("product_id").remaining_qty.sum()
    p["difference"] = p.batch_snapshot - p.ledger_closing
    # end-of-day product balance must never go negative
    daily = mv.groupby(["product_id", "movement_date"]).quantity.sum().groupby(level=0).cumsum()
    p["lowest_daily_balance"] = daily.groupby(level=0).min()
    p["status_check"] = (p["difference"].eq(0) & p["lowest_daily_balance"].ge(0)).map({True: "Reconciled", False: "MISMATCH"})
    p = p.reset_index().merge(products[["product_id", "product_name"]], on="product_id")
    return p, b.reset_index()
