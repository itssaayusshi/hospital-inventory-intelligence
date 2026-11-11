"""Data-quality checks for the four synthetic hospital datasets.

Every check returns rows of (check, table, severity, count, detail) so results
can be shown in a table, saved to CSV, or asserted in tests. Nothing is fixed
silently: this module only *reports*.
"""
from __future__ import annotations
import pandas as pd

INVOICE_TOLERANCE = 0.02  # flag invoices that differ from qty x unit cost by >2%
PRICE_TOLERANCE = 0.05    # paid price may drift from master list price by up to 5%


def _row(check, table, severity, count, detail=""):
    return {"check": check, "table": table, "severity": severity, "count": int(count), "detail": detail}


def validate(products, batches, consumption, procurement, as_of=None):
    """Run all checks. severity: ERROR (blocks use), WARN (needs a decision), INFO."""
    r = []
    as_of = pd.Timestamp(as_of) if as_of is not None else consumption["transaction_date"].max()
    tables = {"products": (products, "product_id"), "batches": (batches, "batch_id"),
              "consumption": (consumption, "transaction_id"), "procurement": (procurement, "purchase_order_id")}

    for name, (df, key) in tables.items():
        r.append(_row("duplicate primary keys", name, "ERROR", df[key].duplicated().sum()))
        r.append(_row("missing values", name, "ERROR", df.isna().sum().sum()))

    # negative / zero quantities
    for name, df, col in [("batches", batches, "received_qty"), ("batches", batches, "remaining_qty"),
                          ("consumption", consumption, "quantity_issued"), ("procurement", procurement, "order_qty")]:
        r.append(_row(f"negative {col}", name, "ERROR", (df[col] < 0).sum()))
        sev = "INFO" if col == "remaining_qty" else "WARN"   # a fully used batch legitimately has 0 left
        r.append(_row(f"zero {col}", name, sev, (df[col] == 0).sum()))

    # foreign keys
    valid = set(products["product_id"])
    for name, df in [("batches", batches), ("consumption", consumption), ("procurement", procurement)]:
        r.append(_row("product_id not in product master", name, "ERROR", (~df["product_id"].isin(valid)).sum()))
    r.append(_row("products with no consumption", "products", "WARN", len(valid - set(consumption["product_id"]))))

    # date logic
    r.append(_row("expiry_date <= purchase_date", "batches", "ERROR", (batches.expiry_date <= batches.purchase_date).sum()))
    r.append(_row("delivery_date < order_date", "procurement", "ERROR", (procurement.delivery_date < procurement.order_date).sum()))
    r.append(_row("purchase_date after as_of", "batches", "ERROR", (batches.purchase_date > as_of).sum()))
    r.append(_row("transaction_date after as_of", "consumption", "ERROR", (consumption.transaction_date > as_of).sum()))
    r.append(_row("batches past expiry date at as_of (incl. used-up ones)", "batches", "INFO", (batches.expiry_date <= as_of).sum(), f"as_of={as_of.date()}"))
    r.append(_row("expired batches still holding stock", "batches", "ERROR", ((batches.expiry_date <= as_of) & (batches.remaining_qty > 0)).sum()))

    # consistency with product master
    m = batches.merge(products, on="product_id", suffixes=("", "_m"))
    r.append(_row(f"batch unit cost differs from master by >{PRICE_TOLERANCE:.0%}", "batches", "ERROR",
                  ((m.unit_cost_inr / m.unit_cost_inr_m - 1).abs() > PRICE_TOLERANCE + 1e-9).sum()))
    r.append(_row("batch vendor != preferred vendor", "batches", "WARN", (m.vendor != m.preferred_vendor).sum()))
    m = procurement.merge(products, on="product_id", suffixes=("", "_m"))
    r.append(_row(f"PO unit cost differs from master by >{PRICE_TOLERANCE:.0%}", "procurement", "ERROR",
                  ((m.unit_cost_inr / m.unit_cost_inr_m - 1).abs() > PRICE_TOLERANCE + 1e-9).sum()))
    r.append(_row("PO price differs from master list price (any amount)", "procurement", "INFO",
                  (m.unit_cost_inr != m.unit_cost_inr_m).sum(), "normal: negotiated price variation"))
    r.append(_row("PO vendor != preferred vendor", "procurement", "WARN", (m.vendor != m.preferred_vendor).sum()))
    gap = (procurement.total_invoice_value_inr / (procurement.order_qty * procurement.unit_cost_inr) - 1).abs()
    r.append(_row(f"invoice differs from qty x cost by >{INVOICE_TOLERANCE:.0%}", "procurement", "WARN",
                  (gap > INVOICE_TOLERANCE).sum(), f"max deviation {gap.max():.1%}"))
    r.append(_row("remaining_qty > received_qty", "batches", "ERROR", (batches.remaining_qty > batches.received_qty).sum()))
    r.append(_row("batches with remaining == received (nothing consumed)", "batches", "INFO",
                  (batches.remaining_qty == batches.received_qty).sum(), f"of {len(batches)} batches"))
    if "quantity_requested" in consumption:
        r.append(_row("issued > requested", "consumption", "ERROR", (consumption.quantity_issued > consumption.quantity_requested).sum()))
        r.append(_row("days with unmet demand (stockout)", "consumption", "INFO",
                      (consumption.quantity_issued < consumption.quantity_requested).sum()))
    if "status" in procurement:
        r.append(_row("purchase orders still in transit", "procurement", "INFO", (procurement.status == "In transit").sum()))

    # consumption shape
    days = (consumption.transaction_date.max() - consumption.transaction_date.min()).days + 1
    per = consumption.groupby("product_id").transaction_date.nunique()
    r.append(_row("products with missing consumption days", "consumption", "WARN", (per < days).sum(), f"window={days} days"))
    r.append(_row("duplicate product/date rows", "consumption", "WARN", consumption.duplicated(["product_id", "transaction_date"]).sum()))
    q1, q3 = consumption.groupby("product_id").quantity_issued.transform(lambda s: s.quantile(.25)), \
             consumption.groupby("product_id").quantity_issued.transform(lambda s: s.quantile(.75))
    iqr = q3 - q1
    out = (consumption.quantity_issued > q3 + 3 * iqr) | (consumption.quantity_issued < q1 - 3 * iqr)
    r.append(_row("extreme consumption outliers (3xIQR)", "consumption", "INFO", out.sum()))
    return pd.DataFrame(r)
