"""One place that turns the five raw tables into everything the dashboard shows.

The dashboard pages contain NO business logic: they only display what `build_model` returns.
That keeps every number testable and guarantees all pages agree with each other.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

from src.eda import demand_summary, vendor_performance, price_comparison, spend_trend, day_of_week_pattern, add_delivery_metrics
from src.forecasting import forecast, backtest, best_method, DEFAULT_METHOD
from src.abc_ved import abc_classification, abc_ved_matrix
from src.expiry import expiry_status, exposure_by_window
from src.kpis import product_kpis, store_kpis, inventory_value_timeline
from src.reorder import lead_time_stats, reorder_table, SERVICE_LEVEL, REVIEW_DAYS, MAX_COVER_DAYS
from src.validation import validate
from src.reconciliation import reconcile_movements

# ---------------------------------------------------------------- schemas for uploaded CSVs
SCHEMA = {
    "products": (["product_id", "product_name", "category", "unit_cost_inr", "preferred_vendor", "lead_time_days", "criticality"], []),
    "batches": (["batch_id", "product_id", "purchase_order_id", "purchase_date", "received_date", "expiry_date", "received_qty",
                 "remaining_qty", "vendor", "unit_cost_inr"], ["purchase_date", "received_date", "expiry_date"]),
    "consumption": (["transaction_id", "product_id", "transaction_date", "quantity_requested", "quantity_issued", "department"], ["transaction_date"]),
    "procurement": (["purchase_order_id", "product_id", "vendor", "order_date", "promised_delivery_date", "delivery_date", "order_qty",
                     "total_invoice_value_inr", "unit_cost_inr", "status"], ["order_date", "promised_delivery_date", "delivery_date"]),
    "movements": (["movement_id", "movement_date", "product_id", "batch_id", "movement_type", "quantity", "reference_id"], ["movement_date"]),
}
FILE_NAMES = {"products": "product_master.csv", "batches": "inventory_batches.csv", "consumption": "consumption_transactions.csv",
              "procurement": "procurement_transactions.csv", "movements": "stock_movements.csv"}


class DataError(ValueError):
    """Raised with a readable message when uploaded data cannot be used."""


def parse_tables(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Check columns and convert date columns. Raises DataError listing every problem found."""
    problems, out = [], {}
    for name, (cols, dates) in SCHEMA.items():
        if name not in frames:
            problems.append(f"Missing table: {FILE_NAMES[name]}")
            continue
        df = frames[name].copy()
        missing = [c for c in cols if c not in df.columns]
        if missing:
            problems.append(f"{FILE_NAMES[name]}: missing columns {missing}")
            continue
        for c in dates:
            parsed = pd.to_datetime(df[c], errors="coerce")
            bad = int(parsed.isna().sum() - df[c].isna().sum())
            if bad:
                problems.append(f"{FILE_NAMES[name]}: {bad} unreadable dates in '{c}'")
            df[c] = parsed
        out[name] = df
    if problems:
        raise DataError("; ".join(problems))
    return out


def check_tables(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Data-quality report for a parsed set of tables, with a blocking-errors count in .attrs."""
    rep = validate(t["products"], t["batches"], t["consumption"], t["procurement"])
    _, batch_rec = reconcile_movements(t["products"], t["batches"], t["movements"])
    rep.loc[len(rep)] = {"check": "batch balance != sum of movements", "table": "batches", "severity": "ERROR",
                         "count": int((batch_rec.difference != 0).sum()), "detail": ""}
    return rep


# ---------------------------------------------------------------- the model
@dataclass
class Params:
    review_days: int = REVIEW_DAYS
    max_cover_days: int = MAX_COVER_DAYS
    excess_days: int = MAX_COVER_DAYS
    service_levels: dict = field(default_factory=lambda: dict(SERVICE_LEVEL))
    method: str = DEFAULT_METHOD


@dataclass
class Model:
    tables: dict
    as_of: pd.Timestamp
    params: Params
    demand: pd.DataFrame
    abc: pd.DataFrame
    expiry: pd.DataFrame
    kpi: pd.DataFrame
    store: dict
    reorder: pd.DataFrame
    lead: pd.DataFrame
    vendors: pd.DataFrame
    prices: pd.DataFrame
    backtest: pd.DataFrame
    ranking: pd.DataFrame
    alerts: pd.DataFrame
    inv_timeline: pd.DataFrame


def build_model(tables: dict[str, pd.DataFrame], params: Params | None = None) -> Model:
    p = params or Params()
    products, batches, cons, po, mv = (tables[k] for k in ["products", "batches", "consumption", "procurement", "movements"])
    as_of = cons.transaction_date.max()
    ds = demand_summary(cons, products)
    dsi = ds.set_index("product_id")
    abc = abc_classification(products, dsi.avg_daily_demand)
    exp = expiry_status(batches, as_of, dsi.avg_daily_demand)
    kpi = product_kpis(products, batches, mv, po, ds, exp, excess_days=p.excess_days)
    store = store_kpis(kpi, cons)
    bt = backtest(cons)
    mu = pd.Series({pid: forecast(g.sort_values("transaction_date").quantity_requested, g.sort_values("transaction_date").transaction_date, 28, p.method).mean()
                    for pid, g in cons.groupby("product_id")})
    lt = lead_time_stats(po, products)
    rt = reorder_table(products, mu, dsi.std_daily_demand, lt, batches, po, as_of, exp,
                       review_days=p.review_days, max_cover_days=p.max_cover_days, service_levels=p.service_levels)
    rt = rt.merge(abc[["product_id", "abc_ved", "priority"]], on="product_id", how="left")
    m = Model(tables, as_of, p, ds, abc, exp, kpi, store, rt, lt, vendor_performance(po), price_comparison(po, products), bt, best_method(bt),
              pd.DataFrame(), inventory_value_timeline(mv, batches))
    m.alerts = build_alerts(m)
    return m


def build_alerts(m: Model) -> pd.DataFrame:
    """Plain-language action list. severity: High / Medium. Sorted most urgent first."""
    rows = []
    for r in m.reorder.itertuples():
        if r.status.startswith("Stocked out"):
            rows.append(("High", r.product_id, r.product_name, r.status, r.reason))
        elif r.status == "Reorder now":
            rows.append(("High" if r.criticality == "Vital" else "Medium", r.product_id, r.product_name, r.status,
                         f"Order {r.recommended_qty} units (about Rs {r.order_value_inr:,.0f}). {r.reason}"))
        elif r.status.startswith("Watch") or r.status == "Overstocked":
            rows.append(("Medium", r.product_id, r.product_name, r.status, r.reason))
    names = m.tables["products"].set_index("product_id").product_name
    for r in m.expiry[m.expiry.days_to_expiry <= 90].itertuples():
        sev = "High" if (r.days_to_expiry <= 30 or r.at_risk_qty > 0) else "Medium"
        rows.append((sev, r.product_id, names.get(r.product_id, r.product_id), "Near expiry",
                     f"Batch {r.batch_id}: {r.remaining_qty} units (Rs {r.stock_value_inr:,.0f}) expire in {r.days_to_expiry} days. Issue this batch first (FEFO)."))
    a = pd.DataFrame(rows, columns=["severity", "product_id", "product", "issue", "detail"])
    if a.empty:
        return a
    a["_o"] = a.severity.map({"High": 0, "Medium": 1})
    return a.sort_values(["_o", "product_id"]).drop(columns="_o").reset_index(drop=True)


def text_filter(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Keep rows where ANY text column contains the query (case-insensitive)."""
    q = (query or "").strip().lower()
    if not q:
        return df
    txt = df.select_dtypes(include=["object", "string"]).astype(str)
    mask = txt.apply(lambda col: col.str.lower().str.contains(q, regex=False)).any(axis=1)
    return df[mask]


def waste_scenarios(m: Model, waste_cut: float, inventory_cut: float, holding_rate: float) -> pd.DataFrame:
    """SIMULATED what-if figures from this synthetic dataset. Every input is an assumption set by the user.

    expiry savings    = expired value in period x waste_cut
    carrying savings  = average inventory value x holding_rate x (period days / 365) x inventory_cut
    """
    days = m.tables["consumption"].transaction_date.nunique()
    avg_inv = float(m.inv_timeline.sum(axis=1).mean())
    expired = float(m.store["expired_value_inr_in_period"])
    rows = [
        {"scenario": f"Cut expiry waste by {waste_cut:.0%}", "base_inr": expired, "simulated_saving_inr": expired * waste_cut,
         "formula": "expired value in period x reduction"},
        {"scenario": f"Cut average inventory by {inventory_cut:.0%}", "base_inr": avg_inv,
         "simulated_saving_inr": avg_inv * holding_rate * days / 365 * inventory_cut,
         "formula": f"avg inventory value x {holding_rate:.0%} holding cost x {days}/365 x reduction"},
    ]
    return pd.DataFrame(rows)


def inventory_frame(m: Model) -> pd.DataFrame:
    """One row per product for the Inventory Overview page."""
    cat = m.tables["products"][["product_id", "category"]]
    k = m.kpi[["product_id", "stock_value_inr", "annualised_turnover"]]
    r = m.reorder[["product_id", "product_name", "criticality", "abc_ved", "on_hand", "on_order", "days_cover_on_hand", "lead_time_mean", "status"]]
    return r.merge(cat, on="product_id").merge(k, on="product_id")


def unmet_demand(m: Model) -> pd.DataFrame:
    """Units and rupee value of demand that could not be issued (stockouts)."""
    d = m.demand[["product_id", "product_name", "stockout_days", "units_short", "unit_cost_inr"]].copy()
    d["unmet_demand_value_inr"] = d.units_short * d.unit_cost_inr
    return d


def expired_history(m: Model) -> pd.DataFrame:
    """Every expiry write-off with its rupee value."""
    mv, b = m.tables["movements"], m.tables["batches"]
    w = mv[mv.movement_type == "EXPIRY_WRITE_OFF"].merge(b[["batch_id", "unit_cost_inr", "expiry_date"]], on="batch_id")
    w["units_expired"] = -w.quantity
    w["value_inr"] = w.units_expired * w.unit_cost_inr
    return w[["movement_date", "product_id", "batch_id", "units_expired", "value_inr"]].sort_values("movement_date").reset_index(drop=True)


def open_orders(m: Model) -> pd.DataFrame:
    po = m.tables["procurement"]
    o = po[po.status == "In transit"].copy()
    o["days_overdue"] = (m.as_of - o.delivery_date).dt.days.clip(lower=0)
    return o[["purchase_order_id", "product_id", "vendor", "order_date", "delivery_date", "order_qty", "total_invoice_value_inr", "days_overdue"]]
