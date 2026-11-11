import pandas as pd
from src.reconciliation import build_ledger, reconcile
from src.validation import validate
from src.data_loader import load_data


def toy():
    prod = pd.DataFrame({"product_id": ["A"], "product_name": ["Item"], "unit_cost_inr": [10.0], "preferred_vendor": ["V"]})
    cons = pd.DataFrame({"transaction_id": ["t1", "t2", "t3"], "product_id": "A",
                         "transaction_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
                         "quantity_issued": [10, 10, 10], "department": "OPD"})
    po = pd.DataFrame({"purchase_order_id": ["p1"], "product_id": "A", "vendor": "V",
                       "order_date": pd.to_datetime(["2025-12-30"]), "delivery_date": pd.to_datetime(["2026-01-02"]),
                       "order_qty": [15], "total_invoice_value_inr": [150.0], "unit_cost_inr": [10.0]})
    bat = pd.DataFrame({"batch_id": ["b1"], "product_id": "A", "purchase_date": pd.to_datetime(["2026-01-01"]),
                        "expiry_date": pd.to_datetime(["2026-06-01"]), "received_qty": [20], "remaining_qty": [20],
                        "vendor": "V", "unit_cost_inr": [10.0]})
    return prod, bat, cons, po


def test_ledger_balance_math():
    prod, bat, cons, po = toy()
    led = build_ledger(prod, cons, po)
    # day1: -10, day2: -10+15=+5 (cum -5), day3: -10 (cum -15)
    assert led.cum_net.tolist() == [-10, -5, -15]


def test_required_opening_stock_is_deepest_deficit():
    prod, bat, cons, po = toy()
    r = reconcile(prod, bat, cons, po).iloc[0]
    assert r.required_opening_stock == 15
    assert r.implied_closing_stock == 0  # 15 opening - 15 net
    assert r.first_negative_date == pd.Timestamp("2026-01-01")


def test_shortfall_with_and_without_batches():
    prod, bat, cons, po = toy()
    r = reconcile(prod, bat, cons, po).iloc[0]
    assert r.issues_not_covered_by_PO == 15       # 30 issued - 15 received
    assert r.issues_not_covered_by_PO_or_batches == 0  # 20-unit batch would cover it


def test_validation_catches_injected_errors():
    prod, bat, cons, po = toy()
    bat.loc[0, "expiry_date"] = pd.Timestamp("2025-01-01")
    cons.loc[0, "quantity_issued"] = -5
    cons.loc[1, "product_id"] = "ZZ"
    rep = validate(prod, bat, cons, po).set_index("check")["count"]
    assert rep["expiry_date <= purchase_date"] == 1
    assert rep["negative quantity_issued"] == 1
    assert rep.loc["product_id not in product master"].iloc[1] == 1  # consumption table


def test_real_data_loads_and_has_no_hard_errors():
    d = load_data()
    rep = validate(d["products"], d["batches"], d["consumption"], d["procurement"])
    assert rep[(rep.severity == "ERROR") & (rep["count"] > 0)].empty
    assert len(d["consumption"]) == 1800 and d["products"].shape[0] == 10
