import pandas as pd
import pytest
from src.eda import demand_summary, demand_trend, add_delivery_metrics, vendor_performance, day_of_week_pattern


def toy_consumption():
    return pd.DataFrame({
        "transaction_id": list("abcd"), "product_id": "A",
        "transaction_date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]),  # Mon-Thu
        "quantity_requested": [10, 20, 10, 20], "quantity_issued": [10, 20, 10, 15], "department": "OPD"})


PRODUCTS = pd.DataFrame({"product_id": ["A"], "product_name": ["Item"], "category": "c", "criticality": "Vital", "unit_cost_inr": [2.0]})


def test_demand_summary_known_values():
    r = demand_summary(toy_consumption(), PRODUCTS).iloc[0]
    assert r.avg_daily_demand == 15
    assert r.stockout_days == 1 and r.units_short == 5
    assert r.fill_rate == pytest.approx(55 / 60)
    assert r.cv == pytest.approx(pd.Series([10, 20, 10, 20]).std() / 15)  # sample std
    assert r.annual_demand_value_inr == pytest.approx(15 * 365 * 2)


def test_demand_uses_requested_not_issued():
    r = demand_summary(toy_consumption(), PRODUCTS).iloc[0]
    assert r.total_requested == 60 and r.total_issued == 55


def test_partial_week_is_flagged():
    t = demand_trend(toy_consumption(), "W")
    assert len(t) == 1 and bool(t.is_partial.iloc[0]) and t.avg_daily_demand.iloc[0] == 15


def test_delivery_metrics_and_in_transit_excluded():
    po = pd.DataFrame({
        "purchase_order_id": ["p1", "p2", "p3"], "product_id": "A", "vendor": "V",
        "order_date": pd.to_datetime(["2026-01-01"] * 3),
        "promised_delivery_date": pd.to_datetime(["2026-01-05", "2026-01-05", "2026-01-05"]),
        "delivery_date": pd.to_datetime(["2026-01-04", "2026-01-07", "2026-01-05"]),
        "order_qty": 10, "total_invoice_value_inr": 100.0, "unit_cost_inr": 10.0,
        "status": ["Delivered", "Delivered", "In transit"]})
    m = add_delivery_metrics(po)
    assert m.delay_days.iloc[:2].tolist() == [-1, 2] and pd.isna(m.delay_days.iloc[2])
    v = vendor_performance(po).iloc[0]
    assert v.orders == 3 and v.delivered_orders == 2 and v.on_time_rate == 0.5
    assert v.avg_actual_lead_days == pytest.approx(4.5)


def test_weekday_pattern_order():
    d = day_of_week_pattern(toy_consumption())
    assert d.weekday.iloc[0] == "Monday" and d.avg_total_units.iloc[0] == 10
