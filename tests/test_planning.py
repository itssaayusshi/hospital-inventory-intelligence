import numpy as np
import pandas as pd
import pytest
from src.forecasting import moving_average, ses_level, weekday_index, forecast, backtest_series, METHODS
from src.reorder import z_score, safety_stock, order_quantity, lead_time_stats, reorder_table


# ---------- forecasting
def test_moving_average_and_ses_known_values():
    assert moving_average([1, 2, 3, 4, 5], 3) == 4
    assert ses_level([10, 20, 30], 0.5) == pytest.approx(22.5)   # 10 -> 15 -> 22.5
    with pytest.raises(ValueError):
        ses_level([1, 2], 0)


def test_forecast_starts_next_day_and_has_horizon():
    dates = pd.date_range("2026-01-01", periods=60); f = forecast(np.ones(60) * 5, dates, horizon=10)
    assert f.index[0] == pd.Timestamp("2026-03-02") and len(f) == 10 and np.allclose(f, 5)


def test_weekday_method_wins_on_a_weekly_pattern():
    dates = pd.date_range("2026-01-05", periods=140)                      # starts on a Monday
    y = np.tile([20, 20, 20, 10, 10, 10, 10], 20).astype(float)           # Mon-Wed high
    idx = weekday_index(dates, y)
    assert idx.iloc[0] == pytest.approx(20 / y.mean()) and idx.iloc[3] == pytest.approx(10 / y.mean())
    bt = backtest_series(y, dates)
    assert bt.loc["MA28 x weekday", "wape"] < 0.01 < bt.loc["Moving avg 28", "wape"]
    assert set(bt.index) == set(METHODS)


# ---------- reorder maths
def test_z_scores():
    assert z_score(0.975) == pytest.approx(1.95996, abs=1e-4)
    assert z_score(0.99) == pytest.approx(2.32635, abs=1e-4)
    with pytest.raises(ValueError):
        z_score(0.5)


def test_safety_stock_formula():
    assert safety_stock(10, 3, 4, 0, 2) == pytest.approx(12.0)                      # 2*sqrt(4*9)
    assert safety_stock(10, 3, 4, 1, 2) == pytest.approx(2 * np.sqrt(36 + 100))     # supplier delay adds risk
    with pytest.raises(ValueError):
        safety_stock(-1, 1, 1, 1, 1)


def test_order_quantity_rules():
    assert order_quantity(100, 40, 50) == 60
    assert order_quantity(100, 41.5, 50) == 60          # 58.5 rounds UP to the pack of 10
    assert order_quantity(100, 55, 50) == 0             # above reorder point: no order
    assert order_quantity(100, 50, 50) == 50            # exactly at ROP: order
    assert order_quantity(40, 45, 50) == 0


def test_lead_time_falls_back_when_few_orders():
    prods = pd.DataFrame({"product_id": ["A", "B"], "lead_time_days": [5, 9]})
    d = pd.to_datetime
    rows = [("A", "2026-01-01", "2026-01-05"), ("A", "2026-02-01", "2026-02-07"), ("A", "2026-03-01", "2026-03-06"),
            ("B", "2026-01-01", "2026-01-12"), ("B", "2026-02-01", "2026-02-12")]
    po = pd.DataFrame(rows, columns=["product_id", "order_date", "delivery_date"]).assign(status="Delivered")
    po[["order_date", "delivery_date"]] = po[["order_date", "delivery_date"]].apply(d)
    lt = lead_time_stats(po, prods).set_index("product_id")
    assert lt.loc["A", "lead_time_mean"] == pytest.approx(5.0) and "actual" in lt.loc["A", "lead_time_source"]
    assert lt.loc["B", "lead_time_mean"] == 9 and "planned" in lt.loc["B", "lead_time_source"]


def table(on_hand, on_order, next_delivery="2026-01-10", mu=10.0):
    prods = pd.DataFrame({"product_id": ["A"], "product_name": ["Item"], "criticality": ["Vital"], "unit_cost_inr": [2.0], "lead_time_days": [5]})
    lt = pd.DataFrame({"product_id": ["A"], "lead_time_mean": [5.0], "lead_time_std": [0.0], "lead_time_source": ["x"]})
    b = pd.DataFrame({"product_id": ["A"], "remaining_qty": [on_hand]})
    po = pd.DataFrame({"product_id": ["A"], "order_qty": [on_order], "status": ["In transit"] if on_order else ["Delivered"],
                       "delivery_date": pd.to_datetime([next_delivery])})
    return reorder_table(prods, pd.Series({"A": mu}), pd.Series({"A": 0.0}), lt, b, po, "2026-01-12").iloc[0]


def test_incoming_orders_count_toward_inventory_position():
    r = table(on_hand=20, on_order=100)           # ROP = 10*5 + 0 = 50; IP = 120 -> no order
    assert r.inventory_position == 120 and r.recommended_qty == 0


def test_reorder_triggers_and_rounds_to_pack():
    r = table(on_hand=30, on_order=0)             # ROP 50, S = min(50+70, 450)=120, IP 30 -> 90
    assert r.reorder_point == 50 and r.recommended_qty == 90 and r.status == "Reorder now"


def test_stocked_out_with_overdue_po_is_flagged():
    r = table(on_hand=0, on_order=100, next_delivery="2026-01-10")   # as_of is 12 Jan -> 2 days overdue
    assert r.days_overdue == 2 and r.status == "Stocked out - PO overdue"


def test_overstock_flag():
    r = table(on_hand=600, on_order=0)            # 60 days of cover > 45
    assert r.status == "Overstocked" and r.recommended_qty == 0
