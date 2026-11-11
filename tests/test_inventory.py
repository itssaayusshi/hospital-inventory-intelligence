import pandas as pd
import pytest
from src.abc_ved import abc_classification, abc_ved_matrix
from src.expiry import expiry_status, exposure_by_window, fefo_issue_plan
from src.kpis import store_kpis, inventory_turnover


def prods(vals):
    return pd.DataFrame({"product_id": list(vals), "product_name": list(vals), "unit_cost_inr": 1.0,
                         "criticality": ["Vital", "Essential", "Desirable"][: len(vals)]})


def test_abc_cutoffs_on_exact_70_20_10_split():
    demand = pd.Series({"X": 700 / 365, "Y": 200 / 365, "Z": 100 / 365})
    r = abc_classification(prods(demand.index), demand).set_index("product_id")
    assert r.abc.to_dict() == {"X": "A", "Y": "B", "Z": "C"}      # exactly on the line goes to the next class
    assert r.abc_ved.to_dict() == {"X": "AV", "Y": "BE", "Z": "CD"}
    assert abc_ved_matrix(r.reset_index()).loc["A", "Vital"] == 1


def test_abc_missing_demand_raises():
    with pytest.raises(ValueError):
        abc_classification(prods(["X", "Y"]), pd.Series({"X": 1.0}))


def batches():
    return pd.DataFrame({
        "batch_id": ["b1", "b2", "b3", "b4"], "product_id": ["A", "A", "A", "A"],
        "expiry_date": pd.to_datetime(["2026-01-21", "2026-03-01", "2026-01-10", "2026-02-01"]),
        "remaining_qty": [100, 500, 50, 0], "unit_cost_inr": 2.0})


def test_expiry_status_drops_empty_batches_and_ranks_fefo():
    s = expiry_status(batches(), "2026-01-01")
    assert "b4" not in s.batch_id.values
    assert s.batch_id.tolist() == ["b3", "b1", "b2"] and s.fefo_rank.tolist() == [1, 2, 3]
    assert s.days_to_expiry.tolist() == [9, 20, 59]
    assert s.bucket.tolist() == ["0-30 days", "0-30 days", "31-60 days"]


def test_at_risk_math_is_fefo_aware():
    # demand 5/day: by Jan 10 (9 days) 45 units sold -> b3 (50) has 5 at risk;
    # by Jan 21 (20 days) 100 sold; cumulative stock 150 -> 50 over -> b1 has 50 at risk; b2: cum 650 - 295 = 355 -> capped at 500 (355)
    s = expiry_status(batches(), "2026-01-01", pd.Series({"A": 5.0})).set_index("batch_id")
    assert s.at_risk_qty.to_dict() == {"b3": 5, "b1": 50, "b2": 355}
    assert s.at_risk_value_inr.sum() == (5 + 50 + 355) * 2.0


def test_exposure_windows_are_cumulative():
    e = exposure_by_window(expiry_status(batches(), "2026-01-01"))
    assert e.units.tolist() == [150, 650, 650]   # b2 expires in 59 days, so it is inside the 60-day window


def test_fefo_plan_picks_earliest_and_skips_expired():
    plan = fefo_issue_plan(batches(), "A", 120, "2026-01-11")  # b3 expired yesterday -> skipped
    assert plan.batch_id.tolist() == ["b1", "b2"] and plan.pick_qty.tolist() == [100, 20]
    assert plan.shortfall.eq(0).all()


def test_fefo_plan_reports_shortfall_and_rejects_negative():
    plan = fefo_issue_plan(batches(), "A", 10_000, "2026-01-01")
    assert plan.pick_qty.sum() == 650 and plan.shortfall.iloc[0] == 9350
    with pytest.raises(ValueError):
        fefo_issue_plan(batches(), "A", -1, "2026-01-01")


def test_turnover_and_store_kpis_use_real_period_length():
    b = pd.DataFrame({"batch_id": ["b"], "unit_cost_inr": [10.0]})
    mv = pd.DataFrame({"movement_id": [1, 2], "movement_date": pd.to_datetime(["2026-01-01", "2026-01-10"]),
                       "product_id": "A", "batch_id": "b", "movement_type": ["OPENING", "ISSUE"], "quantity": [100, -50]})
    t = inventory_turnover(mv, b).iloc[0]
    # days 1..9 hold 100 units (1000), day 10 holds 50 (500) -> avg 950; issued 500; annualised x 365/10
    assert t.avg_inventory_value_inr == pytest.approx(950)
    assert t.annualised_turnover == pytest.approx(500 * 36.5 / 950)
    k = pd.DataFrame({"stock_value_inr": [1.0], "expired_value_inr": [0.0], "expiring_90d_value_inr": [0.0],
                      "at_risk_expiry_value_inr": [0.0], "excess_value_inr": [0.0], "stock_status": ["OK"],
                      "issued_value_inr": [500.0], "avg_inventory_value_inr": [950.0]})
    c = pd.DataFrame({"transaction_date": pd.date_range("2026-01-01", periods=10), "quantity_issued": 5, "quantity_requested": 5})
    assert store_kpis(k, c)["store_turnover"] == pytest.approx(500 * 36.5 / 950)
