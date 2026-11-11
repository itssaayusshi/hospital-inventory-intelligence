import importlib
import sys
import pandas as pd
import pytest
from src.data_loader import load_data, load_movements, ROOT
from src.pipeline import (build_model, parse_tables, check_tables, DataError, Params, text_filter, waste_scenarios,
                          inventory_frame, unmet_demand, expired_history, open_orders, FILE_NAMES)
from tests.fakes import FakeSt, install_fake_plotly


@pytest.fixture(scope="module")
def tables():
    d = load_data(); d["movements"] = load_movements(); return d


@pytest.fixture(scope="module")
def model(tables):
    return build_model(tables)


# ---------------- pipeline
def test_model_numbers_match_standalone_scripts(model):
    assert model.store["stock_value_inr"] == pytest.approx(15102.592)
    assert model.store["expired_value_inr_in_period"] == pytest.approx(3637.2)
    assert len(model.reorder) == 10 and model.as_of == pd.Timestamp("2026-11-10")


def test_alerts_flag_overdue_catheter_first(model):
    a = model.alerts
    assert a.iloc[0].severity == "High" and a.iloc[0].product_id == "P007" and "overdue" in a.iloc[0].issue
    assert set(a.severity) <= {"High", "Medium"}


def test_parameters_change_results(tables):
    strict = build_model(tables, Params(service_levels={"Vital": 0.995, "Essential": 0.99, "Desirable": 0.97}))
    base = build_model(tables)
    assert (strict.reorder.safety_stock > base.reorder.safety_stock).all()


def test_helper_tables(model):
    assert len(inventory_frame(model)) == 10
    assert unmet_demand(model).units_short.sum() > 0
    assert expired_history(model).value_inr.sum() == pytest.approx(model.store["expired_value_inr_in_period"])
    o = open_orders(model); assert (o.days_overdue >= 0).all() and len(o) == 4


def test_text_filter():
    df = pd.DataFrame({"name": ["Gauze Roll", "Face Mask"], "n": [1, 2]})
    assert text_filter(df, "gauze").n.tolist() == [1] and len(text_filter(df, "")) == 2 and text_filter(df, "zzz").empty


def test_waste_scenarios_math(model):
    s = waste_scenarios(model, 0.5, 0.0, 0.2)
    assert s.simulated_saving_inr.iloc[0] == pytest.approx(3637.2 * 0.5) and s.simulated_saving_inr.iloc[1] == 0


# ---------------- upload path
def raw_frames():
    return {k: pd.read_csv(ROOT / "data" / "raw" / v) for k, v in FILE_NAMES.items()}


def test_upload_roundtrip_gives_same_results(tables, model):
    parsed = parse_tables(raw_frames())
    assert not (check_tables(parsed).query("severity == 'ERROR'")["count"] > 0).any()
    assert build_model(parsed).store["stock_value_inr"] == pytest.approx(model.store["stock_value_inr"])


def test_upload_missing_column_and_bad_dates_are_reported():
    f = raw_frames()
    f["batches"] = f["batches"].drop(columns=["expiry_date"])
    f["consumption"].loc[0, "transaction_date"] = "not a date"
    with pytest.raises(DataError) as e:
        parse_tables(f)
    assert "expiry_date" in str(e.value) and "unreadable dates" in str(e.value)


def test_upload_with_broken_ledger_is_blocked():
    f = raw_frames(); f["batches"].loc[0, "remaining_qty"] += 5
    rep = check_tables(parse_tables(f))
    assert rep[rep.check.str.contains("sum of movements")]["count"].iloc[0] == 1


# ---------------- every page executes (with stand-in streamlit)
def load_views(monkeypatch):
    fake = FakeSt()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    try:
        import plotly.express  # noqa: F401  (use real plotly if installed)
    except ImportError:
        install_fake_plotly(monkeypatch)
    for name in ["app.ui", "app.charts", "app.views"]:
        sys.modules.pop(name, None)
    return importlib.import_module("app.views"), fake


def test_all_eight_pages_run_and_show_key_numbers(monkeypatch, model):
    views, fake = load_views(monkeypatch)
    assert len(views.PAGES) == 8
    for name, page in views.PAGES.items():
        fake.log.clear()
        page(model)
        assert any(c[0] in ("dataframe", "plotly_chart", "metric") for c in fake.log), name
    fake.log.clear(); views.executive(model)
    metrics = {c[1]: c[2] for c in fake.log if c[0] == "metric"}
    assert metrics["Stock value"] == "₹15,103" and metrics["Fill rate"] == "99.4%"
    assert metrics["Overdue purchase orders"] == "1"


def test_pages_handle_filters_that_remove_everything(monkeypatch, model):
    views, fake = load_views(monkeypatch)
    fake.text_input = lambda label, value="", **k: "no-such-product"
    views.inventory(model); views.reorder(model)    # must not crash on empty tables
