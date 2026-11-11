import pandas as pd
from src.data_generator import generate
from src.data_loader import load_movements, load_data
from src.reconciliation import reconcile_movements

ORIG = "data/raw_original_unreconciled/"


def _inputs():
    p = pd.read_csv(ORIG + "product_master.csv")
    c = pd.read_csv(ORIG + "consumption_transactions.csv")
    return c, p


def test_same_seed_gives_identical_data():
    c, p = _inputs()
    a, b = generate(c, p, seed=7), generate(c, p, seed=7)
    pd.testing.assert_frame_equal(a["movements"], b["movements"])


def test_different_seed_changes_data():
    c, p = _inputs()
    assert not generate(c, p, seed=1)["procurement"].equals(generate(c, p, seed=2)["procurement"])


def test_every_batch_and_product_reconciles():
    d = load_data()
    prod, batch = reconcile_movements(d["products"], d["batches"], load_movements())
    assert (batch.difference == 0).all()
    assert (prod.status_check == "Reconciled").all()


def test_no_negative_stock_and_never_issue_more_than_requested():
    d = load_data()
    prod, _ = reconcile_movements(d["products"], d["batches"], load_movements())
    assert (prod.lowest_daily_balance >= 0).all()
    assert (d["consumption"].quantity_issued <= d["consumption"].quantity_requested).all()
    assert (d["batches"].remaining_qty >= 0).all()


def test_demand_is_preserved_from_original_file():
    d = load_data()
    c, _ = _inputs()
    assert d["consumption"].quantity_requested.sum() == c.quantity_issued.sum()


def test_fefo_issues_use_earliest_expiry_first():
    d = load_data(); mv = load_movements(); exp = d["batches"].set_index("batch_id").expiry_date
    pid = "P008"
    m = mv[mv.product_id == pid].sort_values(["movement_date"])
    for day, g in m.groupby("movement_date"):
        before = m[m.movement_date < day].groupby("batch_id").quantity.sum()
        today_receipts = g[g.movement_type.isin(["RECEIPT", "OPENING"])].groupby("batch_id").quantity.sum()
        avail = before.add(today_receipts, fill_value=0)
        avail = avail[avail > 0]
        used = g[g.movement_type == "ISSUE"].batch_id.unique()
        for b in used:
            others = [x for x in avail.index if x != b and exp[x] > day]
            # a batch may only be used if no other available batch expires earlier... unless that one is fully used today
            earlier = [x for x in others if exp[x] < exp[b] and x not in used]
            assert not earlier, f"{day}: used {b} while earlier-expiry {earlier} had stock"
