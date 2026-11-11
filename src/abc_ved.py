"""ABC-VED classification.

ABC  = how much MONEY a product consumes   (A: top ~70% of annual value, B: next ~20%, C: last ~10%)
VED  = how CRITICAL it is clinically        (Vital / Essential / Desirable, from the product master)

Rule for ABC cut-offs: sort by annual value (largest first) and look at the cumulative share
BEFORE adding the item. If that share is below 70% the item is A, below 90% it is B, else C.
(So the product that crosses the 70% line stays in A.)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

A_CUTOFF, B_CUTOFF = 0.70, 0.90

# Convention used in many hospital stores; change it here if your hospital uses another one.
PRIORITY = {"AV": 1, "BV": 1, "AE": 1, "CV": 2, "BE": 2, "AD": 2, "CE": 3, "BD": 3, "CD": 3}
PRIORITY_LABEL = {1: "Tight control", 2: "Regular review", 3: "Simple / bulk control"}
VED_CODE = {"Vital": "V", "Essential": "E", "Desirable": "D"}


def abc_classification(products: pd.DataFrame, avg_daily_demand: pd.Series,
                       a_cutoff: float = A_CUTOFF, b_cutoff: float = B_CUTOFF) -> pd.DataFrame:
    """avg_daily_demand: Series indexed by product_id. Annual value = daily demand x 365 x unit cost."""
    df = products[["product_id", "product_name", "unit_cost_inr", "criticality"]].copy()
    df["avg_daily_demand"] = df.product_id.map(avg_daily_demand)
    if df.avg_daily_demand.isna().any():
        raise ValueError("Missing demand for: " + ", ".join(df[df.avg_daily_demand.isna()].product_id))
    df["annual_value_inr"] = df.avg_daily_demand * 365 * df.unit_cost_inr
    total = df.annual_value_inr.sum()
    if total <= 0:
        raise ValueError("Total annual value is zero; cannot classify.")
    df = df.sort_values("annual_value_inr", ascending=False).reset_index(drop=True)
    df["value_share"] = df.annual_value_inr / total
    df["cum_share_after"] = df.value_share.cumsum()
    before = (df.cum_share_after - df.value_share).round(9)   # round: avoids 0.8999999 vs 0.9 float noise
    df["abc"] = np.where(before < a_cutoff, "A", np.where(before < b_cutoff, "B", "C"))
    df["ved"] = df.criticality.map(VED_CODE)
    df["abc_ved"] = df.abc + df.ved
    df["priority"] = df.abc_ved.map(PRIORITY)
    df["priority_label"] = df.priority.map(PRIORITY_LABEL)
    return df


def abc_ved_matrix(classified: pd.DataFrame) -> pd.DataFrame:
    """3x3 count table: rows A/B/C, columns Vital/Essential/Desirable."""
    m = pd.crosstab(classified.abc, classified.criticality)
    return m.reindex(index=["A", "B", "C"], columns=["Vital", "Essential", "Desirable"], fill_value=0)
