"""Simple, explainable demand forecasting + a backtest to choose between methods.

We forecast DAILY demand per product. All methods use only data available up to the forecast origin
(no peeking into the future), so the backtest is an honest test.

Methods
-------
- Naive:           tomorrow = today.
- Moving average:  mean of the last N days.
- SES:             simple exponential smoothing. level_t = a*y_t + (1-a)*level_{t-1}; forecast = last level.
- MA28 x weekday:  4-week average (balanced across weekdays) multiplied by a weekday index learned from history.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def moving_average(y, window):
    y = np.asarray(y, float)
    return float(y[-window:].mean())


def ses_level(y, alpha):
    """Final smoothed level. Starts at the first observation."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    level = float(y[0])
    for v in y[1:]:
        level = alpha * float(v) + (1 - alpha) * level
    return level


def weekday_index(dates, y):
    """Mean demand of each weekday / overall mean (1.0 = average day). Index 0 = Monday."""
    s = pd.Series(np.asarray(y, float), index=pd.DatetimeIndex(dates))
    idx = (s.groupby(s.index.dayofweek).mean() / s.mean()).reindex(range(7))
    return idx.fillna(1.0)


METHODS = {
    "Naive (yesterday)": lambda y, d, f: np.repeat(float(y[-1]), len(f)),
    "Moving avg 7": lambda y, d, f: np.repeat(moving_average(y, 7), len(f)),
    "Moving avg 28": lambda y, d, f: np.repeat(moving_average(y, 28), len(f)),
    "SES a=0.1": lambda y, d, f: np.repeat(ses_level(y, 0.1), len(f)),
    "SES a=0.3": lambda y, d, f: np.repeat(ses_level(y, 0.3), len(f)),
    "MA28 x weekday": lambda y, d, f: moving_average(y, 28) * weekday_index(d, y).reindex(pd.DatetimeIndex(f).dayofweek).to_numpy(),
}
DEFAULT_METHOD = "MA28 x weekday"


def forecast(y, dates, horizon=14, method=DEFAULT_METHOD):
    """Forecast the next `horizon` days. Returns a Series indexed by future dates."""
    dates = pd.DatetimeIndex(dates)
    future = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.Series(METHODS[method](np.asarray(y, float), dates, future), index=future, name="forecast")


def backtest_series(y, dates, horizon=7, min_history=56, methods=METHODS):
    """Rolling-origin test for ONE product. At every origin t, forecast the next `horizon` days
    using data up to t only, then compare with what really happened. Returns errors per method."""
    y = np.asarray(y, float); dates = pd.DatetimeIndex(dates)
    rows = {m: {"abs_err": 0.0, "err": 0.0, "actual": 0.0, "n": 0} for m in methods}
    for t in range(min_history - 1, len(y) - horizon):
        future = dates[t + 1: t + 1 + horizon]; actual = y[t + 1: t + 1 + horizon]
        for m, fn in methods.items():
            f = fn(y[: t + 1], dates[: t + 1], future)
            rows[m]["abs_err"] += np.abs(f - actual).sum(); rows[m]["err"] += (f - actual).sum()
            rows[m]["actual"] += actual.sum(); rows[m]["n"] += 1
    out = pd.DataFrame(rows).T
    out["wape"] = out.abs_err / out.actual           # weighted absolute % error (lower = better)
    out["bias"] = out.err / out.actual               # + means over-forecasting
    return out[["n", "wape", "bias"]].rename(columns={"n": "origins"})


def backtest(consumption, horizon=7, min_history=56):
    """Backtest every product. Uses requested demand (true demand), not issued."""
    res = []
    for pid, g in consumption.sort_values("transaction_date").groupby("product_id"):
        r = backtest_series(g.quantity_requested.to_numpy(), g.transaction_date, horizon, min_history)
        res.append(r.assign(product_id=pid).reset_index(names="method"))
    return pd.concat(res, ignore_index=True)


def best_method(bt: pd.DataFrame) -> pd.DataFrame:
    """Average WAPE of each method across products (simple, equal weight), best first."""
    return bt.groupby("method").agg(avg_wape=("wape", "mean"), avg_bias=("bias", "mean")).sort_values("avg_wape").reset_index()
