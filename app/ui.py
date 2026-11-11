"""Small display helpers (no business logic)."""
import pandas as pd

ICONS = [("Stocked out", "🔴"), ("Reorder", "🟠"), ("Watch", "🟡"), ("Overstocked", "🟣"), ("OK", "🟢"),
         ("High", "🔴"), ("Medium", "🟠"), ("Near expiry", "🟠")]


def inr(x) -> str:
    return f"₹{x:,.0f}"


def icon(text: str) -> str:
    for key, ic in ICONS:
        if str(text).startswith(key):
            return f"{ic} {text}"
    return str(text)


def pretty(df: pd.DataFrame, rename: dict | None = None, decimals: int = 1, icon_cols=()) -> pd.DataFrame:
    """Display copy: rounded numbers, friendly column names, status icons."""
    x = df.copy()
    for c in icon_cols:
        if c in x:
            x[c] = x[c].map(icon)
    num = x.select_dtypes(include="float").columns
    x[num] = x[num].round(decimals)
    for c in x.select_dtypes(include=["datetime"]).columns:
        x[c] = x[c].dt.strftime("%Y-%m-%d")
    return x.rename(columns=rename or {})


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
