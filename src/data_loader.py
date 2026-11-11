from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw'


def load_data():
    """Load the four core tables (kept as 4 keys so older code that unpacks them still works)."""
    return {
        'products': pd.read_csv(DATA / 'product_master.csv'),
        'batches': pd.read_csv(DATA / 'inventory_batches.csv', parse_dates=['purchase_date', 'received_date', 'expiry_date']),
        'consumption': pd.read_csv(DATA / 'consumption_transactions.csv', parse_dates=['transaction_date']),
        'procurement': pd.read_csv(DATA / 'procurement_transactions.csv',
                                   parse_dates=['order_date', 'promised_delivery_date', 'delivery_date']),
    }


def load_movements():
    """Stock movement ledger: OPENING, RECEIPT, ISSUE, EXPIRY_WRITE_OFF (issues/write-offs are negative)."""
    return pd.read_csv(DATA / 'stock_movements.csv', parse_dates=['movement_date'])
