"""Hospital Inventory Intelligence dashboard.   Run from the project folder:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `import src` works when launched by streamlit

import pandas as pd
import streamlit as st
from src.data_loader import load_data, load_movements
from src.pipeline import DataError, FILE_NAMES, Params, build_model, check_tables, parse_tables
from src.reorder import SERVICE_LEVEL
from src.forecasting import METHODS, DEFAULT_METHOD
from app.views import PAGES

st.set_page_config(page_title="Hospital Inventory Intelligence", layout="wide")


@st.cache_data(show_spinner=False)
def default_tables():
    d = load_data(); d["movements"] = load_movements()
    return d


@st.cache_data(show_spinner="Calculating...")
def get_model(tables, review_days, max_cover_days, excess_days, sl_v, sl_e, sl_d, method):
    p = Params(review_days, max_cover_days, excess_days, {"Vital": sl_v, "Essential": sl_e, "Desirable": sl_d}, method)
    return build_model(tables, p)


st.title("Hospital Inventory Intelligence & Waste Reduction")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Page", list(PAGES))

# ---------------- data source: bundled synthetic data, or the user's own CSVs
tables = default_tables()
with st.sidebar.expander("Use your own CSV data"):
    st.caption("Upload all five files with the same columns as the bundled data: " + ", ".join(FILE_NAMES.values()))
    files = st.file_uploader("CSV files", type="csv", accept_multiple_files=True)
    if files:
        by_file = {v: k for k, v in FILE_NAMES.items()}
        frames = {by_file[f.name]: pd.read_csv(f) for f in files if f.name in by_file}
        if len(frames) < len(FILE_NAMES):
            st.info("Waiting for: " + ", ".join(FILE_NAMES[k] for k in FILE_NAMES if k not in frames))
        else:
            try:
                parsed = parse_tables(frames)
                rep = check_tables(parsed)
                if (rep[rep.severity == "ERROR"]["count"] > 0).any():
                    st.error("Uploaded data has errors, so the bundled data is still in use.")
                    st.dataframe(rep[(rep.severity == "ERROR") & (rep["count"] > 0)])
                else:
                    tables = parsed
                    st.success("Using your uploaded data.")
            except DataError as e:
                st.error(str(e))

# ---------------- planning assumptions
with st.sidebar.expander("Planning assumptions"):
    review = st.slider("Review period (days)", 1, 21, 7)
    cover = st.slider("Max days of cover", 20, 90, 45)
    sv = st.slider("Service level: Vital", 0.90, 0.995, SERVICE_LEVEL["Vital"], 0.005)
    se = st.slider("Service level: Essential", 0.90, 0.995, SERVICE_LEVEL["Essential"], 0.005)
    sd = st.slider("Service level: Desirable", 0.80, 0.995, SERVICE_LEVEL["Desirable"], 0.005)
    method = st.selectbox("Forecast method", list(METHODS), index=list(METHODS).index(DEFAULT_METHOD))

model = get_model(tables, review, cover, cover, sv, se, sd, method)
PAGES[page](model)
