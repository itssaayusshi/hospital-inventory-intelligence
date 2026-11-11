"""Stand-ins for streamlit (and plotly when it is not installed) so every dashboard page can be executed in tests.
They accept the calls our pages make and record them. They do NOT prove the real libraries render correctly."""
import sys
import types
import pandas as pd


class FakeFigure:
    def __init__(self, *a, **k): self.traces = []
    def add_trace(self, t, *a, **k): self.traces.append(t); return self
    def add_scatter(self, *a, **k): return self
    def update_layout(self, *a, **k): return self
    def update_traces(self, *a, **k): return self


class FakeSt:
    """Plays the role of the `streamlit` module, columns, sidebar, tabs and expanders at once."""
    def __init__(self, log=None):
        self.log = log if log is not None else []
        self.session_state = {}
        self.sidebar = FakeSt(self.log) if log is None else self
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name):                       # any display call we don't model: record and return a stub
        def call(*a, **k):
            self.log.append((name, a, k)); return FakeSt(self.log)
        return call
    # layout
    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec); return [FakeSt(self.log) for _ in range(n)]
    def tabs(self, labels): return [FakeSt(self.log) for _ in labels]
    def expander(self, *a, **k): return FakeSt(self.log)
    # inputs return their defaults
    def selectbox(self, label, options, index=0, **k): self.log.append(("selectbox", label, k)); return list(options)[index]
    def radio(self, label, options, index=0, **k): return list(options)[index]
    def multiselect(self, label, options, default=None, **k): return list(default) if default is not None else []
    def slider(self, label, min_value=0, max_value=100, value=None, *a, **k): return min_value if value is None else value
    def number_input(self, label, min_value=None, max_value=None, value=0, **k): return value
    def text_input(self, label, value="", **k): return value
    def checkbox(self, label, value=False, **k): return value
    def file_uploader(self, *a, **k): return [] if k.get("accept_multiple_files") else None
    def button(self, *a, **k): return False
    def download_button(self, label, data, *a, **k):
        assert isinstance(data, (bytes, str)); self.log.append(("download_button", label, len(data))); return False
    # display
    def metric(self, label, value, *a, **k):
        assert isinstance(label, str) and str(value) != "nan", f"bad metric {label}={value}"
        self.log.append(("metric", label, str(value))); return self
    def dataframe(self, df, *a, **k):
        assert isinstance(df, pd.DataFrame), f"dataframe got {type(df)}"
        self.log.append(("dataframe", df.shape)); return self
    def plotly_chart(self, fig, *a, **k):
        assert fig is not None; self.log.append(("plotly_chart",)); return self
    def cache_data(self, *a, **k):
        if a and callable(a[0]): return a[0]
        return lambda f: f


def install_fake_plotly(monkeypatch):
    """Only used when real plotly is missing."""
    px = types.ModuleType("plotly.express")
    for fn in ["bar", "line", "pie", "scatter", "histogram", "box", "area"]:
        setattr(px, fn, lambda *a, **k: FakeFigure())
    go = types.ModuleType("plotly.graph_objects")
    go.Figure = FakeFigure
    for cls in ["Bar", "Scatter", "Heatmap"]:
        setattr(go, cls, lambda *a, **k: types.SimpleNamespace(args=a, kwargs=k))
    pkg = types.ModuleType("plotly"); pkg.express = px; pkg.graph_objects = go
    for name, mod in [("plotly", pkg), ("plotly.express", px), ("plotly.graph_objects", go)]:
        monkeypatch.setitem(sys.modules, name, mod)
