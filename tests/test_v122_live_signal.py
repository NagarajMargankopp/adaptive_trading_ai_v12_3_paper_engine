import numpy as np
from app.v12_live_signal import percentile

def test_percentile_empty():
    assert percentile(np.array([]), 1.0) == 0.0

def test_percentile_sorted():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    assert percentile(ref, 2.0) == 0.5
    assert percentile(ref, 4.0) == 1.0


def test_timestamp_normalization_contract():
    import pandas as pd
    x = pd.DataFrame({"timestamp": [pd.Timestamp("2026-09-28 17:15:00", tz="UTC")]})
    ts = pd.Timestamp("2026-09-28 17:20:00+00:00")
    latest = {"timestamp": ts}
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    x = x[x["timestamp"] != ts]
    x = pd.concat([x, pd.DataFrame([latest])], ignore_index=True)
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    out = x.sort_values("timestamp")
    assert out["timestamp"].dtype.tz is not None
    assert out.iloc[-1]["timestamp"] == ts
