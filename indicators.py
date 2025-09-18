import pandas as pd
import numpy as np

def ema(series: pd.Series, period: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    return s.ewm(span=period, adjust=False).mean().astype("float64")

def _atr_numpy(df: pd.DataFrame, period: int = 7) -> pd.Series:
    """Wilder-style ATR via EMA of True Range, fully NumPy (robust)."""
    h = pd.to_numeric(df["High"], errors="coerce").astype("float64").to_numpy()
    l = pd.to_numeric(df["Low"],  errors="coerce").astype("float64").to_numpy()
    c = pd.to_numeric(df["Close"],errors="coerce").astype("float64").to_numpy()
    pc = np.r_[np.nan, c[:-1]]

    tr1 = np.abs(h - l)
    tr2 = np.abs(h - pc)
    tr3 = np.abs(l - pc)
    tr = np.nanmax(np.vstack([tr1, tr2, tr3]), axis=0)

    alpha = 1.0 / float(period)
    atr = np.empty_like(tr, dtype=np.float64)
    atr[:] = np.nan

    # initialize at first non-nan
    start_idx = None
    for i in range(len(tr)):
        if not np.isnan(tr[i]):
            atr[i] = tr[i]
            start_idx = i + 1
            break
    if start_idx is None:
        return pd.Series(atr, index=df.index, dtype="float64")

    prev = atr[start_idx - 1]
    for i in range(start_idx, len(tr)):
        val = tr[i]
        if np.isnan(val):
            atr[i] = prev
        else:
            prev = alpha * val + (1 - alpha) * prev
            atr[i] = prev
    return pd.Series(atr, index=df.index, dtype="float64")

def supertrend(df: pd.DataFrame, period: int = 7, multiplier: float = 3.5) -> pd.DataFrame:
    """
    Vectorized, stable Supertrend.
    Returns:
      ST      - float64 supertrend line
      ST_DIR  - int64 (+1 uptrend, -1 downtrend)
    """
    h = pd.to_numeric(df["High"], errors="coerce").astype("float64").to_numpy()
    l = pd.to_numeric(df["Low"],  errors="coerce").astype("float64").to_numpy()
    c = pd.to_numeric(df["Close"],errors="coerce").astype("float64").to_numpy()

    atr = _atr_numpy(df, period).to_numpy()
    hl2 = (h + l) / 2.0

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    n = len(c)
    f_upper = np.copy(upper)
    f_lower = np.copy(lower)

    for i in range(1, n):
        f_upper[i] = min(upper[i], f_upper[i-1]) if c[i-1] <= f_upper[i-1] else upper[i]
        f_lower[i] = max(lower[i], f_lower[i-1]) if c[i-1] >= f_lower[i-1] else lower[i]

    st = np.zeros(n, dtype=np.float64)
    dir_ = np.zeros(n, dtype=np.int64)
    st[0] = f_lower[0]
    dir_[0] = 1

    for i in range(1, n):
        if c[i] > f_upper[i]:
            dir_[i] = 1
            st[i] = f_lower[i]
        elif c[i] < f_lower[i]:
            dir_[i] = -1
            st[i] = f_upper[i]
        else:
            dir_[i] = dir_[i-1]
            st[i] = st[i-1]

    return pd.DataFrame({"ST": st, "ST_DIR": dir_}, index=df.index)
