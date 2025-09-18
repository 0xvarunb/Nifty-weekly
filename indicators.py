import pandas as pd
import numpy as np

def ema(series: pd.Series, period: int) -> pd.Series:
    # Ensure numeric
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    return s.ewm(span=period, adjust=False).mean().astype("float64")

def _atr(df: pd.DataFrame, period: int = 7) -> pd.Series:
    """Wilder-style ATR via EMA of True Range."""
    high = pd.to_numeric(df["High"], errors="coerce").astype("float64")
    low  = pd.to_numeric(df["Low"],  errors="coerce").astype("float64")
    close= pd.to_numeric(df["Close"],errors="coerce").astype("float64")
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean().astype("float64")
    return atr

def supertrend(df: pd.DataFrame, period: int = 7, multiplier: float = 3.5) -> pd.DataFrame:
    """
    Vectorized, stable Supertrend.
    Returns:
      ST      - float64 supertrend line
      ST_DIR  - int64 (+1 uptrend, -1 downtrend)
    """
    high = pd.to_numeric(df["High"], errors="coerce").astype("float64").to_numpy()
    low  = pd.to_numeric(df["Low"],  errors="coerce").astype("float64").to_numpy()
    close= pd.to_numeric(df["Close"],errors="coerce").astype("float64").to_numpy()

    atr = _atr(df, period).to_numpy()
    hl2 = (high + low) / 2.0

    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    n = len(close)
    final_upper = np.copy(upperband)
    final_lower = np.copy(lowerband)

    # carry-forward min/max refinement
    for i in range(1, n):
        final_upper[i] = min(upperband[i], final_upper[i-1]) if close[i-1] <= final_upper[i-1] else upperband[i]
        final_lower[i] = max(lowerband[i], final_lower[i-1]) if close[i-1] >= final_lower[i-1] else lowerband[i]

    st = np.zeros(n, dtype=np.float64)
    direction = np.zeros(n, dtype=np.int64)

    # init
    st[0] = final_lower[0]
    direction[0] = 1

    for i in range(1, n):
        if close[i] > final_upper[i]:
            direction[i] = 1
            st[i] = final_lower[i]
        elif close[i] < final_lower[i]:
            direction[i] = -1
            st[i] = final_upper[i]
        else:
            direction[i] = direction[i-1]
            st[i] = st[i-1]

    return pd.DataFrame({"ST": st, "ST_DIR": direction}, index=df.index)
