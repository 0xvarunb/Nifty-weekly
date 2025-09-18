import pandas as pd
import numpy as np

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def _atr(df: pd.DataFrame, period: int = 7) -> pd.Series:
    """Wilder-style ATR using EMA of True Range."""
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder’s smoothing == EMA with alpha=1/period
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def supertrend(df: pd.DataFrame, period: int = 7, multiplier: float = 3.5) -> pd.DataFrame:
    """
    Vectorized Supertrend.
    Returns:
      ST      : supertrend line
      ST_DIR  : +1 uptrend, -1 downtrend
    """
    # Ensure float dtypes
    high = df["High"].astype(float).to_numpy()
    low  = df["Low"].astype(float).to_numpy()
    close= df["Close"].astype(float).to_numpy()

    atr = _atr(df, period).to_numpy()
    hl2 = (high + low) / 2.0

    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    n = len(df)
    final_upper = np.copy(upperband)
    final_lower = np.copy(lowerband)

    # Refine bands (carry-forward min/max logic)
    for i in range(1, n):
        if close[i-1] <= final_upper[i-1]:
            final_upper[i] = min(upperband[i], final_upper[i-1])
        else:
            final_upper[i] = upperband[i]

        if close[i-1] >= final_lower[i-1]:
            final_lower[i] = max(lowerband[i], final_lower[i-1])
        else:
            final_lower[i] = lowerband[i]

    st = np.zeros(n, dtype=float)
    direction = np.zeros(n, dtype=int)

    # Initialize
    st[0] = final_lower[0]
    direction[0] = 1  # start as up unless broken in next step

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
