import pandas as pd
import numpy as np

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def _atr(df: pd.DataFrame, period: int = 7) -> pd.Series:
    """Wilder ATR (EMA of True Range)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def supertrend(df: pd.DataFrame, period: int = 7, multiplier: float = 3.5) -> pd.DataFrame:
    """
    Returns DataFrame with:
      ST      - supertrend line
      ST_DIR  - +1 uptrend, -1 downtrend
    """
    hl2 = (df["High"] + df["Low"]) / 2.0
    atr = _atr(df, period)

    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    # Final bands
    final_upper = upperband.copy()
    final_lower = lowerband.copy()

    for i in range(1, len(df)):
        # if close prev <= final_upper prev, take min(current upper, final_upper prev)
        if df["Close"].iat[i-1] <= final_upper.iat[i-1]:
            final_upper.iat[i] = min(upperband.iat[i], final_upper.iat[i-1])
        else:
            final_upper.iat[i] = upperband.iat[i]
        # if close prev >= final_lower prev, take max(current lower, final_lower prev)
        if df["Close"].iat[i-1] >= final_lower.iat[i-1]:
            final_lower.iat[i] = max(lowerband.iat[i], final_lower.iat[i-1])
        else:
            final_lower.iat[i] = lowerband.iat[i]

    st = pd.Series(index=df.index, dtype="float64")
    dir_ = pd.Series(index=df.index, dtype="int64")

    for i in range(len(df)):
        if df["Close"].iat[i] > final_upper.iat[i]:
            st.iat[i] = final_lower.iat[i]
            dir_.iat[i] = 1  # uptrend
        elif df["Close"].iat[i] < final_lower.iat[i]:
            st.iat[i] = final_upper.iat[i]
            dir_.iat[i] = -1 # downtrend
        else:
            # keep prior direction if inside bands
            if i == 0:
                st.iat[i] = final_lower.iat[i]
                dir_.iat[i] = 1
            else:
                st.iat[i] = st.iat[i-1]
                dir_.iat[i] = dir_.iat[i-1]

    return pd.DataFrame({"ST": st, "ST_DIR": dir_})
