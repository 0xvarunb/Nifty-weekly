#!/usr/bin/env python3
"""
NIFTY hourly scanner for Supertrend + EMA(20)

Signals (evaluated on latest completed hourly candle, IST):
- Bullish  = Close > Supertrend AND Close > EMA20
- Bearish  = Close < Supertrend AND Close < EMA20
Alert triggers only on a NEW flip into bull/bear.

Outputs JSON and (optionally) sends a Telegram alert with:
- Side, Close, Supertrend, EMA20
- Suggested option legs (ATM sell + OTM hedge)
- Next expiry (Tue-based weekly; monthly = last Tue)

Extras:
- `--ping` sends a Telegram test message and exits.
- `HEARTBEAT=1` sends a short daily "alive" ping (used by heartbeat workflow).
"""
from __future__ import annotations

import os, sys, json, math, traceback, calendar, argparse
from datetime import datetime, timedelta, timezone, time as dtime, date
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import requests
import yfinance as yf

# ---------------- Config ----------------
TICKER            = os.getenv("NIFTY_TICKER", "^NSEI")   # Yahoo NIFTY 50 index
INTERVAL          = "60m"
LOOKBACK_DAYS     = int(os.getenv("LOOKBACK_DAYS", "60"))

EMA_PERIOD        = int(os.getenv("EMA_PERIOD", "20"))
ST_ATR_PERIOD     = int(os.getenv("ST_ATR_PERIOD", "7"))
ST_MULTIPLIER     = float(os.getenv("ST_MULTIPLIER", "3.5"))

OTM_GAP_POINTS    = int(os.getenv("OTM_GAP_POINTS", "100"))   # hedge distance suggestion
ATM_STEP_POINTS   = int(os.getenv("ATM_STEP_POINTS", "50"))    # NIFTY=50, BANKNIFTY=100

# Expiry weekday (NSE change: Tuesday). Accepts strings like "TUE" or ints 0..6 (Mon..Sun)
EXPIRY_WEEKDAY    = os.getenv("EXPIRY_WEEKDAY", "TUE")
LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO")

# Heartbeat toggle (optional)
HEARTBEAT         = os.getenv("HEARTBEAT", "0") == "1"

# Telegram
TG_TOKEN          = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID        = os.getenv("TELEGRAM_CHAT_ID", "")
SEND_TELEGRAM     = bool(TG_TOKEN and TG_CHAT_ID)

IST = ZoneInfo("Asia/Kolkata")

def log(msg: str):
    prefix = f"[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}] " if LOG_LEVEL.upper()=="DEBUG" else ""
    print(prefix + str(msg), flush=True)

# ---------------- Expiry helpers ----------------
def _weekday_to_int(w: str | int) -> int:
    if isinstance(w, int):
        return max(0, min(6, w))
    m = {
        "mon":0,"monday":0,"0":0,
        "tue":1,"tuesday":1,"1":1,
        "wed":2,"wednesday":2,"2":2,
        "thu":3,"thursday":3,"3":3,
        "fri":4,"friday":4,"4":4,
        "sat":5,"saturday":5,"5":5,
        "sun":6,"sunday":6,"6":6
    }
    return m.get(str(w).strip().lower(), 1)

EXPIRY_WD_INT = _weekday_to_int(EXPIRY_WEEKDAY)

def round_to_step(x: float, step: int) -> int:
    return int(round(float(x)/float(step))*step)

def last_weekday_of_month(y: int, m: int, weekday_int: int) -> date:
    last_dom = calendar.monthrange(y, m)[1]
    d = date(y, m, last_dom)
    return d - timedelta(days=(d.weekday() - weekday_int) % 7)

def is_monthly_expiry(d: date, weekday_int: int) -> bool:
    return d == last_weekday_of_month(d.year, d.month, weekday_int)

def next_weekly_expiry_ist(now_ist: datetime) -> tuple[str, str]:
    dow = now_ist.weekday()
    days_ahead = (EXPIRY_WD_INT - dow) % 7
    candidate = (now_ist + timedelta(days=days_ahead)).date()
    if dow == EXPIRY_WD_INT and now_ist.time() > dtime(15,30):
        candidate = candidate + timedelta(days=7)
    exp_type = "Monthly" if is_monthly_expiry(candidate, EXPIRY_WD_INT) else "Weekly"
    return str(candidate), exp_type

# ---------------- Option legs ----------------
def suggested_option_legs(side: str, spot: float, now_ist: datetime) -> dict:
    atm = round_to_step(spot, ATM_STEP_POINTS)
    if side=="BULL":
        legs = {"sell": f"{atm} PE (ATM)", "buy": f"{max(ATM_STEP_POINTS, atm-OTM_GAP_POINTS)} PE (hedge ~{OTM_GAP_POINTS})"}
    else:
        legs = {"sell": f"{atm} CE (ATM)", "buy": f"{atm+OTM_GAP_POINTS} CE (hedge ~{OTM_GAP_POINTS})"}
    expiry_str, exp_type = next_weekly_expiry_ist(now_ist)
    legs["expiry"], legs["expiry_type"] = expiry_str, exp_type
    return legs

# ---------------- Data utils ----------------
def _force_float_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df

def _flatten_if_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    # try select by ticker on any level
    try:
        for lvl in range(df.columns.nlevels):
            if TICKER in df.columns.get_level_values(lvl):
                return df.xs(TICKER, axis=1, level=lvl, drop_level=True)
    except Exception:
        pass
    # fallback to first key on last level
    try:
        key = df.columns.get_level_values(-1)[0]
        return df.xs(key, axis=1, level=-1, drop_level=True)
    except Exception:
        # ultimate flatten
        df.columns = ["|".join([str(x) for x in tup if x is not None]) for tup in df.columns.to_list()]
        ren = {}
        for col in df.columns:
            low = col.lower()
            if "open" in low and "Open" not in ren.values(): ren[col]="Open"
            elif "high" in low and "High" not in ren.values(): ren[col]="High"
            elif "low" in low and "Low" not in ren.values(): ren[col]="Low"
            elif "adj" in low and "close" in low and "Adj Close" not in ren.values(): ren[col]="Adj Close"
            elif "close" in low and "Close" not in ren.values(): ren[col]="Close"
            elif "volume" in low and "Volume" not in ren.values(): ren[col]="Volume"
        return df.rename(columns=ren)

def fetch_hourly() -> pd.DataFrame:
    try:
        df = yf.Ticker(TICKER).history(period=f"{LOOKBACK_DAYS}d", interval=INTERVAL, auto_adjust=False)
    except Exception:
        df = None
    if df is None or df.empty:
        end = datetime.now(timezone.utc); start = end - timedelta(days=LOOKBACK_DAYS)
        df = yf.download(TICKER, start=start, end=end, interval=INTERVAL, auto_adjust=False, progress=False, group_by="column")
    if df is None or df.empty:
        raise RuntimeError("No data returned from yfinance.")
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    df = df.tz_convert(IST)
    df = _flatten_if_multiindex(df)
    for n in ["Open","High","Low","Close"]:
        if n not in df.columns:
            raise RuntimeError(f"Missing '{n}' after fetch; cols={list(df.columns)}")
    return _force_float_cols(df, ["Open","High","Low","Close","Adj Close","Volume"])

# ---------------- Indicators ----------------
def ema(series: pd.Series, period: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    return s.ewm(span=period, adjust=False).mean().astype("float64")

def _atr_numpy(df: pd.DataFrame, period: int = 7) -> pd.Series:
    h = pd.to_numeric(df["High"], errors="coerce").astype("float64").to_numpy()
    l = pd.to_numeric(df["Low"],  errors="coerce").astype("float64").to_numpy()
    c = pd.to_numeric(df["Close"],errors="coerce").astype("float64").to_numpy()
    pc = np.r_[np.nan, c[:-1]]
    tr1 = np.abs(h-l); tr2 = np.abs(h-pc); tr3 = np.abs(l-pc)
    tr = np.nanmax(np.vstack([tr1,tr2,tr3]), axis=0)
    alpha = 1.0/float(period)
    atr = np.empty_like(tr, dtype=np.float64); atr[:] = np.nan
    start_idx = None
    for i in range(len(tr)):
        if not np.isnan(tr[i]):
            atr[i] = tr[i]; start_idx = i+1; break
    if start_idx is None:
        return pd.Series(atr, index=df.index, dtype="float64")
    prev = atr[start_idx-1]
    for i in range(start_idx, len(tr)):
        val = tr[i]
        prev = prev if np.isnan(val) else (alpha*val + (1-alpha)*prev)
        atr[i] = prev
    return pd.Series(atr, index=df.index, dtype="float64")

def supertrend(df: pd.DataFrame, period: int = 7, multiplier: float = 3.5) -> pd.DataFrame:
    h = pd.to_numeric(df["High"], errors="coerce").astype("float64").to_numpy()
    l = pd.to_numeric(df["Low"],  errors="coerce").astype("float64").to_numpy()
    c = pd.to_numeric(df["Close"],errors="coerce").astype("float64").to_numpy()
    atr = _atr_numpy(df, period).to_numpy()
    hl2 = (h+l)/2.0
    upper = hl2 + multiplier*atr
    lower = hl2 - multiplier*atr
    n = len(c)
    f_upper = np.copy(upper); f_lower = np.copy(lower)
    for i in range(1,n):
        f_upper[i] = min(upper[i], f_upper[i-1]) if c[i-1] <= f_upper[i-1] else upper[i]
        f_lower[i] = max(lower[i], f_lower[i-1]) if c[i-1] >= f_lower[i-1] else lower[i]
    st = np.zeros(n, dtype=np.float64); dir_ = np.zeros(n, dtype=np.int64)
    st[0] = f_lower[0]; dir_[0] = 1
    for i in range(1,n):
        if c[i] > f_upper[i]:
            dir_[i] = 1; st[i] = f_lower[i]
        elif c[i] < f_lower[i]:
            dir_[i] = -1; st[i] = f_upper[i]
        else:
            dir_[i] = dir_[i-1]; st[i] = st[i-1]
    return pd.DataFrame({"ST": st, "ST_DIR": dir_}, index=df.index)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"] = ema(out["Close"], EMA_PERIOD).astype("float64")
    st = supertrend(out, period=ST_ATR_PERIOD, multiplier=ST_MULTIPLIER)
    out["ST"] = pd.to_numeric(st["ST"], errors="coerce").astype("float64")
    out["ST_DIR"] = pd.to_numeric(st["ST_DIR"], errors="coerce").astype("int64")
    return out

# ---------------- Signal logic ----------------
def latest_hourly_close_row_index(df: pd.DataFrame) -> int | None:
    idx_last = len(df)-1
    ts_last: datetime = df.index[idx_last]
    now_ist = datetime.now(IST)
    if ts_last.date() != now_ist.date():
        log(f"Last bar {ts_last} not from today; skipping."); return None
    if not (10 <= int(ts_last.minute) <= 20):
        log(f"Last bar minute {ts_last.minute} not in 10..20; skipping (ts={ts_last})."); return None
    if (now_ist - ts_last) > timedelta(hours=6):
        log(f"Last bar {ts_last} stale (>6h); skipping."); return None
    return idx_last

def _get_scalar(x) -> float:
    try:
        v = float(x)
        if math.isnan(v): raise ValueError("NaN")
        return v
    except Exception:
        raise ValueError("not scalar")

def evaluate_signal(df: pd.DataFrame) -> dict:
    idx_last = latest_hourly_close_row_index(df)
    if idx_last is None:
        return {"status": "NO_ACTION"}
    try:
        c  = _get_scalar(df["Close"].iat[idx_last])
        st = _get_scalar(df["ST"].iat[idx_last])
        e  = _get_scalar(df["EMA20"].iat[idx_last])
    except ValueError:
        return {"status": "NO_ACTION"}

    pc=pst=pe=np.nan
    if idx_last>=1:
        try:
            pc  = _get_scalar(df["Close"].iat[idx_last-1])
            pst = _get_scalar(df["ST"].iat[idx_last-1])
            pe  = _get_scalar(df["EMA20"].iat[idx_last-1])
        except ValueError:
            pc=pst=pe=np.nan

    bull = (c>st) and (c>e)
    bear = (c<st) and (c<e)

    bull_prev = (not np.isnan(pc)) and (not np.isnan(pst)) and (not np.isnan(pe)) and (pc>pst) and (pc>pe)
    bear_prev = (not np.isnan(pc)) and (not np.isnan(pst)) and (not np.isnan(pe)) and (pc<pst) and (pc<pe)

    crossed_to_bull = bull and (not bull_prev)
    crossed_to_bear = bear and (not bear_prev)

    ts_last: datetime = df.index[idx_last]
    now_ist = datetime.now(IST)

    if crossed_to_bull:
        legs = suggested_option_legs("BULL", c, now_ist)
        return {"status":"SIGNAL","side":"BULLISH","timestamp":ts_last.isoformat(),"close":c,"ema20":e,"supertrend":st,"option_legs":legs}
    if crossed_to_bear:
        legs = suggested_option_legs("BEAR", c, now_ist)
        return {"status":"SIGNAL","side":"BEARISH","timestamp":ts_last.isoformat(),"close":c,"ema20":e,"supertrend":st,"option_legs":legs}

    return {"status":"NO_SIGNAL","timestamp":ts_last.isoformat(),"close":c,"ema20":e,"supertrend":st}

# ---------------- Telegram ----------------
def send_telegram(text: str) -> None:
    if not SEND_TELEGRAM:
        log("Telegram disabled (missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID)."); return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text})
    if r.status_code != 200:
        log(f"Telegram error: {r.status_code} {r.text}")
    else:
        log("Telegram sent.")

def fmt_signal_text(sig: dict) -> str:
    if sig.get("status")!="SIGNAL": return ""
    ts_ist = datetime.fromisoformat(sig["timestamp"]).astimezone(IST).strftime("%Y-%m-%d %H:%M IST")
    legs = sig["option_legs"]
    return (
        f"📈 NIFTY Hourly Signal ({ts_ist})\n"
        f"Side: {sig['side']}\n"
        f"Close: {sig['close']:.2f} | ST: {sig['supertrend']:.2f} | EMA20: {sig['ema20']:.2f}\n"
        f"Suggested ({legs['expiry_type']}):\n"
        f"Expiry: {legs['expiry']}\n"
        f" • SELL {legs['sell']}\n"
        f" • BUY  {legs['buy']}\n"
        f"Exit if Supertrend flips or by Tue midday."
    )

# ---------------- CLI / Main ----------------
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ping", action="store_true", help="Send a Telegram test message and exit")
    args = parser.parse_args(argv)

    if args.ping:
        now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        send_telegram(f"✅ Telegram test ping from NIFTY scanner at {now}")
        print(json.dumps({"status":"PING_SENT","time":now}, indent=2))
        return

    if HEARTBEAT:
        now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        send_telegram(f"🟢 Heartbeat: NIFTY scanner alive at {now}")
        print(json.dumps({"status":"HEARTBEAT_SENT","time":now}, indent=2))
        return

    try:
        log("Fetching data...")
        df = fetch_hourly()
        log(f"Fetched: shape={df.shape}, cols={list(df.columns)}")
        log(f"Last bar: {df.index[-1]}")
        df = compute_indicators(df)
        df = _force_float_cols(df, ["EMA20","ST"])

        sig = evaluate_signal(df)
        print(json.dumps(sig, indent=2))

        if sig.get("status")=="SIGNAL":
            msg = fmt_signal_text(sig)
            log(msg); send_telegram(msg)
        else:
            log(f"No new signal. Status: {sig['status']}")
    except Exception as e:
        log("ERROR: "+repr(e)); traceback.print_exc(); raise

if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
