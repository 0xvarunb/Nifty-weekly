#!/usr/bin/env python3
"""
NIFTY hourly scanner for Supertrend + EMA(20)

Signal rules (evaluated on the latest completed hourly candle, IST):
- Bullish  = Close > Supertrend AND Close > EMA20
- Bearish  = Close < Supertrend AND Close < EMA20
Alert triggers only on a NEW flip into bull/bear.

Outputs a JSON summary and (optionally) sends a Telegram alert with:
- Side (BULLISH/BEARISH)
- Close / Supertrend / EMA20
- Suggested weekly option legs (ATM sell + OTM hedge)
- Next weekly expiry date (Thu, IST)
"""
from __future__ import annotations

import os, sys, json, math
from datetime import datetime, timedelta, timezone, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import requests
import yfinance as yf

from indicators import ema, supertrend

# ---------- Config ----------
TICKER            = os.getenv("NIFTY_TICKER", "^NSEI")   # Yahoo NIFTY50 index
INTERVAL          = "60m"
LOOKBACK_DAYS     = int(os.getenv("LOOKBACK_DAYS", "60"))

EMA_PERIOD        = int(os.getenv("EMA_PERIOD", "20"))
ST_ATR_PERIOD     = int(os.getenv("ST_ATR_PERIOD", "7"))
ST_MULTIPLIER     = float(os.getenv("ST_MULTIPLIER", "3.5"))

OTM_GAP_POINTS    = int(os.getenv("OTM_GAP_POINTS", "100"))   # hedge gap suggestion
ATM_STEP_POINTS   = int(os.getenv("ATM_STEP_POINTS", "50"))    # NIFTY=50, BANKNIFTY=100

LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO")

# Telegram (optional)
TG_TOKEN          = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID        = os.getenv("TELEGRAM_CHAT_ID", "")
SEND_TELEGRAM     = bool(TG_TOKEN and TG_CHAT_ID)

IST = ZoneInfo("Asia/Kolkata")

def log(msg: str):
    prefix = f"[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}] " if LOG_LEVEL.upper()=="DEBUG" else ""
    print(prefix + str(msg), flush=True)

# ---------- Helpers ----------
def round_to_step(x: float, step: int) -> int:
    # round to nearest step (e.g., 50 for NIFTY)
    return int(round(float(x) / float(step)) * step)

def next_weekly_expiry_ist(now_ist: datetime) -> str:
    """ Weekly index options typically expire on Thursday (IST). """
    dow = now_ist.weekday()  # Mon=0..Sun=6
    days_ahead = (3 - dow) % 7  # 3 == Thu
    expiry = (now_ist + timedelta(days=days_ahead)).date()
    if dow == 3 and now_ist.time() > dtime(15, 30):
        expiry = (now_ist + timedelta(days=7)).date()
    return str(expiry)

def suggested_option_legs(side: str, spot: float, now_ist: datetime) -> dict:
    """
    Bullish -> Sell ATM PE, Buy OTM PE
    Bearish -> Sell ATM CE, Buy OTM CE
    """
    atm = round_to_step(spot, ATM_STEP_POINTS)
    if side == "BULL":
        legs = {
            "sell": f"{atm} PE (ATM)",
            "buy":  f"{max(ATM_STEP_POINTS, atm - OTM_GAP_POINTS)} PE (hedge ~{OTM_GAP_POINTS})"
        }
    else:
        legs = {
            "sell": f"{atm} CE (ATM)",
            "buy":  f"{atm + OTM_GAP_POINTS} CE (hedge ~{OTM_GAP_POINTS})"
        }
    legs["expiry"] = next_weekly_expiry_ist(now_ist)
    return legs

# ---------- Data & Indicators ----------
def _force_float_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df

def fetch_hourly() -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    df = yf.download(TICKER, start=start, end=end, interval=INTERVAL, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("No data returned from yfinance. Try later or change ticker.")
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    df = df.tz_convert(IST)
    # ensure numeric types
    df = _force_float_cols(df, ["Open", "High", "Low", "Close", "Adj Close", "Volume"])
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"] = ema(out["Close"], EMA_PERIOD).astype("float64")
    st = supertrend(out, period=ST_ATR_PERIOD, multiplier=ST_MULTIPLIER)
    out["ST"] = pd.to_numeric(st["ST"], errors="coerce").astype("float64")
    out["ST_DIR"] = pd.to_numeric(st["ST_DIR"], errors="coerce").astype("int64")
    return out

def latest_hourly_close_row_index(df: pd.DataFrame) -> int | None:
    """
    Accept a bar that looks like today's hourly close (~:15 past the hour).
    """
    idx_last = len(df) - 1
    ts_last: datetime = df.index[idx_last]
    now_ist = datetime.now(IST)

    if ts_last.date() != now_ist.date():
        log(f"Last bar {ts_last} not from today; skipping.")
        return None
    # hourly close window tolerance
    if not (10 <= int(ts_last.minute) <= 20):
        log(f"Last bar minute {ts_last.minute} not in 10..20; skipping (ts={ts_last}).")
        return None
    if (now_ist - ts_last) > timedelta(hours=6):
        log(f"Last bar {ts_last} stale (>6h); skipping.")
        return None
    return idx_last

# ---------- Safe scalar extraction ----------
def _get_scalar(series_like) -> float:
    """
    Convert any pandas/NumPy scalar-ish object to a plain Python float.
    If it's NaN or cannot be converted, raise ValueError.
    """
    try:
        # handle numpy scalar, pandas scalar, or regular float/int
        val = float(series_like)
        if math.isnan(val):
            raise ValueError("NaN")
        return val
    except Exception as _:
        raise ValueError("not a scalar")

# ---------- Signal ----------
def evaluate_signal(df: pd.DataFrame) -> dict:
    idx_last = latest_hourly_close_row_index(df)
    if idx_last is None:
        return {"status": "NO_ACTION"}

    # Scalars via .iat and safe cast
    try:
        c  = _get_scalar(df["Close"].iat[idx_last])
        st = _get_scalar(df["ST"].iat[idx_last])
        e  = _get_scalar(df["EMA20"].iat[idx_last])
    except ValueError:
        return {"status": "NO_ACTION"}

    pc = pst = pe = np.nan
    if idx_last >= 1:
        try:
            pc  = _get_scalar(df["Close"].iat[idx_last-1])
            pst = _get_scalar(df["ST"].iat[idx_last-1])
            pe  = _get_scalar(df["EMA20"].iat[idx_last-1])
        except ValueError:
            pc = pst = pe = np.nan

    # Current regime (pure Python bools)
    bull = (c > st) and (c > e)
    bear = (c < st) and (c < e)

    # Prior regime
    bull_prev = (not np.isnan(pc)) and (not np.isnan(pst)) and (not np.isnan(pe)) and (pc > pst) and (pc > pe)
    bear_prev = (not np.isnan(pc)) and (not np.isnan(pst)) and (not np.isnan(pe)) and (pc < pst) and (pc < pe)

    crossed_to_bull = bull and (not bull_prev)
    crossed_to_bear = bear and (not bear_prev)

    ts_last: datetime = df.index[idx_last]
    now_ist = datetime.now(IST)

    if crossed_to_bull:
        legs = suggested_option_legs("BULL", c, now_ist)
        return {
            "status": "SIGNAL",
            "side": "BULLISH",
            "timestamp": ts_last.isoformat(),
            "close": c, "ema20": e, "supertrend": st,
            "option_legs": legs
        }
    if crossed_to_bear:
        legs = suggested_option_legs("BEAR", c, now_ist)
        return {
            "status": "SIGNAL",
            "side": "BEARISH",
            "timestamp": ts_last.isoformat(),
            "close": c, "ema20": e, "supertrend": st,
            "option_legs": legs
        }

    return {
        "status": "NO_SIGNAL",
        "timestamp": ts_last.isoformat(),
        "close": c, "ema20": e, "supertrend": st
    }

# ---------- Alerts ----------
def send_telegram(msg: str) -> None:
    if not SEND_TELEGRAM:
        log("Telegram disabled (missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID).")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": msg})
        if r.status_code != 200:
            log(f"Telegram error: {r.status_code} {r.text}")
        else:
            log("Telegram sent.")
    except Exception as e:
        log(f"Telegram send failed: {e}")

def format_signal_text(sig: dict) -> str:
    if sig.get("status") != "SIGNAL":
        return ""
    side = sig["side"]
    ts_ist = datetime.fromisoformat(sig["timestamp"]).astimezone(IST).strftime("%Y-%m-%d %H:%M IST")
    spot = sig["close"]; st = sig["supertrend"]; ema20 = sig["ema20"]; legs = sig["option_legs"]
    return (
        f"📈 NIFTY Hourly Signal ({ts_ist})\n"
        f"Side: {side}\n"
        f"Close: {spot:.2f} | ST: {st:.2f} | EMA20: {ema20:.2f}\n"
        f"Suggested (weekly):\n"
        f"Expiry: {legs['expiry']}\n"
        f" • SELL {legs['sell']}\n"
        f" • BUY  {legs['buy']}\n"
        f"Exit if Supertrend flips or by Wed EOD/Thu AM."
    )

# ---------- Main ----------
def main():
    log("Fetching data...")
    df = fetch_hourly()
    log(f"Rows: {len(df)} | Last bar: {df.index[-1]}")

    df = compute_indicators(df)
    # force numeric again just in case
    df = _force_float_cols(df, ["EMA20", "ST"])

    sig = evaluate_signal(df)

    # Always print JSON (for logs/consumers)
    print(json.dumps(sig, indent=2))

    if sig.get("status") == "SIGNAL":
        msg = format_signal_text(sig)
        log(msg)
        send_telegram(msg)
    else:
        log(f"No new signal. Status: {sig['status']}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)
