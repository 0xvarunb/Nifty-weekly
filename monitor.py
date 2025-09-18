#!/usr/bin/env python3
"""
NIFTY hourly scanner for Supertrend + EMA(20) regime
- Bullish signal  = Close > Supertrend AND Close > EMA20
- Bearish signal  = Close < Supertrend AND Close < EMA20

Outputs a JSON summary and (optionally) sends a Telegram alert with:
- Side (BULLISH/BEARISH)
- Close / Supertrend / EMA20
- Suggested weekly option legs (ATM sell + OTM hedge)
- Next weekly expiry date (IST)

Author: you + ChatGPT
"""
from __future__ import annotations

import os, sys, json
from datetime import datetime, timedelta, timezone, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import requests
import yfinance as yf

from indicators import ema, supertrend

# ---------- Config (override via ENV in GitHub Secrets if you want) ----------
TICKER            = os.getenv("NIFTY_TICKER", "^NSEI")   # Yahoo: NIFTY 50 index
INTERVAL          = "60m"                                # hourly
LOOKBACK_DAYS     = int(os.getenv("LOOKBACK_DAYS", "60"))
EMA_PERIOD        = int(os.getenv("EMA_PERIOD", "20"))
ST_ATR_PERIOD     = int(os.getenv("ST_ATR_PERIOD", "7"))
ST_MULTIPLIER     = float(os.getenv("ST_MULTIPLIER", "3.5"))
OTM_GAP_POINTS    = int(os.getenv("OTM_GAP_POINTS", "100"))  # hedge distance suggestion
ATM_STEP_POINTS   = int(os.getenv("ATM_STEP_POINTS", "50"))   # NIFTY=50, BANKNIFTY=100
LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO")

# Telegram (optional). Put these in GitHub repo -> Settings -> Secrets -> Actions
TG_TOKEN          = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID        = os.getenv("TELEGRAM_CHAT_ID", "")
SEND_TELEGRAM     = bool(TG_TOKEN and TG_CHAT_ID)

IST = ZoneInfo("Asia/Kolkata")

def log(msg: str):
    prefix = f"[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}] " if LOG_LEVEL.upper() == "DEBUG" else ""
    print(prefix + str(msg), flush=True)

# ---------- Helpers ----------
def round_to_step(x: float, step: int) -> int:
    return int(round(x / step) * step)

def next_weekly_expiry_ist(now_ist: datetime) -> str:
    """
    Weekly index options in India typically expire on Thursday.
    If it's already Thu after market close (15:30), roll to next Thu.
    """
    dow = now_ist.weekday()  # Mon=0 ... Sun=6
    days_ahead = (3 - dow) % 7  # 3 = Thu
    expiry_date = (now_ist + timedelta(days=days_ahead)).date()
    # If today is Thursday and after market close, push to next week
    if dow == 3 and now_ist.time() > dtime(15, 30):
        expiry_date = (now_ist + timedelta(days=7)).date()
    return str(expiry_date)

def suggested_option_legs(side: str, spot: float, now_ist: datetime) -> dict:
    """
    For a bullish signal, suggest: SELL ATM PUT, BUY OTM PUT
    For a bearish signal, suggest: SELL ATM CALL, BUY OTM CALL
    Includes next weekly expiry (IST).
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

# ---------- Data / Indicators ----------
def fetch_hourly() -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    df = yf.download(
        TICKER, start=start, end=end, interval=INTERVAL, auto_adjust=False, progress=False
    )
    if df.empty:
        raise RuntimeError("No data returned from yfinance. Try again or change ticker.")
    # Ensure tz-aware; convert to IST for convenience.
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    df = df.tz_convert(IST)
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"] = ema(out["Close"], EMA_PERIOD)
    st = supertrend(out, period=ST_ATR_PERIOD, multiplier=ST_MULTIPLIER)
    out["ST"] = st["ST"]
    out["ST_DIR"] = st["ST_DIR"]  # +1 uptrend, -1 downtrend
    return out

def latest_hourly_close_row(df: pd.DataFrame) -> pd.Series | None:
    """
    NSE hourly bars typically close at ~:15 (10:15, 11:15, ... 15:15 IST).
    We'll use the last row, but only act if it's from today and looks like a close bar.
    """
    last = df.iloc[-1]
    ts: datetime = df.index[-1]
    now_ist = datetime.now(IST)

    if ts.date() != now_ist.date():
        log(f"Last bar {ts} not from today; skipping.")
        return None

    # Accept bars that look like closes (10..20 minutes past the hour)
    if not (10 <= ts.minute <= 20):
        log(f"Last bar minute {ts.minute} not in 10..20 (ts={ts}); likely mid-bar; skipping.")
        return None

    # Basic staleness guard
    if now_ist - ts > timedelta(hours=6):
        log(f"Last bar {ts} looks stale (>6h); skipping.")
        return None

    return last

# ---------- Signal ----------
def evaluate_signal(df: pd.DataFrame) -> dict:
    last = latest_hourly_close_row(df)
    if last is None:
        return {"status": "NO_ACTION"}

    # Cast to plain floats to avoid ambiguous truth errors
    try:
        c  = float(last["Close"])
        st = float(last["ST"])
        e  = float(last["EMA20"])
    except Exception:
        return {"status": "NO_ACTION"}

    prev = df.iloc[-2] if len(df) >= 2 else None
    if prev is not None:
        try:
            pc  = float(prev["Close"])
            pst = float(prev["ST"])
            pe  = float(prev["EMA20"])
        except Exception:
            pc = pst = pe = np.nan
    else:
        pc = pst = pe = np.nan

    # Current regime (scalars only)
    bull = (c > st) and (c > e)
    bear = (c < st) and (c < e)

    # Prior regime (handle NaNs safely)
    bull_prev = (not np.isnan(pc)) and (not np.isnan(pst)) and (not np.isnan(pe)) and (pc > pst) and (pc > pe)
    bear_prev = (not np.isnan(pc)) and (not np.isnan(pst)) and (not np.isnan(pe)) and (pc < pst) and (pc < pe)

    crossed_to_bull = bull and (not bull_prev)
    crossed_to_bear = bear and (not bear_prev)

    spot = c
    ts = df.index[-1]
    now_ist = datetime.now(IST)

    if crossed_to_bull:
        legs = suggested_option_legs("BULL", spot, now_ist)
        return {
            "status": "SIGNAL",
            "side": "BULLISH",
            "timestamp": ts.isoformat(),
            "close": spot,
            "ema20": e,
            "supertrend": st,
            "option_legs": legs
        }

    if crossed_to_bear:
        legs = suggested_option_legs("BEAR", spot, now_ist)
        return {
            "status": "SIGNAL",
            "side": "BEARISH",
            "timestamp": ts.isoformat(),
            "close": spot,
            "ema20": e,
            "supertrend": st,
            "option_legs": legs
        }

    return {
        "status": "NO_SIGNAL",
        "timestamp": ts.isoformat(),
        "close": spot,
        "ema20": e,
        "supertrend": st
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
    spot = sig["close"]
    st = sig["supertrend"]
    ema20 = sig["ema20"]
    legs = sig["option_legs"]
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
    sig = evaluate_signal(df)

    # Always print a JSON summary
    print(json.dumps(sig, indent=2))

    # Optional Telegram alert only when a new signal appears
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
