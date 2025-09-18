# NIFTY Hourly Trend Scanner (Supertrend + EMA20)

**What it does**  
On every hourly close (IST) during trading days, this repo checks NIFTY’s trend:
- **Bullish** if `Close > Supertrend` **and** `Close > EMA20`
- **Bearish** if `Close < Supertrend` **and** `Close < EMA20`
It alerts only on a **fresh flip** into bull/bear.

**Alert includes:**
- Side, Close, ST, EMA20
- Suggested weekly option legs (ATM sell + OTM hedge)
- **Next weekly expiry (Thu, IST)**

> This project does **not** place orders. It scans & alerts.

---

## Quick Start
1. Create a GitHub repo and add these files:  
   `monitor.py`, `indicators.py`, `requirements.txt`, `.github/workflows/scan.yml`
2. (Optional) In **Settings → Secrets and variables → Actions**, add:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
3. Push. The workflow will run around each hourly close (Mon–Fri).
4. View logs in **Actions**. If a new signal fires, you’ll also get a Telegram message.

---

## Env overrides (optional)
- `NIFTY_TICKER` (default `^NSEI`)
- `EMA_PERIOD` (default `20`)
- `ST_ATR_PERIOD` (default `7`)
- `ST_MULTIPLIER` (default `3.5`)
- `OTM_GAP_POINTS` (default `100`)
- `ATM_STEP_POINTS` (default `50`)
- `LOG_LEVEL` (`INFO` or `DEBUG`)

---

## Notes
- Intraday data via Yahoo (`yfinance`); availability can vary.
- Signals are educational, not financial advice.
