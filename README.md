# NIFTY Hourly Trend Scanner (Supertrend + EMA20)

**What it does:**  
Every hourly close (IST) on trading days, this repo checks NIFTY’s trend using **Supertrend(7, 3.5)** + **EMA(20)**.  
- If **bullish** (Close > ST and > EMA20) with a fresh flip → it prints a signal and (optionally) sends a **Telegram** alert with suggested weekly option legs:
  - **Sell ATM Put, Buy OTM Put (~100 pts lower)**  
- If **bearish** with a fresh flip → suggested:
  - **Sell ATM Call, Buy OTM Call (~100 pts higher)**

> This repo **does not place orders**. It only scans and alerts.

---

## Quick start

1. **Create a new GitHub repo** and add these four files:
   - `monitor.py`  
   - `indicators.py`  
   - `requirements.txt`  
   - `.github/workflows/scan.yml`

2. (Optional) **Enable Telegram alerts**  
   - In your repo: **Settings → Secrets and variables → Actions**  
   - Add:  
     - `TELEGRAM_BOT_TOKEN` = your bot token  
     - `TELEGRAM_CHAT_ID`   = your chat id  

3. **Run it**  
   - It will trigger automatically at **10:20, 11:20, 12:20, 13:20, 14:20, 15:20 IST** (approx) Mon–Fri.  
   - Or run manually: **Actions → Nifty Hourly Scanner → Run workflow**.

4. **See results**  
   - Open the workflow run logs. You’ll see a JSON summary.  
   - If a new signal occurs, you’ll also get a Telegram message (if configured).

---

## Parameters (optional ENV)

- `EMA_PERIOD` (default **20**)  
- `ST_ATR_PERIOD` (default **7**)  
- `ST_MULTIPLIER` (default **3.5**)  
- `OTM_GAP_POINTS` (default **100**) – hedge leg distance suggestion  
- `NIFTY_TICKER` (default `^NSEI`)

---

## How the signal is decided

- **Bullish:** `Close > Supertrend` **and** `Close > EMA20`  
- **Bearish:** `Close < Supertrend` **and** `Close < EMA20`  
- Alert fires **only on a fresh flip** (e.g., was not bullish on the previous bar).

**Exit idea (for your trading plan):**  
- Exit when Supertrend flips against you or by Wed EOD / Thu AM.

---

## Notes
- Intraday data comes from Yahoo Finance via `yfinance`. Availability can vary.
- This is **not** investment advice. For education only.
