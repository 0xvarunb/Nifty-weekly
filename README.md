# NIFTY Hourly Trend Scanner (Supertrend + EMA20)

This repo scans **NIFTY 50** on **hourly closes (IST)** and alerts when a new
trend regime starts, using:

- **Supertrend(ATR=7, Mult=3.5)**  
- **EMA(20)**  
- **Signal:**  
  - **Bullish** when `Close > Supertrend` **and** `Close > EMA20`  
  - **Bearish** when `Close < Supertrend` **and** `Close < EMA20`  
- Alerts fire **only on a fresh flip** into bull/bear.

It **does not place orders**. It sends a Telegram message with suggested weekly
option legs (sell ATM + buy OTM hedge) and the next **Tuesday** expiry
(weekly; monthly = last Tuesday).

---

## What you get

- `monitor.py` – robust scanner (yfinance quirks handled), plus:
  - `--ping` to test Telegram instantly
  - optional **daily heartbeat** message
- `indicators.py` – vectorized, stable Supertrend & EMA
- `.github/workflows/scan.yml` – runs each hourly close (IST) on weekdays
- `.github/workflows/telegram_ping.yml` – manual Telegram test
- `.github/workflows/heartbeat.yml` – optional daily “alive” ping
- `requirements.txt`

---

## Quick Start

1) **Create Telegram bot** (if you haven’t):
   - Talk to **@BotFather** → `/newbot` → get your **BOT TOKEN**.
   - Send a message to your bot once, then get your **CHAT ID** (e.g., via
     `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` after you’ve messaged the bot).

2) **Add GitHub Secrets**:  
   Repo → **Settings → Secrets and variables → Actions** → **New repository secret**
   - `TELEGRAM_BOT_TOKEN` = your bot token
   - `TELEGRAM_CHAT_ID`   = your chat id

3) **Push this repo** (or paste the files into your existing repo).

4) **Run a Telegram test** (no market data needed):
   - GitHub → **Actions → Telegram Ping (Manual)** → **Run workflow**  
   You should receive: `✅ Telegram test ping from NIFTY scanner ...`

5) **Let the hourly scanner run**:
   - The `scan.yml` workflow schedules runs close to each hourly close (IST) on **Mon–Fri**.
   - You can also run it on-demand: **Actions → Nifty Hourly Scanner → Run workflow**.

---

## What the alert looks like

📈 NIFTY Hourly Signal (2025-09-23 11:15 IST)
Side: BULLISH
Close: 20234.50 | ST: 20110.22 | EMA20: 20180.10
Suggested (Weekly):
Expiry: 2025-09-23
• SELL 20250 PE (ATM)
• BUY 20150 PE (hedge ~100)
Exit if Supertrend flips or by Tue midday.


- **ATM** is spot rounded to **50** (NIFTY step).
- **OTM hedge** defaults to **±100** points (configurable).
- **Expiry** honors the NSE change: **weekly = every Tuesday**; **monthly = last Tuesday**.

---

## Files

- `monitor.py` – main scanner & Telegram alerts (supports `--ping` and heartbeat)
- `indicators.py` – EMA + Supertrend (NumPy-robust ATR)
- `.github/workflows/scan.yml` – hourly schedule (Mon–Fri)
- `.github/workflows/telegram_ping.yml` – manual Telegram test
- `.github/workflows/heartbeat.yml` – optional daily “alive” ping
- `requirements.txt`

---

## Configuration

You can override defaults via **Workflow env** or **repo secrets/variables**.

| Env var            | Default | Purpose |
|--------------------|---------|---------|
| `NIFTY_TICKER`     | `^NSEI` | Yahoo finance ticker for NIFTY 50 |
| `EMA_PERIOD`       | `20`    | EMA length |
| `ST_ATR_PERIOD`    | `7`     | Supertrend ATR length |
| `ST_MULTIPLIER`    | `3.5`   | Supertrend multiplier |
| `OTM_GAP_POINTS`   | `100`   | Hedge distance from ATM (± points) |
| `ATM_STEP_POINTS`  | `50`    | Strike step (NIFTY=50, BankNifty=100) |
| `LOOKBACK_DAYS`    | `60`    | History window for indicators |
| `EXPIRY_WEEKDAY`   | `TUE`   | Expiry weekday (Mon=0..Sun=6 or names like `TUE`) |
| `LOG_LEVEL`        | `INFO`  | Set `DEBUG` for verbose logs |
| `HEARTBEAT`        | `0`     | Set to `1` to send daily heartbeat (used in `heartbeat.yml`) |
| `TELEGRAM_BOT_TOKEN` | —     | **Secret**: Telegram bot token |
| `TELEGRAM_CHAT_ID` | —       | **Secret**: Telegram chat id |

> If NSE changes expiries again, just change `EXPIRY_WEEKDAY`.

---

## Schedules

- **Hourly Scanner** (`scan.yml`)  
  Triggers around each hourly close (IST) Mon–Fri. Times are set in UTC via cron;
  the script itself only proceeds if the **last bar looks like a proper hourly close**.

- **Telegram Ping (Manual)** (`telegram_ping.yml`)  
  Run on demand to verify secrets and Telegram delivery:
Actions → Telegram Ping (Manual) → Run workflow


- **Daily Heartbeat** (`heartbeat.yml`, optional)  
Sends a short “alive” ping around **12:00 IST** each day.

---

## How it decides a signal

1. Pulls hourly data from Yahoo (tries `Ticker.history`, falls back to `download`).
2. Computes **EMA(20)** and **Supertrend(7, 3.5)**.
3. On the **latest completed hourly candle** (IST):
 - **Bullish** if `Close > ST` **and** `Close > EMA20`.
 - **Bearish** if `Close < ST` **and** `Close < EMA20`.
4. Fires **only on a fresh flip** (wasn’t bull/bear on previous bar).
5. Suggests **Sell ATM** + **Buy OTM hedge**; adds next **Tuesday** expiry and tags **Weekly/Monthly**.

The code is hardened to avoid pandas/yfinance pitfalls: scalar-only comparisons, MultiIndex
column handling, tz-awareness, numeric coercion, and robust ATR.

---

## Common questions

**Q: How do I know Telegram is set correctly?**  
A: Run **Actions → Telegram Ping (Manual)**. You should get a ✅ test message.
You can also enable the **Daily Heartbeat**.

**Q: Can I change hedge distance or strikes?**  
A: Yes — set `OTM_GAP_POINTS` and `ATM_STEP_POINTS` in the workflow env.

**Q: It says “No new signal. Status: NO_ACTION.”**  
A: That’s normal when (a) it’s not an hourly close window yet, (b) the last bar looks stale,
or (c) there’s no fresh flip. Check the logs for timestamps and status.

**Q: Can this place orders automatically?**  
A: No. This repo scans & alerts. Use your broker/platform to place the suggested option basket.

---

## Local run (optional)

If you want to test locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Telegram ping test (requires env vars set or edit the script with your token/id)
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_CHAT_ID=...
python monitor.py --ping

# Manual scan
python monitor.py

Disclaimer

For educational purposes only. No investment advice. Options are risky; use hedges and
position sizing that fit your risk tolerance.


If you want me to embed broker-specific symbol formatting (e.g., Zerodha/Sensibull/Angel), say which broker and I’ll add a small formatter section to the README + code.
::contentReference[oaicite:0]{index=0}
