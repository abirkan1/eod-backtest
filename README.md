# EOD Backtesting Workbench (NIFTY & BANKNIFTY)

A simple, practical EOD backtesting UI:
- Universe: NIFTY 50, BANKNIFTY
- Signals on close (t), execution at next open (t+1)
- Long-only V1
- Fixed capital per trade
- Rule Builder (no coding)

## Setup (Windows/macOS/Linux)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Data sources
### Auto (default)
- Tries Zerodha Kite **if** configured, else uses Yahoo Finance.

### Zerodha Kite (recommended for indices)
Set environment variables before running:
- `KITE_API_KEY`
- `KITE_ACCESS_TOKEN`

Example (Windows PowerShell):
```powershell
setx KITE_API_KEY "your_key"
setx KITE_ACCESS_TOKEN "your_access_token"
```

Then restart terminal and run:
```bash
streamlit run app.py
```

> Note: Access token expires; refresh it when needed.

## Notes
- This is a V1. Next upgrades we can add quickly:
  - Shorting (long/short)
  - Portfolio-level equity with reinvestment / compounding
  - Better fill model (gap stops, intraday stop simulation using next-day O/H/L)
  - More rule templates (VWAP regimes, Donchian, ATR filters, etc.)
  - PDF report export
