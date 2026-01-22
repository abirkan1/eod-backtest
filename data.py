
import os
import pandas as pd

def _load_from_kite(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Requires:
      - pip install kiteconnect
      - env vars: KITE_API_KEY, KITE_ACCESS_TOKEN
    """
    try:
        from kiteconnect import KiteConnect
    except Exception as e:
        raise RuntimeError("kiteconnect not installed") from e

    api_key = os.getenv("KITE_API_KEY", "")
    access_token = os.getenv("KITE_ACCESS_TOKEN", "")
    if not api_key or not access_token:
        raise RuntimeError("KITE_API_KEY / KITE_ACCESS_TOKEN not set")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    # Map to Zerodha index symbols (as instrument tokens need lookup).
    # We'll search instruments for NSE indices; this is a bit slow first time.
    instruments = kite.instruments("NSE")
    inst_df = pd.DataFrame(instruments)

    if symbol == "NIFTY":
        tradingsymbol = "NIFTY 50"
    elif symbol == "BANKNIFTY":
        tradingsymbol = "NIFTY BANK"
    else:
        raise ValueError("Only NIFTY and BANKNIFTY supported")

    row = inst_df[(inst_df["name"] == tradingsymbol) & (inst_df["segment"] == "NSE")].head(1)
    if row.empty:
        # Some accounts return indices under different exchange; try NFO-INDICES instruments list
        try:
            instruments2 = kite.instruments("NFO")
            inst2 = pd.DataFrame(instruments2)
            row = inst2[(inst2["name"] == tradingsymbol) | (inst2["tradingsymbol"] == tradingsymbol)].head(1)
        except Exception:
            pass
    if row.empty:
        raise RuntimeError(f"Could not find instrument for {tradingsymbol}")

    instrument_token = int(row.iloc[0]["instrument_token"])

    # Kite expects python datetime
    from_d = start.to_pydatetime()
    to_d = end.to_pydatetime()

    candles = kite.historical_data(instrument_token, from_d, to_d, interval="day", continuous=False, oi=False)
    df = pd.DataFrame(candles)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"date": "Date", "open":"Open", "high":"High", "low":"Low", "close":"Close", "volume":"Volume"})
    df = df.set_index("Date").sort_index()
    return df[["Open","High","Low","Close","Volume"]]

def _load_from_yahoo(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance not installed") from e

    ticker_map = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
    }
    tkr = ticker_map.get(symbol)
    if not tkr:
        raise ValueError("Only NIFTY and BANKNIFTY supported")

    df = yf.download(tkr, start=start.date(), end=(end + pd.Timedelta(days=1)).date(), auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize columns
    df = df.rename(columns={
        "Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"
    })
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    for c in ["Open","High","Low","Close","Volume"]:
        if c not in df.columns:
            df[c] = pd.NA
    return df[["Open","High","Low","Close","Volume"]].dropna(subset=["Open","High","Low","Close"])

def load_daily_data(symbol: str, start: pd.Timestamp, end: pd.Timestamp, source_choice: str = "Auto (Kite if configured, else Yahoo)") -> pd.DataFrame:
    """
    Local cache per symbol + date range + source.
    """
    cache_dir = os.path.join(os.path.dirname(__file__), "..", ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    key = f"{symbol}_{start.date()}_{end.date()}_{source_choice.replace(' ','_').replace('/','_')}"
    path = os.path.join(cache_dir, key + ".parquet")

    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    df = pd.DataFrame()
    if source_choice.startswith("Auto"):
        # try kite first
        try:
            df = _load_from_kite(symbol, start, end)
        except Exception:
            df = _load_from_yahoo(symbol, start, end)
    elif source_choice.startswith("Zerodha"):
        df = _load_from_kite(symbol, start, end)
    else:
        df = _load_from_yahoo(symbol, start, end)

    if df is None:
        df = pd.DataFrame()
    if not df.empty:
        df.to_parquet(path, index=True)
    return df
