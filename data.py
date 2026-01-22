import pandas as pd
import yfinance as yf


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    return df


def _load_from_yahoo(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ticker_map = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
    }
    tkr = ticker_map.get(symbol)
    if not tkr:
        raise ValueError("Only NIFTY and BANKNIFTY supported")

    df = yf.download(
        tkr,
        start=start.date(),
        end=(end + pd.Timedelta(days=1)).date(),
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = _flatten_cols(df)

    # Standardize column names
    rename = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ["open", "high", "low", "close", "volume"]:
            rename[c] = lc.capitalize()

    df = df.rename(columns=rename)

    # Ensure required cols exist
    required = ["Open", "High", "Low", "Close"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    if "Volume" not in df.columns:
        df["Volume"] = 0

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
    return df


def load_daily_data(symbol: str, start: pd.Timestamp, end: pd.Timestamp, source_choice: str = "Yahoo Finance") -> pd.DataFrame:
    # Hosted V1: Yahoo only
    return _load_from_yahoo(symbol, start, end)
