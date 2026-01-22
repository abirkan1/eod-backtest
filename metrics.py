
import pandas as pd
import numpy as np

def compute_metrics(equity: pd.DataFrame, trades: pd.DataFrame) -> dict:
    if equity is None or equity.empty:
        return {}

    eq = equity["Equity"].copy().dropna()
    # convert to returns (daily)
    rets = eq.pct_change().dropna()

    # CAGR approximation
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25 if days > 0 else 0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1 if years > 0 and eq.iloc[0] > 0 else np.nan

    dd = eq / eq.cummax() - 1.0
    max_dd = dd.min()

    win_rate = float((trades["NetPnL"] > 0).mean()) if trades is not None and not trades.empty else np.nan
    profit_factor = float(trades.loc[trades["NetPnL"] > 0, "NetPnL"].sum() / abs(trades.loc[trades["NetPnL"] < 0, "NetPnL"].sum())) if trades is not None and not trades.empty and (trades["NetPnL"] < 0).any() else np.nan
    avg_win = float(trades.loc[trades["NetPnL"] > 0, "NetPnL"].mean()) if trades is not None and not trades.empty and (trades["NetPnL"] > 0).any() else np.nan
    avg_loss = float(trades.loc[trades["NetPnL"] < 0, "NetPnL"].mean()) if trades is not None and not trades.empty and (trades["NetPnL"] < 0).any() else np.nan
    expectancy = float(trades["NetPnL"].mean()) if trades is not None and not trades.empty else np.nan

    sharpe = (rets.mean() / (rets.std() + 1e-12)) * np.sqrt(252) if len(rets) > 2 else np.nan

    out = {
        "CAGR": None if pd.isna(cagr) else round(float(cagr)*100, 2),
        "MaxDrawdown": round(float(max_dd)*100, 2) if pd.notna(max_dd) else None,
        "Trades": int(len(trades)) if trades is not None else 0,
        "WinRate": None if pd.isna(win_rate) else round(win_rate*100, 2),
        "ProfitFactor": None if pd.isna(profit_factor) else round(profit_factor, 3),
        "AvgWin": None if pd.isna(avg_win) else round(avg_win, 2),
        "AvgLoss": None if pd.isna(avg_loss) else round(avg_loss, 2),
        "Expectancy": None if pd.isna(expectancy) else round(expectancy, 2),
        "Sharpe": None if pd.isna(sharpe) else round(float(sharpe), 3),
    }
    return out

def monthly_returns_table(equity: pd.DataFrame) -> pd.DataFrame:
    eq = equity["Equity"].dropna().copy()
    # convert equity to monthly returns
    m = eq.resample("M").last().pct_change().dropna()
    if m.empty:
        return pd.DataFrame()
    m.index = pd.to_datetime(m.index)
    df = m.to_frame("Return")
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    pivot = df.pivot_table(index="Year", columns="Month", values="Return", aggfunc="sum").sort_index()
    pivot = (pivot * 100).round(2)
    # nice month names
    month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    pivot = pivot.rename(columns=month_map)
    return pivot
