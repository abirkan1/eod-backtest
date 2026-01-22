import numpy as np
import pandas as pd


def _safe_div(a, b):
    if b == 0:
        return np.nan
    return a / b


def compute_equity_curve(trades: pd.DataFrame, initial_capital: float = 1_000_000.0) -> pd.Series:
    """
    Builds an equity curve from trade-level PnL.
    Assumes trades contain:
      - ExitDate
      - NetPnL (preferred) or GrossPnL
    """
    if trades is None or trades.empty:
        return pd.Series([initial_capital], index=[pd.Timestamp("2000-01-01")])

    df = trades.copy()

    if "ExitDate" not in df.columns:
        # fallback if ExitDate missing
        df["ExitDate"] = df.get("EntryDate", pd.Timestamp.today())

    df["ExitDate"] = pd.to_datetime(df["ExitDate"])

    if "NetPnL" in df.columns:
        pnl_col = "NetPnL"
    elif "GrossPnL" in df.columns:
        pnl_col = "GrossPnL"
    else:
        # nothing to compute
        return pd.Series([initial_capital], index=[pd.Timestamp("2000-01-01")])

    df[pnl_col] = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)

    # sort by exit time
    df = df.sort_values("ExitDate")

    # cumulative equity
    eq = initial_capital + df[pnl_col].cumsum()
    eq.index = df["ExitDate"]

    # ensure starts at initial
    if len(eq) == 0:
        return pd.Series([initial_capital], index=[pd.Timestamp("2000-01-01")])

    return eq


def _max_drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min() * 100.0)


def _annualized_sharpe(daily_returns: pd.Series) -> float:
    """
    Daily returns assumed.
    """
    if daily_returns is None or len(daily_returns) < 10:
        return 0.0
    mu = daily_returns.mean()
    sig = daily_returns.std(ddof=0)
    if sig == 0:
        return 0.0
    return float((mu / sig) * np.sqrt(252))


def compute_metrics(trades: pd.DataFrame, initial_capital: float = 1_000_000.0) -> dict:
    """
    Returns a metrics dict:
    - CAGR (%)
    - MaxDrawdown (%)
    - Trades
    - WinRate (%)
    - ProfitFactor
    - AvgWin / AvgLoss
    - Expectancy (₹)
    - Sharpe
    """
    if trades is None or trades.empty:
        return {
            "CAGR": 0.0,
            "MaxDrawdown": 0.0,
            "Trades": 0,
            "WinRate": 0.0,
            "ProfitFactor": 0.0,
            "AvgWin": 0.0,
            "AvgLoss": 0.0,
            "Expectancy": 0.0,
            "Sharpe": 0.0,
        }

    df = trades.copy()

    # PnL column
    if "NetPnL" in df.columns:
        pnl_col = "NetPnL"
    elif "GrossPnL" in df.columns:
        pnl_col = "GrossPnL"
    else:
        # fallback
        pnl_col = None

    if pnl_col is None:
        return {
            "CAGR": 0.0,
            "MaxDrawdown": 0.0,
            "Trades": int(len(df)),
            "WinRate": 0.0,
            "ProfitFactor": 0.0,
            "AvgWin": 0.0,
            "AvgLoss": 0.0,
            "Expectancy": 0.0,
            "Sharpe": 0.0,
        }

    df[pnl_col] = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)

    # equity curve (trade-based)
    equity = compute_equity_curve(df, initial_capital=initial_capital)

    # CAGR
    if len(equity) < 2:
        cagr = 0.0
    else:
        start = equity.index.min()
        end = equity.index.max()
        years = max((end - start).days / 365.25, 1e-9)
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
        cagr = float(cagr * 100.0)

    # Drawdown
    max_dd = _max_drawdown(equity)

    # Win/loss stats
    wins = df[df[pnl_col] > 0][pnl_col]
    losses = df[df[pnl_col] < 0][pnl_col]

    n_trades = int(len(df))
    winrate = float((len(wins) / n_trades) * 100.0) if n_trades > 0 else 0.0

    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(-losses.sum()) if len(losses) > 0 else 0.0

    profit_factor = float(_safe_div(gross_profit, gross_loss)) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0  # negative number

    expectancy = float(df[pnl_col].mean()) if n_trades > 0 else 0.0

    # Sharpe (approx using "daily" equity by forward-fill)
    # Convert trade equity into daily equity
    daily_eq = equity.resample("D").last().ffill()
    daily_rets = daily_eq.pct_change().dropna()
    sharpe = _annualized_sharpe(daily_rets)

    return {
        "CAGR": round(cagr, 4),
        "MaxDrawdown": round(max_dd, 4),
        "Trades": n_trades,
        "WinRate": round(winrate, 4),
        "ProfitFactor": round(profit_factor, 4),
        "AvgWin": round(avg_win, 4),
        "AvgLoss": round(avg_loss, 4),
        "Expectancy": round(expectancy, 4),
        "Sharpe": round(sharpe, 4),
    }


def monthly_returns_table(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Takes equity series and returns monthly % returns pivot.
    """
    if equity_curve is None or len(equity_curve) < 2:
        return pd.DataFrame()

    eq = equity_curve.resample("M").last().ffill()
    rets = eq.pct_change().dropna() * 100.0
    df = rets.to_frame("ret")
    df["Year"] = df.index.year
    df["Month"] = df.index.strftime("%b")

    pivot = df.pivot_table(index="Year", columns="Month", values="ret", aggfunc="sum")

    # reorder months
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m in month_order:
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[month_order]

    return pivot.round(2)
