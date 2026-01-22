
import pandas as pd
import numpy as np

def build_entry_signal(
    df: pd.DataFrame,
    templates,
    breakout_n: int,
    ema_fast_col: str,
    ema_slow_col: str,
    rsi_col: str,
    rsi_x: float,
) -> pd.Series:
    """AND across selected templates; signal on day close."""
    sigs = []
    if "Breakout: Close > Highest(High, N)" in templates:
        rolling_max = df["High"].rolling(breakout_n).max().shift(1)  # avoid lookahead
        sigs.append(df["Close"] > rolling_max)
    if "Trend: EMA(fast) > EMA(slow)" in templates:
        sigs.append(df[ema_fast_col] > df[ema_slow_col])
    if "Momentum: RSI(period) > X" in templates:
        sigs.append(df[rsi_col] > rsi_x)

    if not sigs:
        return pd.Series(False, index=df.index)

    out = sigs[0]
    for s in sigs[1:]:
        out = out & s
    return out.fillna(False)

def build_exit_signal(
    df: pd.DataFrame,
    templates,
    time_k: int,
    stop_pct: float,
    atr_col: str,
    atr_mult: float,
    ema_fast_col: str,
    ema_slow_col: str,
) -> pd.Series:
    """
    Exit conditions are OR'ed, but some are evaluated inside backtester (stop %, time, trail) because they require entry price/time.
    Here we return only 'structural' exits that can be known without entry context: Trend flip.
    For other exits, we set flags so backtester knows what to apply.
    """
    # We'll encode which contextual exits are enabled in df attrs via columns.
    df["USE_TIME_EXIT"] = "Time exit: after K bars" in templates
    df["TIME_K"] = time_k
    df["USE_STOP_PCT"] = "Stoploss: % from entry" in templates
    df["STOP_PCT"] = stop_pct
    df["USE_ATR_TRAIL"] = "ATR trailing stop (Chandelier)" in templates
    df["ATR_MULT"] = atr_mult

    sig = pd.Series(False, index=df.index)

    if "Exit on Trend flip: EMA(fast) < EMA(slow)" in templates:
        sig = sig | (df[ema_fast_col] < df[ema_slow_col])

    return sig.fillna(False)
