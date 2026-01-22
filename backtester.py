import pandas as pd
import numpy as np

from indicators import ema, rsi, atr


def _apply_entry_rules(df, entry_cfg):
    """
    Returns a boolean Series entry_signal aligned to df.index
    """
    conds = []

    # Breakout: Close > Highest(High, N)
    if entry_cfg.get("use_breakout", False):
        n = int(entry_cfg.get("breakout_n", 20))
        hh = df["High"].rolling(n).max().shift(1)
        conds.append(df["Close"] > hh)

    # Trend: EMA(fast) > EMA(slow)
    if entry_cfg.get("use_trend", False):
        fast = int(entry_cfg.get("ema_fast", 20))
        slow = int(entry_cfg.get("ema_slow", 50))
        ef = ema(df["Close"], fast)
        es = ema(df["Close"], slow)
        conds.append(ef > es)

    # RSI > X
    if entry_cfg.get("use_rsi", False):
        p = int(entry_cfg.get("rsi_period", 14))
        x = float(entry_cfg.get("rsi_x", 55))
        rv = rsi(df["Close"], p)
        conds.append(rv > x)

    if len(conds) == 0:
        return pd.Series(False, index=df.index)

    out = conds[0].copy()
    for c in conds[1:]:
        out = out & c
    return out.fillna(False)


def _apply_exit_rules(df, exit_cfg):
    """
    Exit rules are applied inside loop because they depend on entry price and dynamic trail.
    We still precompute stuff here.
    """
    out = {}

    # Trend flip
    if exit_cfg.get("exit_on_trend_flip", False):
        fast = int(exit_cfg.get("ema_fast", 20))  # fallback
        slow = int(exit_cfg.get("ema_slow", 50))
        # but our UI passes ema_fast/ema_slow inside entry_cfg mostly.
        # So if missing, it's fine.
        ef = ema(df["Close"], fast)
        es = ema(df["Close"], slow)
        out["trend_flip"] = (ef < es).fillna(False)
    else:
        out["trend_flip"] = pd.Series(False, index=df.index)

    # ATR
    if exit_cfg.get("atr_trailing", False):
        p = int(exit_cfg.get("atr_period", 14))
        out["atr"] = atr(df, p)
    else:
        out["atr"] = pd.Series(np.nan, index=df.index)

    return out


def _run_single_symbol(symbol, df, entry_cfg, exit_cfg, sim_cfg):
    """
    Long-only, signal on close, enter next open.
    """
    df = df.copy()
    df = df.sort_index()

    # precompute entry signal
    entry_signal = _apply_entry_rules(df, entry_cfg)

    # precompute exit helpers
    exit_helpers = _apply_exit_rules(df, exit_cfg)

    capital_per_trade = float(sim_cfg.get("capital_per_trade", 500000))
    slippage_bps = float(sim_cfg.get("slippage_bps", 2.0))
    brokerage_per_order = float(sim_cfg.get("brokerage_per_order", 20.0))

    max_parallel = int(sim_cfg.get("max_parallel", 1))

    trades = []
    open_positions = []

    # slippage helper: price * bps/10000
    def slip(price):
        return price * (slippage_bps / 10000.0)

    idx = df.index

    for i in range(1, len(idx) - 1):
        dt = idx[i]

        # ----- manage exits first -----
        new_open_positions = []
        for pos in open_positions:
            entry_dt = pos["EntryDate"]
            entry_price = pos["EntryPrice"]
            qty = pos["Qty"]

            # today's close for checks
            close_price = float(df.loc[dt, "Close"])

            exit_now = False
            exit_reason = None

            # Stoploss
            if exit_cfg.get("stoploss", False):
                sl_pct = float(exit_cfg.get("stoploss_pct", 2.0)) / 100.0
                if close_price <= entry_price * (1 - sl_pct):
                    exit_now = True
                    exit_reason = "Stoploss"

            # Time exit
            if not exit_now and exit_cfg.get("time_exit", False):
                k = int(exit_cfg.get("time_exit_k", 15))
                bars_held = (idx.get_loc(dt) - idx.get_loc(entry_dt))
                if bars_held >= k:
                    exit_now = True
                    exit_reason = "TimeExit"

            # Trend flip
            if not exit_now and exit_cfg.get("exit_on_trend_flip", False):
                tf = bool(exit_helpers["trend_flip"].loc[dt])
                if tf:
                    exit_now = True
                    exit_reason = "TrendFlip"

            # ATR trailing stop
            if not exit_now and exit_cfg.get("atr_trailing", False):
                atr_mult = float(exit_cfg.get("atr_mult", 3.0))
                atr_val = exit_helpers["atr"].loc[dt]
                if pd.notna(atr_val):
                    # Trail stop based on highest close since entry
                    pos["HighestClose"] = max(pos.get("HighestClose", entry_price), close_price)
                    trail = pos["HighestClose"] - atr_mult * float(atr_val)
                    if close_price <= trail:
                        exit_now = True
                        exit_reason = "ATR_Trail"

            if exit_now:
                # exit at next open (t+1 open)
                exit_dt = idx[i + 1]
                exit_px_raw = float(df.loc[exit_dt, "Open"])
                exit_px = exit_px_raw - slip(exit_px_raw)

                gross_pnl = (exit_px - entry_price) * qty
                cost = brokerage_per_order * 2  # entry+exit brokerage
                net_pnl = gross_pnl - cost

                trades.append({
                    "Symbol": symbol,
                    "EntryDate": entry_dt,
                    "ExitDate": exit_dt,
                    "EntryPrice": entry_price,
                    "ExitPrice": exit_px,
                    "Qty": qty,
                    "GrossPnL": gross_pnl,
                    "Cost": cost,
                    "NetPnL": net_pnl,
                    "ExitReason": exit_reason
                })
            else:
                new_open_positions.append(pos)

        open_positions = new_open_positions

        # ----- entries (signal on close, entry at next open) -----
        if entry_signal.loc[dt]:
            if len(open_positions) < max_parallel:
                entry_dt = idx[i + 1]
                entry_px_raw = float(df.loc[entry_dt, "Open"])
                entry_px = entry_px_raw + slip(entry_px_raw)

                qty = capital_per_trade / entry_px

                open_positions.append({
                    "EntryDate": entry_dt,
                    "EntryPrice": entry_px,
                    "Qty": qty,
                    "HighestClose": entry_px
                })

    return pd.DataFrame(trades)


def run_backtest(symbol, data, entry_cfg, exit_cfg, sim_cfg):
    """
    ✅ Supports BOTH formats:

    1) Single symbol:
       run_backtest("NIFTY", df, entry_cfg, exit_cfg, sim_cfg)

    2) Multi symbol dict:
       run_backtest("ALL", {"NIFTY": df1, "BANKNIFTY": df2}, entry_cfg, exit_cfg, sim_cfg)
    """

    # if dict passed => multi-symbol
    if isinstance(data, dict):
        all_trades = []
        for sym, df in data.items():
            t = _run_single_symbol(sym, df, entry_cfg, exit_cfg, sim_cfg)
            if t is not None and len(t) > 0:
                all_trades.append(t)
        if len(all_trades) == 0:
            return pd.DataFrame()
        return pd.concat(all_trades, ignore_index=True)

    # else single df
    return _run_single_symbol(symbol, data, entry_cfg, exit_cfg, sim_cfg)
