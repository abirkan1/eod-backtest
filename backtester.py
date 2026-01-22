
import pandas as pd
import numpy as np

def _apply_slippage(price: float, side: str, bps: float) -> float:
    # side: 'buy' or 'sell'
    slip = bps / 10000.0
    if side == "buy":
        return price * (1.0 + slip)
    return price * (1.0 - slip)

def run_backtest(df_map, capital_per_trade: float, max_parallel: int, slippage_bps: float, brokerage_per_order: float):
    """
    Long-only, next-open execution.
    One position per instrument.
    Portfolio allows up to max_parallel concurrent open positions across instruments.

    df_map: dict(sym -> df with Open/High/Low/Close/Volume + ENTRY_SIG + EXIT_SIG + contextual columns)
    Returns: trades_df, equity_df(index=Date)
    """
    # Align dates
    all_dates = sorted(set().union(*[set(df.index) for df in df_map.values()]))
    all_dates = pd.DatetimeIndex(all_dates).sort_values()

    # State per symbol
    pos = {sym: {"open": False, "entry_price": np.nan, "entry_date": None, "bars": 0, "trail": np.nan, "qty": 0.0} for sym in df_map}
    trades = []

    equity = []
    cash = 0.0  # we treat each trade as allocated capital_per_trade; equity is sum of per-trade pnl on allocated capital (not margin accounting)
    # Start equity baseline as 0 PnL; users interpret as "P&L on deployed system". We also compute "equity" starting at 1.0 for returns.
    eq_val = 1.0

    for i, d in enumerate(all_dates):
        # Mark-to-market with close (end of day equity)
        day_pnl = 0.0
        deployed = 0.0

        for sym, df in df_map.items():
            if d not in df.index:
                continue
            row = df.loc[d]
            if pos[sym]["open"]:
                # mark-to-market on close using qty
                deployed += capital_per_trade
                day_pnl += (row["Close"] - pos[sym]["entry_price"]) * pos[sym]["qty"]

        # equity = 1 + total return on deployed capital over time with reinvest? For simplicity, compute cumulative PnL on fixed allocation baseline:
        # We'll track cumulative PnL and convert to equity curve assuming initial 1.0 and adding daily return normalized by (capital_per_trade * max_parallel) or by (deployed)
        # Use denominator = max_parallel * capital_per_trade to make equity comparable.
        denom = max_parallel * capital_per_trade if max_parallel * capital_per_trade > 0 else 1.0
        eq_val = 1.0 + (day_pnl / denom)
        equity.append({"Date": d, "Equity": eq_val, "OpenPnL": day_pnl, "Deployed": deployed})

        # Now decide exits/entries based on signals from CLOSE of day d, executed at OPEN of next day (d_next)
        if i == len(all_dates) - 1:
            break
        d_next = all_dates[i + 1]

        # EXITS first (to free slots)
        for sym, df in df_map.items():
            if not pos[sym]["open"]:
                continue
            if d not in df.index or d_next not in df.index:
                continue
            row = df.loc[d]
            next_open = float(df.loc[d_next, "Open"])

            # Update bars held
            pos[sym]["bars"] += 1

            # Update ATR trail (chandelier): trail = max(trail, close - ATR_MULT*ATR)
            if bool(row.get("USE_ATR_TRAIL", False)):
                atr = float(row.get("ATR", np.nan))
                mult = float(row.get("ATR_MULT", 3.0))
                if np.isfinite(atr):
                    candidate = float(row["Close"] - mult * atr)
                    if not np.isfinite(pos[sym]["trail"]):
                        pos[sym]["trail"] = candidate
                    else:
                        pos[sym]["trail"] = max(pos[sym]["trail"], candidate)

            exit_reason = None

            # Structural exit signal
            if bool(row.get("EXIT_SIG", False)):
                exit_reason = "rule_exit"

            # Time exit
            if exit_reason is None and bool(row.get("USE_TIME_EXIT", False)):
                k = int(row.get("TIME_K", 1))
                if pos[sym]["bars"] >= k:
                    exit_reason = "time_exit"

            # Stoploss % (based on close vs entry) triggers on day close, execute next open
            if exit_reason is None and bool(row.get("USE_STOP_PCT", False)):
                stop_pct = float(row.get("STOP_PCT", 0.0)) / 100.0
                if stop_pct > 0 and float(row["Close"]) <= pos[sym]["entry_price"] * (1.0 - stop_pct):
                    exit_reason = "stop_pct"

            # ATR trailing stop triggers if close <= trail
            if exit_reason is None and bool(row.get("USE_ATR_TRAIL", False)) and np.isfinite(pos[sym]["trail"]):
                if float(row["Close"]) <= float(pos[sym]["trail"]):
                    exit_reason = "atr_trail"

            if exit_reason is not None:
                fill = _apply_slippage(next_open, "sell", slippage_bps)
                pnl = (fill - pos[sym]["entry_price"]) * pos[sym]["qty"]
                costs = 2.0 * brokerage_per_order  # entry+exit orders
                pnl_net = pnl - costs

                trades.append({
                    "Symbol": sym,
                    "EntryDate": pos[sym]["entry_date"],
                    "ExitDate": d_next,
                    "EntryPrice": pos[sym]["entry_price"],
                    "ExitPrice": fill,
                    "Qty": pos[sym]["qty"],
                    "GrossPnL": pnl,
                    "Costs": costs,
                    "NetPnL": pnl_net,
                    "BarsHeld": pos[sym]["bars"],
                    "ExitReason": exit_reason,
                })
                # reset position
                pos[sym] = {"open": False, "entry_price": np.nan, "entry_date": None, "bars": 0, "trail": np.nan, "qty": 0.0}

        # ENTRIES (after exits)
        open_positions = sum(1 for s in pos if pos[s]["open"])
        slots = max(0, max_parallel - open_positions)

        if slots > 0:
            # Candidates: signals true on day d, enter next open
            candidates = []
            for sym, df in df_map.items():
                if pos[sym]["open"]:
                    continue
                if d not in df.index or d_next not in df.index:
                    continue
                if bool(df.loc[d, "ENTRY_SIG"]):
                    # Simple scoring: stronger RSI gets priority if available, else just add
                    score = float(df.loc[d].get("RSI", 50.0))
                    candidates.append((score, sym))
            candidates.sort(reverse=True)

            for _, sym in candidates[:slots]:
                df = df_map[sym]
                next_open = float(df.loc[d_next, "Open"])
                fill = _apply_slippage(next_open, "buy", slippage_bps)
                qty = capital_per_trade / fill if fill > 0 else 0.0
                pos[sym] = {"open": True, "entry_price": fill, "entry_date": d_next, "bars": 0, "trail": np.nan, "qty": qty}

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity).set_index("Date")
    return trades_df, equity_df
