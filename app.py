import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import inspect

from data import load_daily_data
from backtester import run_backtest
from metrics import compute_metrics, monthly_returns_table, compute_equity_curve


st.set_page_config(page_title="EOD Backtesting Workbench (NIFTY & BANKNIFTY)", layout="wide")

st.title("EOD Backtesting Workbench — NIFTY & BANKNIFTY")
st.caption("Signals on close (t), execution at next open (t+1). Long-only V1. Yahoo Finance data.")


# -----------------------------
# Formatting helpers
# -----------------------------
def pct(x):
    try:
        return f"{float(x):.2f}%"
    except:
        return str(x)


def money(x):
    try:
        return f"{float(x):,.2f}"
    except:
        return str(x)


# -----------------------------
# Sidebar UI
# -----------------------------
with st.sidebar:
    st.header("Universe")
    use_nifty = st.checkbox("NIFTY 50", value=True)
    use_bank = st.checkbox("BANKNIFTY", value=True)

    st.header("Date Range")
    start = st.date_input("Start", value=date(2015, 1, 1))
    end = st.date_input("End", value=date.today())

    st.header("Sizing & Costs")
    capital_per_trade = st.number_input("Fixed capital per trade (₹)", min_value=10_000, value=500_000, step=10_000)
    max_parallel = st.selectbox("Max simultaneous positions", [1, 2], index=0)

    slippage_bps = st.number_input("Slippage (bps per side)", min_value=0.0, value=2.0, step=0.5)
    brokerage_per_order = st.number_input("Brokerage per order (₹)", min_value=0.0, value=20.0, step=1.0)

    st.divider()
    st.subheader("Entry Rule Builder")

    entry_conditions = st.multiselect(
        "Entry conditions (AND)",
        [
            "Breakout: Close > Highest(High, N)",
            "Trend: EMA(fast) > EMA(slow)",
            "Momentum: RSI(period) > X",
        ],
        default=["Trend: EMA(fast) > EMA(slow)"],
    )

    breakout_n = st.number_input("Breakout N", min_value=5, value=20, step=1)
    ema_fast = st.number_input("EMA fast", min_value=2, value=20, step=1)
    ema_slow = st.number_input("EMA slow", min_value=5, value=50, step=1)

    rsi_period = st.number_input("RSI period", min_value=2, value=14, step=1)
    rsi_x = st.number_input("RSI threshold X", min_value=10, max_value=90, value=55, step=1)

    st.divider()
    st.subheader("Exit Rule Builder")

    exit_conditions = st.multiselect(
        "Exit conditions (OR)",
        [
            "Exit on Trend flip: EMA(fast) < EMA(slow)",
            "ATR trailing stop",
            "Stoploss: % from entry",
            "Time exit: after K bars",
        ],
        default=["ATR trailing stop"],
    )

    time_exit_k = st.number_input("Time exit K bars", min_value=1, value=15, step=1)
    atr_period = st.number_input("ATR period", min_value=2, value=14, step=1)
    atr_mult = st.number_input("ATR multiple (trail)", min_value=0.5, value=3.0, step=0.25)

    stoploss_pct = st.number_input("Stoploss %", min_value=0.1, value=2.0, step=0.1)

    st.divider()
    run_btn = st.button("Run Backtest", type="primary")


# -----------------------------
# Build symbol list
# -----------------------------
symbols = []
if use_nifty:
    symbols.append("NIFTY")
if use_bank:
    symbols.append("BANKNIFTY")

if len(symbols) == 0:
    st.warning("Select at least one symbol.")
    st.stop()


# -----------------------------
# Build config dicts
# -----------------------------
entry_cfg = {
    "use_breakout": "Breakout: Close > Highest(High, N)" in entry_conditions,
    "breakout_n": int(breakout_n),
    "use_trend": "Trend: EMA(fast) > EMA(slow)" in entry_conditions,
    "ema_fast": int(ema_fast),
    "ema_slow": int(ema_slow),
    "use_rsi": "Momentum: RSI(period) > X" in entry_conditions,
    "rsi_period": int(rsi_period),
    "rsi_x": float(rsi_x),
}

exit_cfg = {
    "exit_on_trend_flip": "Exit on Trend flip: EMA(fast) < EMA(slow)" in exit_conditions,
    "atr_trailing": "ATR trailing stop" in exit_conditions,
    "atr_period": int(atr_period),
    "atr_mult": float(atr_mult),
    "stoploss": "Stoploss: % from entry" in exit_conditions,
    "stoploss_pct": float(stoploss_pct),
    "time_exit": "Time exit: after K bars" in exit_conditions,
    "time_exit_k": int(time_exit_k),
}

sim_cfg = {
    "capital_per_trade": float(capital_per_trade),
    "max_parallel": int(max_parallel),
    "slippage_bps": float(slippage_bps),
    "brokerage_per_order": float(brokerage_per_order),
}


# -----------------------------
# Robust runner for different backtester signatures
# -----------------------------
def call_run_backtest(symbol, data, entry_cfg, exit_cfg, sim_cfg):
    """
    Supports multiple versions of run_backtest() without breaking.
    Tries:
    1) Keyword args
    2) Positional args
    3) Other param names (data/df)
    """
    sig = inspect.signature(run_backtest)
    params = list(sig.parameters.keys())

    # Try keyword style
    try:
        kwargs = {}
        for p in params:
            if p in ("symbol", "sym", "ticker"):
                kwargs[p] = symbol
            elif p in ("data", "df", "prices"):
                kwargs[p] = data
            elif p in ("entry_cfg", "entry_config", "entry"):
                kwargs[p] = entry_cfg
            elif p in ("exit_cfg", "exit_config", "exit"):
                kwargs[p] = exit_cfg
            elif p in ("sim_cfg", "sim_config", "sim", "config"):
                kwargs[p] = sim_cfg

        if len(kwargs) > 0:
            return run_backtest(**kwargs)
    except Exception:
        pass

    # Fallback positional
    try:
        return run_backtest(symbol, data, entry_cfg, exit_cfg, sim_cfg)
    except Exception as e:
        raise e


# -----------------------------
# Main run
# -----------------------------
if run_btn:
    with st.spinner("Downloading data & running backtest..."):
        all_trades = []

        for sym in symbols:
            df = load_daily_data(sym, pd.Timestamp(start), pd.Timestamp(end), source_choice="Yahoo Finance")

            if df is None or df.empty:
                st.warning(f"No data returned for {sym}. Skipping.")
                continue

            trades = call_run_backtest(sym, df, entry_cfg, exit_cfg, sim_cfg)

            # Normalize trades output
            if trades is None:
                continue

            if isinstance(trades, list):
                trades = pd.DataFrame(trades)

            if isinstance(trades, pd.DataFrame) and len(trades) > 0:
                all_trades.append(trades)

        if len(all_trades) == 0:
            st.error("No trades generated. Try loosening entry rules (remove RSI / breakout etc).")
            st.stop()

        trades_df = pd.concat(all_trades, ignore_index=True)

        # Force dates if present
        if "EntryDate" in trades_df.columns:
            trades_df["EntryDate"] = pd.to_datetime(trades_df["EntryDate"], errors="coerce")
        if "ExitDate" in trades_df.columns:
            trades_df["ExitDate"] = pd.to_datetime(trades_df["ExitDate"], errors="coerce")

        # Compute equity + metrics (uses your fixed metrics.py)
        equity = compute_equity_curve(trades_df, initial_capital=sim_cfg["capital_per_trade"])
        metrics = compute_metrics(trades_df, initial_capital=sim_cfg["capital_per_trade"])
        monthly = monthly_returns_table(equity)

    st.success("Backtest complete.")

    # -----------------------------
    # Summary
    # -----------------------------
    col1, col2 = st.columns([1.1, 1.0])

    with col1:
        st.subheader("Summary")
        a, b, c, d = st.columns(4)
        a.metric("CAGR", pct(metrics.get("CAGR", 0)))
        b.metric("Max DD", pct(metrics.get("MaxDrawdown", 0)))
        c.metric("Trades", int(metrics.get("Trades", 0)))
        d.metric("Profit Factor", f"{metrics.get('ProfitFactor', 0):.2f}")

        e, f, g, h = st.columns(4)
        e.metric("Win Rate", pct(metrics.get("WinRate", 0)))
        f.metric("Sharpe", f"{metrics.get('Sharpe', 0):.2f}")
        g.metric("Avg Win (₹)", money(metrics.get("AvgWin", 0)))
        h.metric("Avg Loss (₹)", money(metrics.get("AvgLoss", 0)))

        st.caption("Raw metrics JSON")
        st.json(metrics)

    with col2:
        st.subheader("Monthly Returns (%)")
        if monthly is None or monthly.empty:
            st.info("Monthly returns table not available (not enough points).")
        else:
            st.dataframe(monthly, use_container_width=True)

    # -----------------------------
    # Equity curve
    # -----------------------------
    st.subheader("Equity Curve")
    eq_df = pd.DataFrame({"Equity": equity.values}, index=equity.index)
    st.line_chart(eq_df)

    # -----------------------------
    # Trades
    # -----------------------------
    st.subheader("Trades")
    st.dataframe(trades_df, use_container_width=True, height=400)

    csv = trades_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Trades CSV",
        data=csv,
        file_name="backtest_trades.csv",
        mime="text/csv",
    )

else:
    st.info("Set your rules on the left and click **Run Backtest**.")
