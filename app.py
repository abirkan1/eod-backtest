import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

from data import load_daily_data
from backtester import run_backtest
from metrics import compute_metrics, monthly_returns_table, compute_equity_curve

st.set_page_config(page_title="EOD Backtesting Workbench (NIFTY & BANKNIFTY)", layout="wide")

st.title("EOD Backtesting Workbench — NIFTY & BANKNIFTY")
st.caption("Signals on close (t), execution at next open (t+1). Long-only V1. Yahoo Finance data.")


# -----------------------------
# UI helpers
# -----------------------------
def _pct(x):
    try:
        return f"{x:.2f}%"
    except:
        return str(x)


def _fmt(x):
    try:
        return f"{x:,.2f}"
    except:
        return str(x)


# -----------------------------
# Sidebar
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
# Build config dicts
# -----------------------------
symbols = []
if use_nifty:
    symbols.append("NIFTY")
if use_bank:
    symbols.append("BANKNIFTY")

if len(symbols) == 0:
    st.warning("Select at least one symbol in Universe.")
    st.stop()

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
# Run
# -----------------------------
if run_btn:
    with st.spinner("Downloading data & running backtest..."):

        all_trades = []
        all_equity = []

        for sym in symbols:
            df = load_daily_data(sym, pd.Timestamp(start), pd.Timestamp(end), source_choice="Yahoo Finance")

            if df is None or df.empty:
                st.warning(f"No data returned for {sym}. Skipping.")
                continue

            trades = run_backtest(
                symbol=sym,
                data=df,
                entry_cfg=entry_cfg,
                exit_cfg=exit_cfg,
                sim_cfg=sim_cfg,
            )

            if trades is None or len(trades) == 0:
                continue

            all_trades.append(trades)

        if len(all_trades) == 0:
            st.error("No trades generated. Try loosening entry rules (e.g. remove RSI filter).")
            st.stop()

        trades_df = pd.concat(all_trades, ignore_index=True)

        # Build equity from trade PnL
        equity = compute_equity_curve(trades_df, initial_capital=sim_cfg["capital_per_trade"])
        metrics = compute_metrics(trades_df, initial_capital=sim_cfg["capital_per_trade"])
        monthly = monthly_returns_table(equity)

    st.success("Backtest complete.")

    # -----------------------------
    # Summary
    # -----------------------------
    col1, col2 = st.columns([1.05, 1.0])

    with col1:
        st.subheader("Summary")

        # show key metrics in clean way
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CAGR", _pct(metrics["CAGR"]))
        m2.metric("Max DD", _pct(metrics["MaxDrawdown"]))
        m3.metric("Trades", int(metrics["Trades"]))
        m4.metric("Profit Factor", f'{metrics["ProfitFactor"]:.2f}')

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Win Rate", _pct(metrics["WinRate"]))
        m6.metric("Sharpe", f'{metrics["Sharpe"]:.2f}')
        m7.metric("Avg Win (₹)", _fmt(metrics["AvgWin"]))
        m8.metric("Avg Loss (₹)", _fmt(metrics["AvgLoss"]))

        st.caption("Raw metrics JSON (for debugging)")
        st.json(metrics)

    with col2:
        st.subheader("Monthly Returns (%)")
        if monthly is None or monthly.empty:
            st.info("Monthly table unavailable (not enough equity points).")
        else:
            st.dataframe(monthly, use_container_width=True)

    # -----------------------------
    # Equity curve
    # -----------------------------
    st.subheader("Equity Curve")
    eq_df = pd.DataFrame({"Equity": equity.values}, index=equity.index)
    st.line_chart(eq_df)

    # -----------------------------
    # Trades table + download
    # -----------------------------
    st.subheader("Trades")
    st.dataframe(trades_df, use_container_width=True, height=350)

    csv = trades_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Trades CSV",
        data=csv,
        file_name="backtest_trades.csv",
        mime="text/csv",
    )

else:
    st.info("Set your rules on the left and click **Run Backtest**.")
