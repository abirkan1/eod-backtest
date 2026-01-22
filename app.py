
import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from data import load_daily_data
from indicators import ema, rsi, atr
from rules import build_entry_signal, build_exit_signal
from backtester import run_backtest
from metrics import compute_metrics, monthly_returns_table
st.set_page_config(page_title="EOD Backtesting Workbench (NIFTY & BANKNIFTY)", layout="wide")

st.title("EOD Backtesting Workbench — NIFTY & BANKNIFTY")
st.caption("Signals on close (t), execution at next open (t+1). Long-only V1.")

with st.sidebar:
    st.header("Universe")
    use_nifty = st.checkbox("NIFTY 50", value=True)
    use_bank = st.checkbox("BANKNIFTY", value=True)

    st.header("Date Range")
    start = st.date_input("Start", value=date(2015, 1, 1))
    end = st.date_input("End", value=date.today())

    st.header("Sizing & Costs")
    capital_per_trade = st.number_input("Fixed capital per trade (₹)", min_value=10000, value=500000, step=10000)
    max_parallel = st.selectbox("Max simultaneous positions", [1, 2], index=1)
    slippage_bps = st.number_input("Slippage (bps per side)", min_value=0.0, value=2.0, step=0.5)
    brokerage_per_order = st.number_input("Brokerage per order (₹)", min_value=0.0, value=20.0, step=1.0)

    st.header("Data Source")
    source = st.selectbox("Source", ["Auto (Kite if configured, else Yahoo)", "Zerodha Kite (if configured)", "Yahoo Finance"], index=0)
    st.caption("Kite needs env vars: KITE_API_KEY, KITE_ACCESS_TOKEN. (Secret not needed for historical fetch if token is valid).")

    st.header("Entry Rule Builder")
    entry_templates = st.multiselect(
        "Entry conditions (AND)",
        ["Breakout: Close > Highest(High, N)",
         "Trend: EMA(fast) > EMA(slow)",
         "Momentum: RSI(period) > X"],
        default=["Trend: EMA(fast) > EMA(slow)"]
    )
    colA, colB = st.columns(2)
    with colA:
        breakout_n = st.number_input("Breakout N", min_value=5, value=20, step=1)
        ema_fast = st.number_input("EMA fast", min_value=2, value=21, step=1)
        rsi_period = st.number_input("RSI period", min_value=2, value=14, step=1)
    with colB:
        ema_slow = st.number_input("EMA slow", min_value=3, value=55, step=1)
        rsi_x = st.number_input("RSI threshold X", min_value=1, value=55, step=1)

    st.header("Exit Rule Builder")
    exit_templates = st.multiselect(
        "Exit conditions (OR)",
        ["Time exit: after K bars",
         "Stoploss: % from entry",
         "ATR trailing stop (Chandelier)",
         "Exit on Trend flip: EMA(fast) < EMA(slow)"],
        default=["Stoploss: % from entry", "ATR trailing stop (Chandelier)"]
    )
    colC, colD = st.columns(2)
    with colC:
        time_k = st.number_input("Time exit K bars", min_value=1, value=20, step=1)
        stop_pct = st.number_input("Stoploss %", min_value=0.1, value=2.0, step=0.1)
        atr_period = st.number_input("ATR period", min_value=2, value=14, step=1)
    with colD:
        atr_mult = st.number_input("ATR multiple (trail)", min_value=0.5, value=3.0, step=0.25)

    run_btn = st.button("Run Backtest", type="primary")

symbols = []
if use_nifty:
    symbols.append("NIFTY")
if use_bank:
    symbols.append("BANKNIFTY")

if not symbols:
    st.warning("Select at least one instrument.")
    st.stop()

if run_btn:
    with st.spinner("Loading data..."):
        df_map = {}
        for sym in symbols:
            data = load_daily_data(sym, pd.Timestamp(start), pd.Timestamp(end), source_choice=source)
            if data is None or data.empty:
                st.error(f"No data for {sym}. Check data source configuration.")
                st.stop()
            df_map[sym] = data

    with st.spinner("Building signals..."):
        for sym, data in df_map.items():
            # Precompute indicators
            data["EMA_FAST"] = ema(data["Close"], int(ema_fast))
            data["EMA_SLOW"] = ema(data["Close"], int(ema_slow))
            data["RSI"] = rsi(data["Close"], int(rsi_period))
            data["ATR"] = atr(data, int(atr_period))

            entry_sig = build_entry_signal(
                data,
                templates=entry_templates,
                breakout_n=int(breakout_n),
                ema_fast_col="EMA_FAST",
                ema_slow_col="EMA_SLOW",
                rsi_col="RSI",
                rsi_x=float(rsi_x),
            )
            data["ENTRY_SIG"] = entry_sig.astype(bool)

            exit_sig = build_exit_signal(
                data,
                templates=exit_templates,
                time_k=int(time_k),
                stop_pct=float(stop_pct),
                atr_col="ATR",
                atr_mult=float(atr_mult),
                ema_fast_col="EMA_FAST",
                ema_slow_col="EMA_SLOW",
            )
            data["EXIT_SIG"] = exit_sig.astype(bool)
            df_map[sym] = data

    with st.spinner("Running portfolio backtest..."):
        trades, equity = run_backtest(
            df_map,
            capital_per_trade=float(capital_per_trade),
            max_parallel=int(max_parallel),
            slippage_bps=float(slippage_bps),
            brokerage_per_order=float(brokerage_per_order),
        )

    st.success("Backtest complete.")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Summary")
        if equity.empty:
            st.warning("No trades generated for this configuration.")
        else:
            metrics = compute_metrics(equity, trades)
            st.json(metrics)

        st.subheader("Equity Curve")
        if not equity.empty:
            chart_df = equity[["Equity"]].copy()
            st.line_chart(chart_df)

        st.subheader("Drawdown")
        if not equity.empty:
            dd = equity["Equity"] / equity["Equity"].cummax() - 1.0
            st.line_chart(dd.rename("Drawdown"))

    with right:
        st.subheader("Monthly Returns")
        if not equity.empty:
            mtab = monthly_returns_table(equity)
            st.dataframe(mtab, use_container_width=True)

        st.subheader("Trades")
        if trades.empty:
            st.info("No trades.")
        else:
            st.dataframe(trades, use_container_width=True, height=420)

            csv_trades = trades.to_csv(index=False).encode("utf-8")
            st.download_button("Download trades CSV", data=csv_trades, file_name="trades.csv", mime="text/csv")

    st.subheader("Daily Equity (download)")
    if not equity.empty:
        st.dataframe(equity.tail(200), use_container_width=True, height=240)
        csv_eq = equity.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("Download equity CSV", data=csv_eq, file_name="equity_curve.csv", mime="text/csv")
else:
    st.info("Set rules on the left and click **Run Backtest**.")
    st.markdown("""
**Notes**
- This is EOD, realistic execution: signal on close, fill on next open.
- Long-only V1. (Short support can be added next.)
- For Kite: set environment variables `KITE_API_KEY` and `KITE_ACCESS_TOKEN` before running.
""")
