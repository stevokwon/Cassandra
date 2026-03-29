"""Cassandra Dashboard — entry point.

Run with:
    streamlit run dashboard/app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Cassandra",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Cassandra Trading System")
st.write(
    "Use the sidebar to navigate between pages.\n\n"
    "- **Live Monitor** — real-time price chart, signals, and account balance\n"
    "- **Strategy Evaluation** — calibration status, backtest history, shadow pipeline"
)
st.caption("All data refreshes automatically every 60 seconds.")
