
import streamlit as st
from state import store

st.set_page_config(page_title="Budget Optimizer", layout="wide")
store.init()

st.title("Budget Optimizer")
st.markdown("Navigate pages to pick points, run optimization, and export results.")

st.sidebar.header("Global Options")

if store.get_result() is not None:
    res = store.get_result()
    total_cost = sum(v['cost'] for v in res.values())
    total_conv  = sum(v.get('conv', 0.0) for v in res.values())
    st.success(f"Current solution: total cost £{total_cost:,.0f}, conversions {total_conv:,.1f}")
else:
    st.info("No optimization run yet. Head to **Curves** or **Optimize** pages.")
