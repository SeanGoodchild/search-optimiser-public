
import streamlit as st
from domain.frontier import sensible_increment

def increment_picker(xmin: float, xmax: float, key: str, preset: float | None = None):
    default = float(preset or sensible_increment(xmin, xmax))
    st.caption("Choose cost increment for selection/optimization")
    return st.number_input("Increment", min_value=0.000001, value=default, step=default, key=key, help="cost will snap to this step when selecting/optimizing.")
