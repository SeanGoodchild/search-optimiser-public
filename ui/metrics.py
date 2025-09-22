import streamlit as st

def build_metric_card(name: str, strategy_data: dict, delta=None) -> None:
    starting_point_index = strategy_data['starting_point_index']
    st.metric(name, strategy_data['x_fit'][starting_point_index], delta=None, delta_color="normal")
