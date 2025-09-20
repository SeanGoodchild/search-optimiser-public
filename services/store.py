import streamlit as st

def init():
    if 'last_edit' not in st.session_state:
        st.session_state['last_edit'] = {}
    if 'selected_points' not in st.session_state:
        st.session_state['selected_points'] = {}
    if 'all_strategies' not in st.session_state:
        st.session_state['all_strategies'] = {}
    if 'optimal_result' not in st.session_state:
        st.session_state['optimal_result'] = None

def set_last_edit(edit: dict) -> None:
    st.session_state['last_edit'] = edit

def get_last_edit() -> str:
    return st.session_state['last_edit']

def set_selected_point(strategy_id: str, x: float, y: float) -> None:
    st.session_state['selected_points'][strategy_id] = {'x': x, 'y': y}

def get_selected_point(strategy_id: str) -> dict:
    return st.session_state['selected_points'].get(strategy_id, {})

def set_strategy(strategy_id: str, strategy_data: dict) -> None:
    if strategy_id not in st.session_state['all_strategies']:
        st.session_state['all_strategies'][strategy_id] = strategy_data
    else:
        st.session_state['all_strategies'][strategy_id].update(strategy_data)

def get_strategy(strategy_id: str) -> dict:
    return st.session_state['all_strategies'].get(strategy_id, {})

def get_all_strategies() -> dict:
    return st.session_state['all_strategies']

def set_result(optimal_result: dict) -> None:
    st.session_state['optimal_result'] = optimal_result

def get_result():
    return st.session_state['optimal_result']


def init_strategy(strategy_id: str, name: str) -> None:
    if strategy_id not in st.session_state['all_strategies']:
        st.session_state['all_strategies'][strategy_id] = {
            'id': strategy_id,
            'name': name,
            'cost_points': [],
            'conversion_points': [],
            'target_cpa_points': [],
        }