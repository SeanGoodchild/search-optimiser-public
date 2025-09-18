
import streamlit as st

SELECTED_KEY = "selected_points"
SCENARIO_KEY = "scenario"
RESULT_KEY   = "opt_result"
ENTITIES_KEY = "entities"

def init():
    if SELECTED_KEY not in st.session_state:
        st.session_state[SELECTED_KEY] = {}
    if SCENARIO_KEY not in st.session_state:
        st.session_state[SCENARIO_KEY] = {"objective": "max_conv", "budget": 10000.0, "target": None, "increment": None}
    if RESULT_KEY not in st.session_state:
        st.session_state[RESULT_KEY] = None
    if ENTITIES_KEY not in st.session_state:
        st.session_state[ENTITIES_KEY] = None

def get_selected():
    return st.session_state[SELECTED_KEY]

def set_selected(entity_id: str, cost: float, meta: dict | None = None):
    st.session_state[SELECTED_KEY][entity_id] = {"cost": float(cost), "meta": (meta or {})}

def get_scenario() -> dict:
    return st.session_state[SCENARIO_KEY]

def set_scenario(**updates):
    st.session_state[SCENARIO_KEY].update(updates)

def set_result(res):
    st.session_state[RESULT_KEY] = res

def get_result():
    return st.session_state[RESULT_KEY]

def get_entities():
    return st.session_state[ENTITIES_KEY]

def set_entities(entities):
    st.session_state[ENTITIES_KEY] = entities
