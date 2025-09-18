
import streamlit as st
from state import store
from services.ingest import load_raw_df, prepare_entities
from services.optimize import discretize_entities, solve_mckp

store.init()
st.header("Optimize")

raw = load_raw_df()
entities = prepare_entities(raw)
store.set_entities(entities)

with st.form("scenario"):
    c1, c2, c3 = st.columns(3)
    objective = c1.selectbox("Objective", ["max_conv","min_cost_for_target"], index=0)
    budget    = c2.number_input("Budget (ignored for min_cost_for_target)", min_value=0.0, value=float(store.get_scenario().get("budget", 10000.0)), step=100.0)
    target    = c3.number_input("Target conversions (only used for min_cost_for_target)", min_value=0.0, value=float(store.get_scenario().get("target") or 0.0), step=10.0)
    inc       = st.number_input("cost increment (global)", min_value=0.000001, value=float(store.get_scenario().get("increment") or 1.0), step=float(store.get_scenario().get("increment") or 1.0))
    run = st.form_submit_button("Run optimization")

if run:
    store.set_scenario(objective=objective, budget=budget, target=(target if target>0 else None), increment=inc)
    disc = discretize_entities(entities, increment=inc)
    with st.spinner("Solving…"):
        res = solve_mckp(disc, objective=objective, budget=budget, target=(target if target>0 else None))
    store.set_result(res)
    st.success("Optimization complete.")
    total_cost = sum(v['cost'] for v in res.values())
    total_conv  = sum(v['conv'] for v in res.values())
    c1, c2 = st.columns(2)
    c1.metric("Total cost", f"£{total_cost:,.0f}")
    c2.metric("Total Conversions", f"{total_conv:,.2f}")
    st.write(res)
else:
    st.info("Set the scenario and click **Run optimization**.")
