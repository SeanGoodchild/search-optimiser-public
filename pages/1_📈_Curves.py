
import streamlit as st
from state import store
from services.ingest import load_data
from ui.charts import curve_figure
from domain.frontier import sensible_increment, densify_to_increment
from ui.widgets import increment_picker
from streamlit_plotly_events import plotly_events

PLOTLY_EVENTS_AVAILABLE = True

store.init()
st.header("Curves & Selection")

input_df = load_data()
print(input_df)
# Convert to dict of entities
dict_entities = input_df.to_dict()
print(dict_entities)

store.set_entities(dict_entities)

xmin = min(input_df['cost'])
xmax = max(input_df['cost']) 
inc = increment_picker(float(xmin), float(xmax), key="curves_inc", preset=store.get_scenario().get("increment"))
store.set_scenario(increment=inc)

cols = st.columns(3)
for column, values in enumerate(dict_entities.items()):
    print(column)
    print(values)

    st.subheader(eid)
    dense = densify_to_increment(df, inc)
    sel = store.get_selected().get(eid, {}).get("cost")
    fig = curve_figure(eid, dense, selected_cost=sel)
    events = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key=f"plt_{eid}")
    st.plotly_chart(fig, use_container_width=True)
    if events:
        x = float(events[0]['x'])
        snapped = round(x / inc) * inc
        snapped = max(float(dense['cost'].min()), min(float(dense['cost'].max()), snapped))
        store.set_selected(eid, snapped, meta={"source":"click"})
    else:
        st.caption("Tip: install `streamlit-plotly-events` for click-to-select.")
    with st.form(f"form_{eid}"):
        val = st.number_input("Selected cost", min_value=float(dense['cost'].min()), max_value=float(dense['cost'].max()), step=float(inc), value=float(store.get_selected().get(eid, {}).get("cost", float(dense['cost'].min()))), key=f"num_{eid}")
        submitted = st.form_submit_button("Set")
        if submitted:
            store.set_selected(eid, float(val), meta={"source":"input"})
