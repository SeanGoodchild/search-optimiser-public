
import streamlit as st
from utils import app_io
from ui import charts
# Maybe buiild an app def at the start that defines the layout that we will populate later
# Compare altiar and plotly and bokeh for interactivity
# Look into using callback fiunctions for interactivity, rather than checking for a change all the time
# look at all the page_config options. Start with sidebar open. Get Help button, etc
# there are options for getting and setting query params from the URL
# session_state is totally lost on hard manual refresh (cmd+R). Wonder about storing data with the cache to persist in each instance?
# consider duckdb for parsing and cleaning the initial dataframe? Maybe.
# st.container is like flexbox div. st.expander is a collapsible container. 
# Look into st.html (for basic css maybe? Background colour?)

st.set_page_config(page_title="PMG Budget Optimizer", layout="wide", initial_sidebar_state="collapsed")
# Inject global CSS


def main():
    data_upload_section()
    uploaded_data = st.session_state.get('uploaded_data')
    init_app(uploaded_data)
    sidebar()
    header_section()
    strategy_section(uploaded_data)


def init_app(uploaded_data: dict):
    if 'chart_states' not in st.session_state:
        st.session_state['chart_states'] = {}
        for strategy_id, strategy_data in uploaded_data.items():
            chart_id = f"chart_{strategy_id}"
            st.session_state['chart_states'][chart_id] = {
                "selected_point_index": strategy_data['starting_point_index']
            }


def data_upload_section():
    if st.session_state.get('uploaded_data') is not None:
        return

    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="file_uploader")
    use_dummy = st.button("Use Dummy Data", key="use_dummy")
    
    if use_dummy:
        file_path = "data/sample.csv"
    elif uploaded_file is not None:
        file_path = uploaded_file
    else:
        st.info("Please upload a CSV file or use the dummy data.")
        st.stop() 
    
    uploaded_data = app_io.import_upload(file_path)
    st.session_state['uploaded_data'] = uploaded_data
    st.rerun()


def header_section():
    st.title("PMG Budget Optimizer")
    st.markdown("Select custom points on the curves, or allow the optimizer to choose for you.")


def strategy_section(uploaded_data: dict):
    for strategy_id, strategy_data in uploaded_data.items():
        st.subheader(strategy_data['name'])
        starting_point_index = strategy_data['starting_point_index']
        current_chart_state = st.session_state['chart_states'].get(f"chart_{strategy_id}")
        chart_events = charts.build_startegy_chart(strategy_data, current_chart_state)
        handle_events(strategy_id, current_chart_state, chart_events)
        st.metric('Cost', strategy_data['x_fit'][starting_point_index], delta=None, delta_color="normal")
        st.metric('Conversions', strategy_data['y_fit'][starting_point_index], delta=None, delta_color="normal")
        st.divider()


def handle_events(strategy_id: str, current_chart_state: dict, events: dict) -> None:
    if not events:
        return
    e = events[0]  # or handle multi-select
    idx = e.get("pointIndex")
    if current_chart_state is None or idx != current_chart_state.get('selected_point_index'):
        st.session_state[f"chart_{strategy_id}"] = current_chart_state
        x_clicked = e.get("x")
        y_clicked = e.get("y")
        st.toast(f"Selected cost £{x_clicked:.2f} with {y_clicked:.2f}")
        st.session_state['chart_states'][f"chart_{strategy_id}"]['selected_point_index'] = idx
        st.rerun()



def sidebar(expanded: bool = True):
    with st.sidebar:
        st.sidebar.expander("Help", expanded=True).markdown("""
            ### How to use this app
            1. Upload a CSV file in the sidebar, or use the dummy data.
            2. Select points on the curves to set your desired cost and conversions.
            3. Review the optimized budget allocation in the Optimize section.
        """)




if __name__ == "__main__":
    main()
