
import streamlit as st
from utils import app_io
from ui import charts, custom_css
# Maybe buiild an app def at the start that defines the layout that we will populate later
# Compare altiar and plotly and bokeh for interactivity
# Look into using callback fiunctions for interactivity on non-plotly widgets
# look at all the page_config options. Start with sidebar open. Get Help button, etc
# there are options for getting and setting query params from the URL
# session_state is totally lost on hard manual refresh (cmd+R). Wonder about storing data with the cache to persist in each instance?
# consider duckdb for parsing and cleaning the initial dataframe? Maybe.
# st.container is like flexbox div. st.expander is a collapsible container. 


st.set_page_config(
    page_title="PMG Budget Optimizer", 
    layout="wide", 
    page_icon="static/favicon.ico",
    initial_sidebar_state="collapsed"
)
custom_css.inject_custom_styles()
# Inject global CSS


def main():
    uploaded_data = st.session_state.get('uploaded_data')
    init_app(uploaded_data)
    sidebar()
    header_section()
    strategy_section(uploaded_data)


def init_app(uploaded_data: dict | None):
    if uploaded_data is None:
        data_upload_section()
        # the above section has stop and rerun. So we won't get here until we have data.
    if 'chart_states' not in st.session_state:
        st.session_state['chart_states'] = {}
    for strategy_id, strategy_data in uploaded_data.items():
        chart_id = f"chart_{strategy_id}"
        st.session_state['chart_states'].setdefault(chart_id, {
            "strategy_id": strategy_id,
            "strategy_name": strategy_data['name'],
            "chart_id": chart_id,
            "starting_point_index": strategy_data['starting_point_index'],
            "selected_point_index": strategy_data['starting_point_index'],
            "x_fit": strategy_data['x_fit'],
            "y_fit": strategy_data['y_fit'],
            "z_fit": strategy_data['z_fit'],
            "input_x_points": strategy_data['input_x_points'],
            "input_y_points": strategy_data['input_y_points'],
            "input_z_points": strategy_data['input_z_points'],
        })


def data_upload_section():
    if st.session_state.get('uploaded_data') is not None:
        return

    with st.container(horizontal=False, horizontal_alignment="left", width=500):
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="file_uploader")
        
        use_dummy = st.button("Use Dummy Data", key="use_dummy")
        if use_dummy:
            file_path = "data/sample.csv"
        elif uploaded_file is not None:
            file_path = uploaded_file
        else:
            st.info("Please upload a CSV file to get started.")
            st.stop()
        
        uploaded_data = app_io.import_upload(file_path)
        st.session_state['uploaded_data'] = uploaded_data
        st.rerun()


def header_section():
    st.title("PMG Budget Optimizer")
    st.markdown("Select custom points on the curves, or allow the optimizer to choose for you.")


def strategy_section(uploaded_data: dict):
    for strategy_id, strategy_data in uploaded_data.items():
        with st.container(horizontal=True, width=1200, gap='small', vertical_alignment="center"):
            with st.container(horizontal=False,  width=800):
                st.subheader(strategy_data['name'])
                current_chart_state = st.session_state['chart_states'][f"chart_{strategy_id}"]
                chart_events = charts.build_startegy_chart(strategy_data, current_chart_state)
                handle_events(strategy_id, current_chart_state, chart_events)
                # Chart events will cause an instant rerun.
                current_index = st.session_state['chart_states'][f"chart_{strategy_id}"]['selected_point_index']
            with st.container(horizontal=False,  width=300):
                st.metric('Cost', f"${strategy_data['x_fit'][current_index]:,.0f}")
                st.metric('Conversions', f"{strategy_data['y_fit'][current_index]:,.2f}")
                st.metric('CPA', f"${strategy_data['z_fit'][current_index]:,.2f}")
        st.divider()


def handle_events(strategy_id: str, current_chart_state: dict, events: dict) -> None:
    if not events:
        return
    # with st.spinner("Refreshing chart..."):
    e = events[0]  # or handle multi-select
    idx = e.get("pointIndex")
    if current_chart_state is None or int(idx) != int(current_chart_state.get('selected_point_index')):
        st.session_state['chart_states'][f"chart_{strategy_id}"]['selected_point_index'] = idx
        st.rerun()


def sidebar(expanded: bool = True):
    with st.sidebar:
        custom_css.inject_logo(href=None, color="#0f172a", size_px=70)  # tweak color/size here
        st.sidebar.expander("Help", expanded=True).markdown("""
            ### How to use this app
            1. Upload a CSV file in the sidebar, or use the dummy data.
            2. Select points on the curves to set your desired cost and conversions.
            3. Review the optimized budget allocation in the Optimize section.
        """)




if __name__ == "__main__":
    main()
