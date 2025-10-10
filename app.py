
import streamlit as st
from utils import app_io, modelling
from ui import charts, custom_css
import pandas as pd
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
    initial_sidebar_state="expanded"
)
custom_css.inject_custom_styles()
custom_css.remove_top_padding()


def main():
    uploaded_data = st.session_state.get('uploaded_data')
    init_app(uploaded_data)
    sidebar()
    with st.container(horizontal=False, width='stretch', horizontal_alignment="left"):
        header_section()
    strategy_section(uploaded_data)


def init_app(uploaded_data: dict | None):
    if uploaded_data is None:
        if st.session_state.get('uploaded_data') is None:
            uploaded_data = data_upload_section()
            st.session_state['uploaded_data'] = uploaded_data
            st.rerun()
    
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
        return uploaded_data
        


def sum_of_metric(chart_states: dict, metric: str) -> float:
    total = 0.0
    total_cost = 0
    total_conversions = 0
    for state in chart_states.values():
        strategy_id = state['strategy_id']
        selected_index = state['selected_point_index']
        if metric == 'cost':
            total += st.session_state['uploaded_data'][strategy_id]['x_fit'][selected_index]
        elif metric == 'conversions':
            total += st.session_state['uploaded_data'][strategy_id]['y_fit'][selected_index]
        elif metric == 'cpa':
            total_cost += st.session_state['uploaded_data'][strategy_id]['x_fit'][selected_index]
            total_conversions += st.session_state['uploaded_data'][strategy_id]['y_fit'][selected_index]
    if metric == 'cpa' and total_conversions > 0:
        return total_cost / total_conversions
    
    return total


# In app.py, modify header_section():
def header_section():
    st.title("Google Ads Budget Optimizer")

    #Current state summary
    with st.container(border=True):
        col1, col2, col3= st.columns(3)
        with col1:
            st.metric("Total Budget", f"${sum_of_metric(st.session_state['chart_states'], 'cost'):,.0f}")
        with col2:
            st.metric("Total Conversions", f"{sum_of_metric(st.session_state['chart_states'], 'conversions'):,.0f}")
        with col3:
            st.metric("Weighted CPA", f"${sum_of_metric(st.session_state['chart_states'], 'cpa'):,.2f}")

def strategy_section(uploaded_data: dict):
    st.markdown("## Bidding Strategies")
    st.caption("Select a point on each curve to test new spend levels and compare against baseline.")
    for strategy_id, strategy_data in uploaded_data.items():
        with st.container(horizontal=True, width=700, gap='small', vertical_alignment="distribute"):
            st.subheader(strategy_data['name'])
        with st.container(horizontal=True, width=1300, gap='small', vertical_alignment="center"):
            with st.container(horizontal=False,  width=700, horizontal_alignment="distribute"):
                current_chart_state = st.session_state['chart_states'][f"chart_{strategy_id}"]
                chart_events = charts.build_startegy_chart(strategy_data, current_chart_state)
                handle_events(strategy_id, current_chart_state, chart_events)
                # Chart events will cause an instant rerun.
                current_index = st.session_state['chart_states'][f"chart_{strategy_id}"]['selected_point_index']
            with st.container(horizontal=False,  width=500, horizontal_alignment="right"):
                starting_point_index = strategy_data['starting_point_index']
                strategy_table_df = pd.DataFrame(
                    {
                        "Original Values": [strategy_data['x_fit'][starting_point_index], strategy_data['y_fit'][starting_point_index], strategy_data['z_fit'][starting_point_index]],
                        "Selected Values": [strategy_data['x_fit'][current_index], strategy_data['y_fit'][current_index], strategy_data['z_fit'][current_index]],
                        "Incremental Performance": [strategy_data['x_fit'][current_index]-strategy_data['x_fit'][starting_point_index], strategy_data['y_fit'][current_index]-strategy_data['y_fit'][starting_point_index],
                            (strategy_data['x_fit'][current_index]-strategy_data['x_fit'][starting_point_index]) / (strategy_data['y_fit'][current_index]-strategy_data['y_fit'][starting_point_index]) if (strategy_data['y_fit'][current_index]-strategy_data['y_fit'][starting_point_index]) != 0 else 0],
                    },
                    index=["Estimated Cost", "Estimated Conversions", "Estimated CPA"],
                )
                strategy_table_df["Original Values"] = strategy_table_df["Original Values"].map("{:,.0f}".format)
                strategy_table_df["Selected Values"] = strategy_table_df["Selected Values"].map("{:,.0f}".format)
                strategy_table_df["Incremental Performance"] = strategy_table_df["Incremental Performance"].map("{:,.0f}".format)
                st.table(strategy_table_df)
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


def sidebar():
    with st.sidebar:
        uploaded_data = st.session_state.get('uploaded_data')
        custom_css.inject_logo(href=None, color="#0f172a", size_px=70)  # tweak color/size here
        total_cost = sum_of_metric(st.session_state['chart_states'], 'cost')
        total_conversions = sum_of_metric(st.session_state['chart_states'], 'conversions')
        st.header("Total Weekly Estimates")
        target_cost = st.number_input(label='Cost $', value=total_cost, format="%0.2f")
        target_convs = st.number_input(label='Conversions', value=total_conversions)
        target_cpa = st.number_input(label='CPA', value=total_cost/total_conversions if total_conversions > 0 else 0)
        st.markdown("**Budget Allocation**")
        optimise_picked = st.button(
            "Optimize Budget",
            key="optimize_button",
            type="primary",
            use_container_width=True
        )
        st.caption("Redistributes your current budget across channels to minimize weighted CPA")
        if optimise_picked:
            result = modelling.optimize_budget(uploaded_data, target_cost=target_cost, target_conversions=None)
            # This will update the uploaded_data in place
            # We then need to update the chart states to reflect the new picks
            for strategy_id, strategy_data in uploaded_data.items():
                st.session_state['chart_states'][f"chart_{strategy_id}"]['selected_point_index'] = result['indices'][strategy_id]-1
            st.rerun()  # Rerun to refresh all charts with new selections
        st.divider()
        with st.expander("Help & Tips", expanded=False):
            st.markdown("""
            **1. Review Current State**  
            The baseline metrics show your current weekly allocation.

            **2. Adjust Individual Strategies**  
            Click any point on a curve to simulate different spend levels.  
            Watch the metrics update in real-time.

            **3. Optimize Automatically**  
            Click "Optimize Budget" to let the model find the best  
            distribution across all strategies to minimize CPA.
            """)

if __name__ == "__main__":
    main()
