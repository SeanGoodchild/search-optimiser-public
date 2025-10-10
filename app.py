
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
    """
    Displays bidding strategy charts and tables using container-based layout.
    Relies on custom_css.inject_table_styles() for consistent styling.
    """
    custom_css.inject_custom_styles()

    with st.container():
        st.markdown("## Bidding Strategies")
        st.caption("Select a point on each curve to test new spend levels and compare against baseline performance.")

    for strategy_id, strategy_data in uploaded_data.items():
        with st.container(border=False):
            st.subheader(strategy_data["name"])

            # Horizontal container for chart + table
            with st.container(horizontal=True, gap="large", vertical_alignment="center"):
                # ---- Left: Chart ----
                with st.container(horizontal=False):
                    current_chart_state = st.session_state["chart_states"][f"chart_{strategy_id}"]
                    chart_events = charts.build_startegy_chart(strategy_data, current_chart_state)
                    handle_events(strategy_id, current_chart_state, chart_events)
                    current_index = current_chart_state["selected_point_index"]

                # ---- Right: Table ----
                with st.container(horizontal=False):
                    start_idx = strategy_data["starting_point_index"]

                    # Incremental calculations
                    delta_cost = strategy_data["x_fit"][current_index] - strategy_data["x_fit"][start_idx]
                    delta_conv = strategy_data["y_fit"][current_index] - strategy_data["y_fit"][start_idx]
                    delta_cpa = (
                        (strategy_data["x_fit"][current_index] / strategy_data["y_fit"][current_index])
                        - (strategy_data["x_fit"][start_idx] / strategy_data["y_fit"][start_idx])
                        if strategy_data["y_fit"][current_index] and strategy_data["y_fit"][start_idx]
                        else 0
                    )

                    df = pd.DataFrame(
                        {
                            "Original Values": [
                                strategy_data["x_fit"][start_idx],
                                strategy_data["y_fit"][start_idx],
                                strategy_data["z_fit"][start_idx],
                            ],
                            "Selected Values": [
                                strategy_data["x_fit"][current_index],
                                strategy_data["y_fit"][current_index],
                                strategy_data["z_fit"][current_index],
                            ],
                            "Incremental ": [delta_cost, delta_conv, delta_cpa],
                        },
                        index=["Estimated Cost", "Estimated Conversions", "Estimated CPA"],
                    )

                    # Build HTML for styled incremental values
                    def format_increment(val, row_label):
                        """Format incremental values with correct sign and color semantics."""
                        # CPA logic is inverted (lower is better)
                        if row_label == "Estimated CPA":
                            css_class = "inc-positive" if val < 0 else "inc-negative" if val > 0 else "inc-neutral"
                            # Use "-" for improvement (decrease), "+" for decline (increase)
                            symbol = "-" if val < 0 else "+" if val > 0 else ""
                        else:
                            css_class = "inc-positive" if val > 0 else "inc-negative" if val < 0 else "inc-neutral"
                            symbol = "+" if val > 0 else "-" if val < 0 else ""
                        return f"<span class='{css_class}'>{symbol}{abs(val):,.0f}</span>"

                    # Convert to formatted HTML dataframe
                    html_df = df.copy()
                    html_df["Original Values"] = df["Original Values"].map("{:,.0f}".format)
                    html_df["Selected Values"] = df["Selected Values"].map("{:,.0f}".format)
                    html_df["Incremental "] = [
                        format_increment(v, idx) for v, idx in zip(df["Incremental "], df.index)
                    ]

                    # Render HTML table directly to preserve class styling
                    st.markdown(
                        html_df.to_html(escape=False, index=True, justify="center"),
                        unsafe_allow_html=True,
                    )

        st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
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
