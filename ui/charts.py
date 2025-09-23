import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

custom_colors = [
    "#5a6076", "#0c69ea", "#0c69ea", "#0c69ea", "#0c69ea",
    "#00FFFF", "#0000FF", "#CC00FF", "#FFFF00", "#800000"
]


def build_startegy_chart(strategy_data: dict, current_chart_state: dict) -> list[dict]:
    strategy_id = strategy_data['id']
    strategy_name = strategy_data['name']
    starting_point_index = strategy_data['starting_point_index']
    selected_point_index = current_chart_state['selected_point_index']
    
    fig = go.Figure()

    x_series = list(strategy_data['x_fit'])
    y_series = list(strategy_data['y_fit'])
    z_series = list(strategy_data['z_fit'])

    starting_x_point = strategy_data['x_fit'][starting_point_index]
    starting_y_point = strategy_data['y_fit'][starting_point_index]

    selected_x_point = strategy_data['x_fit'][selected_point_index]
    selected_y_point = strategy_data['y_fit'][selected_point_index]
    selected_z_point = strategy_data['z_fit'][selected_point_index]
    
    # Plot our calculated curve
    fig.add_trace(go.Scatter(
        x=x_series, 
        y=y_series,
        meta=z_series,
        mode="lines",
        name=f"{strategy_name} Response Curve",
        showlegend=False,
        hovertemplate="Cost £%{x:.0f}<br>Conversions %{y:.2f}<br>CPA £%{meta:.2f}<extra></extra>",
    ))

    # Plot the actual starting data point
    fig.add_trace(go.Scatter(
        x=[starting_x_point],
        y=[starting_y_point],
        mode="markers",
        marker=dict(size=9, opacity=0.5),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Plot the original data points
    fig.add_trace(go.Scatter(
        x=list(strategy_data['input_x_points']),
        y=list(strategy_data['input_y_points']),
        mode="markers",
        marker=dict(size=9, opacity=0.2),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Highlight the selected point
    fig.add_trace(go.Scatter(
        x=[selected_x_point],
        y=[selected_y_point],
        meta=[selected_z_point],
        mode="markers",
        marker=dict(size=14, symbol="x"),
        name="Selected",
        showlegend=False,
        hovertemplate="Cost £%{x:.0f}<br>Conversions %{y:.2f}<br>CPA £%{meta:.2f}<extra></extra>",
    ))

    # Tidy axes: gridlines, ticks, ranges, formatting
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.25)",
        zeroline=False, showline=True, linewidth=1, linecolor="rgba(128,128,128,0.5)",
        tickformat=",.0f",  # nice thousands
        tickprefix="£",     # if “Cost” on X
        rangemode="tozero"
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.25)",
        zeroline=False, showline=True, linewidth=1, linecolor="rgba(128,128,128,0.5)",
        # tickformat=",.2f",  # conversions with 2dp
        rangemode="tozero"
    )
        
    fig.update_layout(
        colorway=custom_colors,
        clickmode="event+select",
        hovermode="closest",
        height=300,
        margin=dict(l=40, r=10, t=10, b=30),
    )

    events = plotly_events(
        fig, click_event=True, hover_event=False, select_event=False, 
        key=f"plt_{strategy_id}",
        override_height=300, override_width="100%"
    )

    return events
