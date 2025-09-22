import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

def build_startegy_chart(strategy_data: dict, current_chart_state: dict) -> list[dict]:
    strategy_id = strategy_data['id']
    starting_point_index = strategy_data['starting_point_index']
    if current_chart_state is None:
        selected_point_index = starting_point_index
    else:
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
        name=f"{strategy_id} Response Curve",
        showlegend=False,
        hovertemplate="Cost £%{x:.0f}<br>Conversions %{y:.2f}<br>CPA £%{meta:.2f}<extra></extra>",
    ))

    # Plot the actual data points
    fig.add_trace(go.Scatter(
        x=[starting_x_point],
        y=[starting_y_point],
        mode="markers",
        marker=dict(size=9, opacity=0.5),
        name="Response Curve",
        hoverinfo="skip"
    ))

    # Plot the original data points
    fig.add_trace(go.Scatter(
        x=list(strategy_data['input_x_points']),
        y=list(strategy_data['input_y_points']),
        mode="markers",
        marker=dict(size=9, opacity=0.2),
        name="Response Curve",
        hoverinfo="skip"
    ))

    # Highlight the selected point
    fig.add_trace(go.Scatter(
        x=[selected_x_point],
        y=[selected_y_point],
        meta=[selected_z_point],
        mode="markers",
        marker=dict(size=14, symbol="x"),
        name="selected",
        showlegend=False,
        hovertemplate="Cost £%{x:.0f}<br>Conversions %{y:.2f}<br>CPA £%{meta:.2f}<extra></extra>",
    ))
        
    fig.update_layout(template="plotly_white", clickmode="event+select",
                        hovermode="closest", height=300, margin=dict(l=10, r=10, t=30, b=10))

    events = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                        key=f"plt_{strategy_id}", override_height=300, override_width="100%")

    return events