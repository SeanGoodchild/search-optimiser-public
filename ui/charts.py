
import streamlit as st
from services import store
import numpy as np
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events



def chart_with_events(strategy_data: dict) -> list[dict]:
    strategy_id = strategy_data['id']
    strategy_name = strategy_data['name']
    strategy_state = store.get_strategy(strategy_id)
    selected_point = strategy_state['selected_point']
    strategy_curve = strategy_state['curve_fit_data']
    fig = go.Figure()
    
    # Plot our calculated curve
    fig.add_trace(go.Scatter(
        x=list(strategy_curve['x_fit']), y=list(strategy_curve['y_fit']),
        mode="lines",
        name=f"{strategy_id} Response Curve",
        showlegend=False,
        hovertemplate="Cost £%{x:.0f}<br>Conversions %{y:.2f}<extra></extra>",
    ))

    # Plot the actual data points
    fig.add_trace(go.Scatter(
        x=strategy_data['cost_points'],
        y=strategy_data['conversion_points'],
        mode="markers",
        marker=dict(size=9),
        name="Response Curve",
        hoverinfo="skip"
    ))

    # Highlight the selected point
    fig.add_trace(go.Scatter(
        x=[selected_point[0]],
        y=[selected_point[1]],
        mode="markers",
        marker=dict(size=14, symbol="x"),
        name="selected",
        hovertemplate="Chosen<br>Cost £%{x:.0f}<br>Conversions %{y:.1f}<extra></extra>",
    ))
        
    fig.update_layout(template="plotly_white", clickmode="event+select",
                        hovermode="closest", height=300, margin=dict(l=10, r=10, t=30, b=10))

    events = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                        key=f"plt_{strategy_id}", override_height=300, override_width="100%")

    return events



def pick_step(xmin, xmax, target_steps=100):
    # simple “1/2/5/10/20/50/100…” ladder
    ladder = [1,2,5,10,20,50,100,200,500,1000,2000,5000,10000]
    span = max(1e-9, xmax - xmin)
    raw = span / target_steps
    k = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1
    cand = [k*v for v in ladder]
    return min(cand, key=lambda c: abs(c - raw))

def interp_y(x, xs, ys):
    # xs must be sorted by x
    order = np.argsort(xs)
    xs_s = np.asarray(xs)[order]
    ys_s = np.asarray(ys)[order]
    return float(np.interp(x, xs_s, ys_s))