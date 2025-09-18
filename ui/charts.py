
import plotly.graph_objects as go
import pandas as pd

def curve_figure(entity_id: str, df: pd.DataFrame, selected_cost: float | None = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['cost'], y=df['conv'], mode="lines+markers",
        name=f"{entity_id} frontier",
        hovertemplate="cost £%{x:.0f}<br>Conv %{y:.2f}<extra></extra>"
    ))
    if selected_cost is not None:
        nearest = df.iloc[(df['cost'] - selected_cost).abs().argsort().iloc[0]]
        fig.add_trace(go.Scatter(
            x=[nearest['cost']], y=[nearest['conv']],
            mode="markers", marker=dict(size=12, symbol="x"),
            name="selected"
        ))
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=320)
    fig.update_xaxes(title="cost")
    fig.update_yaxes(title="Conversions")
    return fig
