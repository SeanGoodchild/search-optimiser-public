
import streamlit as st
import numpy as np
import csv
import json
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from math import floor, log10, isfinite


st.set_page_config(page_title="PMG Budget Optimizer", layout="wide")
st.title("PMG Budget Optimizer")
st.markdown("Select custom points on the curves, or allow the optimizer to choose for you.")


def main():
    all_strategies = st.session_state.get('strategies')
    if all_strategies is None:
        all_strategies = import_data('data/sample.csv')
        st.session_state['strategies'] = all_strategies

    for strategy_id, strategy in all_strategies.items():
        st.subheader(strategy['name'])
        chart_events = chart_with_events(strategy)
        handle_events(strategy, chart_events)
        st.metric('Cost', strategy['selected_point'][0], delta=None, delta_color="normal")
        st.metric('Conversions', strategy['selected_point'][1], delta=None, delta_color="normal")
        st.divider()
    
    build_optimization_section(all_strategies)


def handle_events(strategy: dict, events: dict) -> None:
    if not events:
        return
    clicked_x = float(events[0]['x'])
    clicked_y = float(events[0]['y'])

    (current_x, current_y) = strategy['selected_point']
    if clicked_x != current_x or clicked_y != current_y:
        strategy.update({'selected_point': (clicked_x, clicked_y)})
        st.session_state['strategies'][strategy['id']] = strategy
        st.toast(f"Selected cost £{clicked_x:.2f} with {clicked_y:.2f}")
        st.rerun()


def build_optimization_section(strategy_data: dict):
    st.header("Optimize")


def import_data(csv_path: str) -> dict:
    output = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strategy_id = row["bidding_strategy_id"]
            entry = {
                "id": strategy_id,
                "name": row["bidding_strategy_name"],
                "cost_points": [],
                "conversion_points": [],
                "target_cpa_points": [],
                "chart_events": [],
                "selected_point": None,
                "curve_fit": None,
            }

            for col, val in row.items():
                if col.endswith("_points"):
                    points = json.loads(val)
                    for p in points:
                        cost = float(p["costMicros"]) / 1_000_000
                        convs = float(p.get("biddableConversions", 0))
                        target_cpa = float(p.get("targetCpaMicros", 0)) / 1_000_000
                        entry["cost_points"].append(cost)
                        entry["conversion_points"].append(convs)
                        entry["target_cpa_points"].append(target_cpa)

                        if round(target_cpa) == round(float(row['target_cpa_target_cpa'])):
                            entry["anchor_point"] = (cost, convs)
                            entry["selected_point"] = (cost, convs)

            entry["curve_fit_data"] = find_curve_fit(entry["cost_points"], entry["conversion_points"], entry["anchor_point"])            
            output[strategy_id] = entry

    return output


def find_curve_fit(x_series: list, y_series: list, anchor_point: tuple[float, float]) -> dict:
    num_steps = 1000
    xs = np.array(x_series, dtype=float)
    ys = np.array(y_series, dtype=float)
    x0, y0 = anchor_point
    eps = 1e-9

    b, ln_a_hat = np.polyfit(np.log(xs + eps), np.log(ys + eps), 1)
    b = float(b)

    # --- hard level anchor (pass through current setting) ---
    a = float(y0 / ((x0 + eps) ** b))

    # grid incl. anchor span
    x_min = xs.min() * 0.33
    x_max = xs.max() * 1.25
    x_fit = make_grid(x_min, x_max, target_steps=num_steps)
    y_fit = a * np.power(np.maximum(x_fit, 0.0), b)

    return {"a": a, "b": b, "x_fit": x_fit, "y_fit": y_fit}


def make_grid(xmin: float, xmax: float, target_steps: int = 1000) -> np.ndarray:
    """Grid snapped to sensible increment that fully covers [xmin, xmax]."""
    step = sensible_increment(xmin, xmax, target_steps)
    start = np.floor(xmin / step) * step
    stop  = np.ceil (xmax / step) * step
    # + step/2 to ensure inclusion of the last point despite FP error
    grid = np.arange(start, stop + step/2, step, dtype=float)
    # Round to a nice number of decimals for clean tooltips/printing
    decimals = max(0, -int(floor(log10(step))) )  # e.g. 0.1 -> 1 dp, 0.01 -> 2 dp
    grid = np.round(grid, decimals)
    return grid


def sensible_increment(xmin: float, xmax: float, target_steps: int = 1000) -> float:
    """Choose a 1–2–5 style step so that (#steps) <= target_steps."""
    if not (isfinite(xmin) and isfinite(xmax)) or xmax <= xmin:
        raise ValueError("Bad range for sensible_increment")
    span = xmax - xmin
    raw = span / max(target_steps, 1)

    # 1–2–5 ladder around raw
    base = 10.0 ** floor(log10(raw))  # decade
    ladder = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10]) * base
    step = ladder[ladder >= raw].min(initial=ladder[-1])  # smallest >= raw
    return float(step)


def snap(value: float, step: float, lo: float | None = None, hi: float | None = None) -> float:
    """Snap a value to nearest multiple of step; optional clamp to [lo, hi]."""
    snapped = round(value / step) * step
    if lo is not None: snapped = max(lo, snapped)
    if hi is not None: snapped = min(hi, snapped)
    # tidy decimals like make_grid()
    decimals = max(0, -int(floor(log10(step))))
    return float(np.round(snapped, decimals))


def chart_with_events(strategy_data: dict) -> list[dict]:
    strategy_id = strategy_data['id']
    selected_point = strategy_data['selected_point']
    strategy_curve = strategy_data['curve_fit_data']
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
        x=[strategy_data['anchor_point'][0]],
        y=[strategy_data['anchor_point'][1]],
        mode="markers",
        marker=dict(size=9, opacity=0.2),
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


if __name__ == "__main__":
    main()
