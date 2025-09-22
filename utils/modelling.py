import numpy as np
from math import floor, log10, isfinite
import streamlit as st


@st.cache_data
def find_curve_fit(x_series: list, y_series: list, starting_point_index: int) -> dict:
    num_steps = 1000
    xs = np.array(x_series, dtype=float)
    ys = np.array(y_series, dtype=float)
    x0, y0 = xs[starting_point_index], ys[starting_point_index]
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
    z_fit = x_fit / y_fit
    
    starting_point_index = np.abs(x_fit - x0).argmin()

    return {
        'a_factor': a,
        'b_factor': b,
        'x_fit': x_fit,
        'y_fit': y_fit,
        'z_fit': z_fit,
        'starting_point_index': starting_point_index
    }


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
    """Choose a 1/2/5 style step so that (#steps) <= target_steps."""
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
    if lo is not None:
            snapped = max(lo, snapped)
    if hi is not None:
            snapped = min(hi, snapped)
    # tidy decimals like make_grid()
    decimals = max(0, -int(floor(log10(step))))
    return float(np.round(snapped, decimals))
