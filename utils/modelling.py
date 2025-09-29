import numpy as np
from math import floor, log10, isfinite
import streamlit as st
import heapq
from typing import Dict, Tuple, Optional


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
    x_min = xs.min() * 0.75
    x_max = xs.max() * 1.1
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


def _extract_xy(entry: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accepts a variety of keys for convenience:
      x:  'x', 'x_fit', 'cost', 'a'      (you mentioned using 'a' for the x grid)
      y:  'y', 'y_fit', 'conversions'
    Returns x, y as ascending arrays; prepends (0,0) if not already there.
    """
    x = None
    for k in ("x", "x_fit", "cost", "a"):
        if k in entry:
            x = np.asarray(entry[k], dtype=float)
            break
    y = None
    for k in ("y", "y_fit", "conversions"):
        if k in entry:
            y = np.asarray(entry[k], dtype=float)
            break
    if x is None or y is None:
        raise KeyError("Entry must contain x-grid and y-grid (e.g., {'x_fit':..., 'y_fit':...}).")

    # sort by x just in case, and dedupe
    order = np.argsort(x)
    x, y = x[order], y[order]
    mask = np.concatenate(([True], np.diff(x) > 0))
    x, y = x[mask], y[mask]

    # ensure starts at (0,0) so the allocator can start from zero spend
    if x[0] > 0 or y[0] > 0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 0.0)

    return x, y


def optimize_budget(
    strategies: Dict[str, dict],
    target_cost: Optional[float] = None,
    target_conversions: Optional[float] = None,
) -> dict:
    """
    Greedy marginal-gain allocator on pre-sampled frontiers.
    Inputs:
      strategies: { strat_id: { 'x_fit'/'a'/..., 'y_fit'/... }, ... }
      target_cost: total budget £ to not exceed (maximize conversions)
      OR
      target_conversions: conversions target to achieve (minimize cost)

    Returns:
      {
        'indices': { strat_id: idx },    # chosen index on each grid
        'total_cost': float,
        'total_conversions': float
      }
    """
    if (target_cost is None) == (target_conversions is None):
        raise ValueError("Provide exactly one of target_cost or target_conversions.")

    # Prepare per-strategy grids and increments
    x_map, y_map, idx_map = {}, {}, {}
    deltas_map = {}  # list of (Δcost, Δconv) between i -> i+1
    for sid, entry in strategies.items():
        x, y = _extract_xy(entry)
        x_map[sid], y_map[sid] = x, y
        idx_map[sid] = 0
        dx = np.diff(x)
        dy = np.diff(y)
        # guard against zero-cost steps (shouldn't happen if you densified sensibly)
        valid = dx > 0
        deltas_map[sid] = list(zip(dx[valid], dy[valid]))

    total_cost = 0.0
    total_conv = 0.0

    # Max-heap of next increments by slope (Δconv / Δcost) for each strategy
    # Heap items: (-slope, sid, step_idx)
    heap = []
    for sid, deltas in deltas_map.items():
        if deltas:
            dcost, dconv = deltas[0]
            slope = dconv / dcost if dcost > 0 else 0.0
            heapq.heappush(heap, (-slope, sid, 0))

    def advance(sid: str, step_idx: int):
        """Advance strategy sid from step_idx -> step_idx+1, update heap with next step."""
        nonlocal total_cost, total_conv
        dcost, dconv = deltas_map[sid][step_idx]
        total_cost += dcost
        total_conv += dconv
        idx_map[sid] += 1
        next_idx = step_idx + 1
        if next_idx < len(deltas_map[sid]):
            ndcost, ndconv = deltas_map[sid][next_idx]
            slope = ndconv / ndcost if ndcost > 0 else 0.0
            heapq.heappush(heap, (-slope, sid, next_idx))

    if target_cost is not None:
        # Maximize conversions subject to budget
        remaining = float(target_cost)
        # If we started at non-zero x due to prepend, the first step already includes that cost
        # The heap contains the first increment for each strategy; pick until no increment fits.
        while heap:
            neg_slope, sid, step_idx = heapq.heappop(heap)
            dcost, dconv = deltas_map[sid][step_idx]
            if dcost <= remaining + 1e-12:
                advance(sid, step_idx)
                remaining -= dcost
            else:
                # Can't afford this step; try the next best step from other strategies
                continue
        # Done; totals already computed
    else:
        # Minimize cost to achieve target conversions
        target = float(target_conversions)
        while heap and total_conv + 1e-12 < target:
            neg_slope, sid, step_idx = heapq.heappop(heap)
            advance(sid, step_idx)
        # We may overshoot the conversions target a bit (snap to grid); that's intended.

    # Build the result using chosen indices snapped to your original grids
    result = {
        "indices": idx_map,
        "total_cost": float(total_cost),
        "total_conversions": float(total_conv),
    }
    return result
