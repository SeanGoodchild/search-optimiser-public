
import numpy as np
import pandas as pd

LADDER = [1,2,5,10,20,50,100,200,500,1000,2000,5000,10000]

def sensible_increment(xmin: float, xmax: float, target_steps: int = 40) -> float:
    span = max(1e-9, xmax - xmin)
    raw = span / target_steps
    k = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1
    candidates = [k * v for v in LADDER] + [0.5 * k * v for v in [1,2,5,10,20,50,100]]
    best = min(candidates, key=lambda c: abs(c - raw))
    return max(1e-9, best)

def piecewise_linear_interp(df: pd.DataFrame, xs: np.ndarray) -> pd.DataFrame:
    df = df.sort_values('cost')
    x = df['cost'].to_numpy()
    y = df['conv'].to_numpy()
    ys = np.interp(xs, x, y)
    return pd.DataFrame({'cost': xs, 'conv': ys})

def densify_to_increment(df: pd.DataFrame, inc: float) -> pd.DataFrame:
    xmin, xmax = float(min(df['cost'])), float(max(df['cost']))
    xs = np.arange(xmin, xmax + 0.5*inc, inc)
    return piecewise_linear_interp(df, xs)
