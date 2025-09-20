import numpy as np
from app import import_data

def main():
    data = import_data('data/sample.csv')
    for strategy_id, strategy_data in data.items():
        result = find_best_curve(strategy_data['cost_points'], strategy_data['conversion_points'])
        print(f"Strategy {strategy_id} best fit metrics:")
        print(result)  # best model


def find_best_curve(x_series: list, y_series: list) -> list:
    """
    Fit several parametric, concave/saturating shapes to (cost -> conversions)
    and return a ranked 2D list [[model, r2], ...] (descending by R^2).

    Models (all extrapolate sensibly; no linear/spline cheats):
      - power:        y = a * x^b
      - exp_sat:      y = L * (1 - exp(-k * x))
      - logistic:     y = L / (1 + exp(-k * (x - x0)))
      - gompertz:     y = L * exp( -exp( -k * (x - x0) ) )
      - hill:         y = L * x^h / (K^h + x^h)

    Returns:
      [["model_name", r2_float], ...] sorted by r2 desc
    """
    xs = np.asarray(x_series, dtype=float)
    ys = np.asarray(y_series, dtype=float)

    # --- basic sanitization ---
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if xs.size < 3:
        raise ValueError("Need at least 3 points")
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    xs = np.maximum(xs, 0.0)
    ys = np.maximum(ys, 0.0)
    eps = 1e-9

    # helpers
    def r2(y_true, y_pred):
        y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
        rss = float(np.sum((y_true - y_pred) ** 2))
        tss = float(np.sum((y_true - y_true.mean()) ** 2))
        return float(1.0 - rss / (tss + eps))

    results = []

    # ---------- 1) Power: y = a * x^b  (fit in log-space; only x>0,y>0) ----------
    pos = (xs > 0) & (ys > 0)
    if np.any(pos):
        Xl = np.log(xs[pos] + eps)
        Yl = np.log(ys[pos] + eps)
        A = np.vstack([np.ones_like(Xl), Xl]).T
        (ln_a, b), *_ = np.linalg.lstsq(A, Yl, rcond=None)
        a = float(np.exp(ln_a)); b = float(b)
        def power_fun(x): return np.where(x > 0, a * (x ** b), 0.0)
        r2_power = r2(ys, power_fun(xs))
        results.append(["power", r2_power])

    # ---------- 2) Exponential saturation: y = L * (1 - exp(-k x)) ----------
    def exp_sat_fun(x, L, k): return L * (1.0 - np.exp(-k * np.maximum(x, 0.0)))
    L0 = max(ys.max(), eps)
    xmax = xs.max() + eps
    k_grid = np.logspace(np.log10(1/(100*xmax)), np.log10(10/xmax), 40)
    L_grid = np.linspace(0.6 * L0, 3.0 * L0, 30)
    best_rss = np.inf; best_L = None; best_k = None
    for Lg in L_grid:
        for kg in k_grid:
            pred = exp_sat_fun(xs, Lg, kg)
            rss = float(np.sum((ys - pred) ** 2))
            if rss < best_rss:
                best_rss, best_L, best_k = rss, float(Lg), float(kg)
    if best_L is not None:
        r2_exp = r2(ys, exp_sat_fun(xs, best_L, best_k))
        results.append(["exp_sat", r2_exp])

    # ---------- 3) Logistic: y = L / (1 + exp(-k (x - x0))) ----------
    def logistic_fun(x, L, k, x0): return L / (1.0 + np.exp(-k * (x - x0)))
    rng = np.ptp(xs)
    k_grid = np.logspace(np.log10(1/(100*xmax)), np.log10(10/xmax), 25)
    L_grid = np.linspace(0.8 * L0, 3.0 * L0, 18)
    x0_grid = np.linspace(xs.min() - 0.25 * rng, xs.max() + 0.25 * rng, 18)
    best_rss = np.inf; best_L = best_k = best_x0 = None
    for Lg in L_grid:
        for kg in k_grid:
            for x0g in x0_grid:
                pred = logistic_fun(xs, Lg, kg, x0g)
                rss = float(np.sum((ys - pred) ** 2))
                if rss < best_rss:
                    best_rss, best_L, best_k, best_x0 = rss, float(Lg), float(kg), float(x0g)
    if best_L is not None:
        r2_log = r2(ys, logistic_fun(xs, best_L, best_k, best_x0))
        results.append(["logistic", r2_log])

    # ---------- 4) Gompertz: y = L * exp( -exp( -k (x - x0) ) ) ----------
    def gompertz_fun(x, L, k, x0): return L * np.exp(-np.exp(-k * (x - x0)))
    # reuse grids with slightly different L range (often < plateau)
    L_grid = np.linspace(0.8 * L0, 3.0 * L0, 18)
    best_rss = np.inf; best_L = best_k = best_x0 = None
    for Lg in L_grid:
        for kg in k_grid:
            for x0g in x0_grid:
                pred = gompertz_fun(xs, Lg, kg, x0g)
                rss = float(np.sum((ys - pred) ** 2))
                if rss < best_rss:
                    best_rss, best_L, best_k, best_x0 = rss, float(Lg), float(kg), float(x0g)
    if best_L is not None:
        r2_gomp = r2(ys, gompertz_fun(xs, best_L, best_k, best_x0))
        results.append(["gompertz", r2_gomp])

    # ---------- 5) Hill (Michaelis–Menten generalization): y = L * x^h / (K^h + x^h) ----------
    def hill_fun(x, L, K, h):
        x = np.maximum(x, 0.0)
        return L * (x**h) / (K**h + x**h + eps)
    L_grid = np.linspace(0.8 * L0, 3.0 * L0, 18)
    # K around spend scale; use logspace over [min..max] with padding
    K_min = max(xs.min() + eps, xmax / 200.0)
    K_max = max(xmax, K_min * 10.0)
    K_grid = np.logspace(np.log10(K_min), np.log10(K_max), 20)
    h_grid = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    best_rss = np.inf; best_L = best_K = best_h = None
    for Lg in L_grid:
        for Kg in K_grid:
            for hg in h_grid:
                pred = hill_fun(xs, Lg, Kg, hg)
                rss = float(np.sum((ys - pred) ** 2))
                if rss < best_rss:
                    best_rss, best_L, best_K, best_h = rss, float(Lg), float(Kg), float(hg)
    if best_L is not None:
        r2_hill = r2(ys, hill_fun(xs, best_L, best_K, best_h))
        results.append(["hill", r2_hill])

    # sort by R^2 desc and return
    results.sort(key=lambda t: t[1], reverse=True)
    return results

if __name__ == "__main__":
    main()
