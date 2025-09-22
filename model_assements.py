# fit_curves.py
# Diminishing-returns model selection with anchor-constrained fitting.
# - Preserves original return shape: [["model_name", r2_float], ...] sorted by R^2 desc
# - Adds anchor pass-through: each model is forced to fit f(x_anchor) = y_anchor before scoring
# - Includes additional models: weibull_cdf, richards, log_saturation
# - No external deps beyond NumPy

from __future__ import annotations
import numpy as np
from utils import app_io


def main() -> None:
    data = app_io.import_upload("data/sample.csv")
    for strategy_id, strategy_data in data.items():
        results = find_best_curve(
            strategy_data["input_x_points"],
            strategy_data["input_y_points"],
            strategy_data["raw_starting_point_index"],  # anchor index (in original arrays)
            # metric="r2",              # alternatives: "adj_r2", "aic", "bic" (kept default "r2")
            # return_params=True,       # set True to also get best-fit params per model
        )
        print(f"Strategy {strategy_id} best fit metrics:")
        print(results)  # [["model_name", r2], ...]


def find_best_curve(
    x_series: list[float],
    y_series: list[float],
    starting_index: int,
    *,
    metric: str = "r2",
    return_params: bool = False,
):
    """
    Fit several concave/saturating shapes to (cost -> KPI) with diminishing returns,
    each constrained to pass through an anchor point provided by `starting_index`,
    then return a ranking by the chosen `metric`.

    Models (all extrapolate sensibly; no linear/spline cheats):
      - power:        y = a * x^b
      - exp_sat:      y = L * (1 - exp(-k * x))
      - logistic:     y = L / (1 + exp(-k * (x - x0)))
      - gompertz:     y = L * exp( -exp( -k * (x - x0) ) )
      - hill:         y = L * x^h / (K^h + x^h)
      - weibull_cdf:  y = L * (1 - exp( -(x/λ)^k ))
      - richards:     y = L * [1 + ν * exp(-k (x - x0))]^(-1/ν)
      - log_saturation: y = a * ln(1 + b x)  (unbounded but diminishing)

    Args:
        x_series, y_series: sequences of equal length (assumed valid per your note)
        starting_index: index in the ORIGINAL input arrays for the anchor point
        metric: "r2" (default), "adj_r2", "aic", or "bic"
        return_params: if True, returns (ranked_list, best_params_dict)

    Returns:
        [["model_name", score_float], ...] sorted by score desc
        If return_params=True: (ranked_list, best_params_dict)
    """
    # --- capture anchor from original arrays BEFORE sanitization/sorting ---
    xA = float(x_series[starting_index])
    yA = float(y_series[starting_index])

    # --- basic sanitization on full series ---
    xs = np.asarray(x_series, dtype=float)
    ys = np.asarray(y_series, dtype=float)

    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if xs.size < 3:
        raise ValueError("Need at least 3 points")

    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    # Clamp to non-negative domain (typical for spend/KPI)
    xs = np.maximum(xs, 0.0)
    ys = np.maximum(ys, 0.0)

    # Align anchor with same non-negativity policy
    xA = max(xA, 0.0)
    yA = max(yA, 0.0)

    n = xs.size
    eps = 1e-12

    # --- helpers: metrics ---
    def r2(y_true, y_pred) -> float:
        y_true = np.asarray(y_true, float)
        y_pred = np.asarray(y_pred, float)
        rss = float(np.sum((y_true - y_pred) ** 2))
        tss = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return float(1.0 - rss / (tss + eps))

    def adj_r2(y_true, y_pred, k_params: int) -> float:
        """Adjusted R^2 with p = number of free parameters actually searched (post-anchor)."""
        y_true = np.asarray(y_true, float)
        y_pred = np.asarray(y_pred, float)
        m = y_true.size
        rss = float(np.sum((y_true - y_pred) ** 2))
        tss = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2_val = 1.0 - rss / (tss + eps)
        p = max(k_params, 1)
        # Guard when m <= p+1; fall back to plain R^2
        if m <= p + 1:
            return float(r2_val)
        return float(1.0 - (1.0 - r2_val) * (m - 1) / (m - p - 1 + eps))

    def aic(y_true, y_pred, k_params: int) -> float:
        """Akaike Information Criterion (lower is better). We invert later to keep 'higher=better' API."""
        y_true = np.asarray(y_true, float)
        y_pred = np.asarray(y_pred, float)
        m = y_true.size
        rss = float(np.sum((y_true - y_pred) ** 2))
        # Gaussian residuals: -2 log L ≈ m*log(rss/m) + const
        return m * np.log((rss + eps) / m) + 2 * k_params

    def bic(y_true, y_pred, k_params: int) -> float:
        """Bayesian Information Criterion (lower is better)."""
        y_true = np.asarray(y_true, float)
        y_pred = np.asarray(y_pred, float)
        m = y_true.size
        rss = float(np.sum((y_true - y_pred) ** 2))
        return m * np.log((rss + eps) / m) + k_params * np.log(m + eps)

    def score(y_true, y_pred, k_params: int) -> float:
        if metric == "r2":
            return r2(y_true, y_pred)
        elif metric == "adj_r2":
            return adj_r2(y_true, y_pred, k_params)
        elif metric == "aic":
            # invert so "higher is better" across all metrics
            return -aic(y_true, y_pred, k_params)
        elif metric == "bic":
            return -bic(y_true, y_pred, k_params)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # --- helpers: grid search ---
    def best_on_grid(pred_fn, grid_kwargs: dict[str, np.ndarray]):
        """
        pred_fn: callable(**params) -> y_pred (vector over xs)
        grid_kwargs: {"name": iterable}
        Returns: (best_params_dict, best_pred_vector, best_rss)
        """
        names = list(grid_kwargs.keys())
        arrays = [np.asarray(v) for v in grid_kwargs.values()]

        best_params = None
        best_pred = None
        best_rss = np.inf

        def _recurse(i: int, current: dict):
            nonlocal best_params, best_pred, best_rss
            if i == len(names):
                try:
                    yhat = pred_fn(**current)
                except FloatingPointError:
                    return
                if yhat is None:
                    return
                if not np.all(np.isfinite(yhat)):
                    return
                rss = float(np.sum((ys - yhat) ** 2))
                if rss < best_rss:
                    best_rss = rss
                    best_params = current.copy()
                    best_pred = yhat
                return
            name = names[i]
            for val in arrays[i]:
                current[name] = float(val)
                _recurse(i + 1, current)

        _recurse(0, {})
        return (best_params or {}), best_pred, best_rss

    # --- scales / grids ---
    xmax = float(xs.max() + eps)
    rng = float(np.ptp(xs) + eps)
    L0 = float(max(ys.max(), eps))

    k_grid_common = np.logspace(np.log10(1 / (100 * xmax)), np.log10(10 / xmax), 25)
    x0_grid = np.linspace(xs.min() - 0.25 * rng, xs.max() + 0.25 * rng, 18)
    h_grid = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])

    # Weibull λ scale around spend
    lam_min = max(xs.min() + eps, xmax / 200.0)
    lam_max = max(xmax, lam_min * 10.0)
    lambda_grid = np.logspace(np.log10(lam_min), np.log10(lam_max), 25)
    weib_k_grid = np.linspace(0.3, 3.5, 25)

    # Hill K scale
    K_min = lam_min
    K_max = max(xmax, K_min * 10.0)
    K_grid = np.logspace(np.log10(K_min), np.log10(K_max), 20)

    # Log-saturation b scale (adapt if your xs are tiny/huge)
    b_grid = np.logspace(-6, 1, 40)

    # --- model fits ---
    results: list[list[object]] = []
    best_params_map: dict[str, dict[str, float]] = {}

    # 1) Power (anchor: a = yA / xA^b); requires xA > 0
    if xA > 0 and yA > 0:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            b_grid_power = np.linspace(0.1, 3.5, 60)

            def power_pred(b: float):
                a = yA / ((xA + eps) ** b)
                x_pos = np.maximum(xs, 0.0)
                return np.where(x_pos > 0, a * (x_pos ** b), 0.0)

            bp, yhat, _ = best_on_grid(lambda b: power_pred(b), {"b": b_grid_power})
            if yhat is not None:
                results.append(["power", score(ys, yhat, k_params=1)])  # only b searched
                best_params_map["power"] = {"a": float(yA / ((xA + eps) ** bp["b"])), **bp}

    # 2) Exponential saturation (anchor L = yA / (1 - exp(-k xA)))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        def exp_sat_pred(k: float):
            denom = 1.0 - np.exp(-k * xA)
            if denom <= eps:
                return None
            L = yA / denom
            return L * (1.0 - np.exp(-k * np.maximum(xs, 0.0)))

        kp, yhat, _ = best_on_grid(lambda k: exp_sat_pred(k), {"k": k_grid_common})
        if yhat is not None:
            results.append(["exp_sat", score(ys, yhat, k_params=1)])  # only k searched
            L_star = float(yA / (1.0 - np.exp(-kp["k"] * xA)))
            best_params_map["exp_sat"] = {"L": L_star, **kp}

    # 3) Logistic (anchor L = yA * (1 + exp(-k (xA - x0))))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        def logistic_pred(k: float, x0c: float):
            L = yA * (1.0 + np.exp(-k * (xA - x0c)))
            return L / (1.0 + np.exp(-k * (xs - x0c)))

        lp, yhat, _ = best_on_grid(lambda k, x0c: logistic_pred(k, x0c),
                                   {"k": k_grid_common, "x0c": x0_grid})
        if yhat is not None:
            results.append(["logistic", score(ys, yhat, k_params=2)])  # k, x0 searched
            L_star = float(yA * (1.0 + np.exp(-lp["k"] * (xA - lp["x0c"]))))
            best_params_map["logistic"] = {"L": L_star, "k": lp["k"], "x0": lp["x0c"]}

    # 4) Gompertz (anchor L = yA / exp(-exp(-k (xA - x0))))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        def gompertz_pred(k: float, x0c: float):
            L = yA / np.exp(-np.exp(-k * (xA - x0c)))
            return L * np.exp(-np.exp(-k * (xs - x0c)))

        gp, yhat, _ = best_on_grid(lambda k, x0c: gompertz_pred(k, x0c),
                                   {"k": k_grid_common, "x0c": x0_grid})
        if yhat is not None:
            results.append(["gompertz", score(ys, yhat, k_params=2)])  # k, x0 searched
            L_star = float(yA / np.exp(-np.exp(-gp["k"] * (xA - gp["x0c"]))))
            best_params_map["gompertz"] = {"L": L_star, "k": gp["k"], "x0": gp["x0c"]}

    # 5) Hill (anchor L = yA * (K^h + xA^h) / xA^h); requires xA > 0
    if xA > 0:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            def hill_pred(K: float, h: float):
                xh = np.power(np.maximum(xs, 0.0), h)
                xAh = xA ** h
                if xAh <= eps:
                    return None
                L = yA * ((K ** h + xAh) / xAh)
                return L * (xh / (K ** h + xh + eps))

            hp, yhat, _ = best_on_grid(lambda K, h: hill_pred(K, h),
                                       {"K": K_grid, "h": h_grid})
            if yhat is not None:
                results.append(["hill", score(ys, yhat, k_params=2)])  # K, h searched
                xAh = xA ** hp["h"]
                L_star = float(yA * ((hp["K"] ** hp["h"] + xAh) / xAh))
                best_params_map["hill"] = {"L": L_star, **hp}

    # 6) Weibull CDF (anchor L = yA / (1 - exp(-(xA/λ)^k)))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        def weibull_pred(lmbda: float, k: float):
            if lmbda <= eps:
                return None
            den = 1.0 - np.exp(- (max(xA, 0.0) / (lmbda + eps)) ** k)
            if den <= eps:
                return None
            L = yA / den
            return L * (1.0 - np.exp(- (np.maximum(xs, 0.0) / (lmbda + eps)) ** k))

        wp, yhat, _ = best_on_grid(lambda lmbda, k: weibull_pred(lmbda, k),
                                   {"lmbda": lambda_grid, "k": weib_k_grid})
        if yhat is not None:
            results.append(["weibull_cdf", score(ys, yhat, k_params=2)])  # λ, k searched
            den = 1.0 - np.exp(- (xA / (wp["lmbda"] + eps)) ** wp["k"])
            L_star = float(yA / den)
            best_params_map["weibull_cdf"] = {"L": L_star, **wp}

    # 7) Richards (anchor L = yA * [1 + ν exp(-k (xA - x0))]^(1/ν))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        nu_grid = np.linspace(0.1, 5.0, 12)

        def richards_pred(k: float, x0c: float, nu: float):
            inner = 1.0 + nu * np.exp(-k * (xA - x0c))
            if inner <= eps:
                return None
            L = yA * (inner ** (1.0 / nu))
            return L * (1.0 + nu * np.exp(-k * (xs - x0c))) ** (-1.0 / nu)

        rp, yhat, _ = best_on_grid(lambda k, x0c, nu: richards_pred(k, x0c, nu),
                                   {"k": k_grid_common, "x0c": x0_grid, "nu": nu_grid})
        if yhat is not None:
            results.append(["richards", score(ys, yhat, k_params=3)])  # k, x0, ν searched
            inner = 1.0 + rp["nu"] * np.exp(-rp["k"] * (xA - rp["x0c"]))
            L_star = float(yA * (inner ** (1.0 / rp["nu"])))
            best_params_map["richards"] = {"L": L_star, **rp}

    # 8) Log-saturation (anchor a = yA / ln(1 + b xA))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        def log_sat_pred(b: float):
            if (1.0 + b * xA) <= 0.0:
                return None
            a = yA / np.log(1.0 + b * xA)
            s = 1.0 + b * np.maximum(xs, 0.0)
            if np.any(s <= 0.0):
                return None
            return a * np.log(s)

        lp, yhat, _ = best_on_grid(lambda b: log_sat_pred(b), {"b": b_grid})
        if yhat is not None:
            results.append(["log_saturation", score(ys, yhat, k_params=1)])  # b searched
            a_star = float(yA / np.log(1.0 + lp["b"] * xA))
            best_params_map["log_saturation"] = {"a": a_star, **lp}

    # --- sort & return (descending score) ---
    results.sort(key=lambda t: t[1], reverse=True)
    return (results, best_params_map) if return_params else results


if __name__ == "__main__":
    main()


"""


RULE OUT
hill, weibull, logistic, gompertz, exp_sat

LEAVES
power, log_saturation, richards

note - if we can avoid the disasters in Hill, it does great.

{"7675366218":
[['hill', 0.9986640571473714],
['weibull_cdf', 0.998629881832037], 
['power', 0.9960011683158365],
['log_saturation', 0.9936855852287562], 
['richards', 0.9913907531967512], 
['logistic', 0.9913056429272957], 
['gompertz', 0.9901809122350567], 
['exp_sat', 0.9624306426572691]]


,"11098511556":
[['richards', 0.9999986659139375], 
['logistic', 0.9999820670916848], 
['log_saturation', 0.9997991897986588], 
['gompertz', 0.9907675206593791],
['power', 0.988273574719985],
['exp_sat', 0.921948464167379],
['weibull_cdf', 0.6830106459515867],
['hill', -0.4810436954057915]]

Strategy 8103063123 best fit metrics:
[['log_saturation', 0.9893953858945801],
['power', 0.9673549840881244],
['weibull_cdf', 0.9670919516093316],
['richards', 0.9493811656623715],
['gompertz', 0.9411377429670286],
['logistic', 0.9394485041000403],
['exp_sat', 0.9188702748144097],
['hill', 0.5628859706334455]]

Strategy 7226214573 best fit metrics:
[['hill', 0.9997499486920148],
['log_saturation', 0.9997124165636178],
['weibull_cdf', 0.9997046289042271],
['richards', 0.9929377301900213],
['logistic', 0.9929038281431993],
['power', 0.9914063911699142],
['gompertz', 0.9862915040149816],
['exp_sat', 0.9797253370587454]]

Strategy 8532782428 best fit metrics:
[['richards', 0.9908453753028335],
['weibull_cdf', 0.9605494972429769],
['log_saturation', 0.9597128071484007],
['hill', 0.9339583459401117],
['power', 0.9326011298728688],
['exp_sat', 0.9258358126842423],
['logistic', 0.8884314827054673],
['gompertz', 0.8400321557330522]]
}

"""