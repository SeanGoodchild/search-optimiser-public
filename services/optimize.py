
from typing import Dict
import pandas as pd


def solve_mckp(discrete: dict[str, pd.DataFrame], budget: float | None, target: float | None) -> Dict[str, dict]:
    """
    Greedy heuristic MCKP: allocate cost across entity ladders by best marginal conversion per £
    (or lowest CPA for target mode). Replace with OR-Tools in production.
    """
    ladders = {eid: df.sort_values('cost').reset_index(drop=True) for eid, df in discrete.items()}
    idx = {eid: 0 for eid in ladders}
    current_cost = {eid: float(df.iloc[0]['cost']) for eid, df in ladders.items()}
    current_conv  = {eid: float(df.iloc[0]['conv']) for eid, df in ladders.items()}

    def totals():
        return sum(current_cost.values()), sum(current_conv.values())

    max_iters = sum(len(df) for df in ladders.values())
    for _ in range(max_iters):
        total_cost, total_conv = totals()

        best_eid = None
        best_metric = -1.0  # efficiency for max_conv
        best_cpa = float('inf')
        for eid, df in ladders.items():
            i = idx[eid]
            if i + 1 >= len(df):
                continue
            s0, c0 = float(df.iloc[i]['cost']), float(df.iloc[i]['conv'])
            s1, c1 = float(df.iloc[i+1]['cost']), float(df.iloc[i+1]['conv'])
            ds, dc = s1 - s0, c1 - c0
            if ds <= 0:
                continue
            else:
                eff = dc / ds
                if eff > best_metric:
                    best_metric = eff
                    best_eid = eid
        if best_eid is None:
            break
        df = ladders[best_eid]
        next_cost = float(df.iloc[idx[best_eid]+1]['cost'])
        next_conv  = float(df.iloc[idx[best_eid]+1]['conv'])
        idx[best_eid] += 1
        current_cost[best_eid] = next_cost
        current_conv[best_eid]  = next_conv

    result = {}
    for eid, df in ladders.items():
        i = idx[eid]
        row = df.iloc[i]
        result[eid] = {"cost": float(row['cost']), "conv": float(row['conv'])}
    return result
