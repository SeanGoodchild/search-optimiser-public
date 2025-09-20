import csv
import json
import numpy as np
import streamlit as st
from services import store


@st.cache_data
def import_and_init(csv_path: str) -> dict:
    output = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strategy_id = row["bidding_strategy_id"]
            entry = output.setdefault(strategy_id, {
                "id": strategy_id,
                "name": row["bidding_strategy_name"],
                "cost_points": [],
                "conversion_points": [],
                "target_cpa_points": [],
                "chart_events": [],
                "selected_point": None,
                "curve_fit": None,
            })
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
                            entry["selected_point"] = (cost, convs)

            entry["curve_fit_data"] = find_curve_fit(entry["cost_points"], entry["conversion_points"])
            store.set_strategy(strategy_id, entry)


def find_curve_fit(x_series: list, y_series:list) -> dict:
    num_steps = 100
    xs = np.array(x_series, dtype=float)
    ys = np.array(y_series, dtype=float)

    # Fit to a power curve y = a * x^b
    coeffs = np.polyfit(np.log(xs + 1e-6), np.log(ys + 1e-6), 1)
    a = np.exp(coeffs[1])
    b = coeffs[0]

    x_fit = np.linspace(xs.min()*0.8, xs.max()*1.2, num_steps)
    y_fit = a * np.power(x_fit, b)

    return {
        'a': a,
        'b': b,
        'x_fit': x_fit,
        'y_fit': y_fit,
    }