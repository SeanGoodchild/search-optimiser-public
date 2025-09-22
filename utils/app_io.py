import streamlit as st
import pandas as pd
import json
from utils import modelling


POINTS_COL = 'target_cpa_point_list_points'
X_COL = 'costMicros'
Y_COL = 'biddableConversions'
Z_COL = 'targetCpaMicros'
TARGET_COL = 'target_cpa_target_cpa'

@st.cache_data
def import_upload(uploaded_file) -> dict:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        return {}

    records = df.to_dict(orient="records")
    return process_records(records)


def process_records(records: list[dict]) -> dict:
    output = {}
    for i, row in enumerate(records, start=1):
        try:
            strategy_id = row["bidding_strategy_id"]
            if strategy_id == '':
                continue  # skip empty rows
            entry = {
                "id": strategy_id,
                "name": row["bidding_strategy_name"],
                "current_target": row[TARGET_COL],
                "input_x_points": [],
                "input_y_points": [],
                "input_z_points": [],
                "starting_point_index": None,
                "meta": {}
            }

            if POINTS_COL not in row.keys():
                raise Exception(f"Row {i} ({strategy_id}): Missing '{POINTS_COL}' column.")

            for column_name, column_value in row.items():
                if column_name == POINTS_COL:
                    points = json.loads(column_value)
                    if X_COL not in points[0].keys() or Y_COL not in points[0].keys():
                        raise Exception(f"Row {i} ({strategy_id}): Points must include '{X_COL}' and '{Y_COL}' fields.")
                    for idx, point in enumerate(points):
                        if 'micros' in X_COL.lower():
                            entry['input_x_points'].append(float(point[X_COL]) / 1_000_000)
                        else:
                            entry['input_x_points'].append(float(point[X_COL]))
                        if 'micros' in Y_COL.lower():
                            entry['input_y_points'].append(float(point[Y_COL]) / 1_000_000)
                        else:
                            entry['input_y_points'].append(float(point[Y_COL]))
                        if 'micros' in Z_COL.lower():
                            entry['input_z_points'].append(float(point[Z_COL]) / 1_000_000)
                        else:
                            entry['input_z_points'].append(float(point[Z_COL]))

                        # Identify starting point (the one that matches the current target CPA)
                        if entry['input_z_points'][-1] == row[TARGET_COL]:
                            raw_starting_point_index = idx

                else:
                    entry["meta"][column_name] = column_value

            # If there's no enough data, we can't fit a curve. This strategy will have to be 'locked' to one point
            if len(entry["input_x_points"]) < 2:
                raise Exception(f"Row {i} ({strategy_id}): not enough points to fit a curve.")

            # Returns a dict with a_factor, b_factor, x_fit, y_fit, z_fit
            curve_fit = modelling.find_curve_fit(
                entry["input_x_points"], entry["input_y_points"], raw_starting_point_index
            )
            entry.update(curve_fit)
    
            # Starting point is the one that corresponds to the current target CPA
            

            output[strategy_id] = entry

        except Exception as e_row:
            st.error(f"Row {i} ({row.get('bidding_strategy_id', 'unknown')}): {e_row}")
            raise e_row

    return output
