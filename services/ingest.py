
import pandas as pd

def load_data() -> pd.DataFrame:
    raw = pd.read_csv("data/sample.csv")
    # Add a point_id column to each entry for a given strategy_id
    raw['point_id'] = raw.groupby(['strategy_id']).cumcount().astype(str).apply(lambda x: f"P{x}")
    return raw