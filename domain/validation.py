
import pandas as pd
import numpy as np

def prune_dominated(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['cost','conv']).reset_index(drop=True)
    keep = []
    best = -np.inf
    for _, row in df.iterrows():
        if row['conv'] > best:
            keep.append(True)
            best = row['conv']
        else:
            keep.append(False)
    return df.loc[keep].reset_index(drop=True)

def enforce_monotone(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('cost').reset_index(drop=True)
    df['conv'] = df['conv'].cummax()
    return df
