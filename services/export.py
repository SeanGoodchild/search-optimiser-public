
import pandas as pd

def result_to_csv(result: dict[str, dict]) -> bytes:
    rows = [{"entity_id": eid, "cost": d['cost'], "conv": d.get('conv', None)} for eid, d in result.items()]
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")
