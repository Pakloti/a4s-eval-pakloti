from pathlib import Path
from datetime import datetime, timedelta
import uuid

import numpy as np
import warnings
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

warnings.filterwarnings("ignore")


def select_driftable_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique() <= 1:
            continue
        if col.startswith("ratio_"):
            continue
        if col.startswith(("id", "uid", "pid")):
            continue
        if df[col].nunique() < 5:
            continue
        cols.append(col)
    return cols


ROOT = Path(__file__).resolve().parents[2]
ref_path = ROOT / "tests" / "data" / "lcld_v2_train_800.csv"
curr_path = ROOT / "tests" / "data" / "lcld_v2_test_400.csv"
out_path = ROOT / "tests" / "data" / "measures" / "data_drift.csv"

reference_df = pd.read_csv(ref_path)
current_df_base = pd.read_csv(curr_path)

numeric_cols = select_driftable_columns(reference_df)
if not numeric_cols:
    raise RuntimeError("No driftable numeric columns found.")

np.random.seed(42)

n_runs = 50
rows = []
base_time = datetime(2016, 3, 1)

for i in range(n_runs):
    current_df = current_df_base.copy()

    n_cols_to_drift = np.random.randint(1, max(2, len(numeric_cols) // 2 + 1))
    drift_cols = np.random.choice(numeric_cols, size=n_cols_to_drift, replace=False)

    for col in drift_cols:
        factor = np.random.uniform(1.1, 1.8)
        current_df[col] = current_df[col] * factor

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    res = report.as_dict()
    result = res["metrics"][0]["result"]
    share_drift = float(result["share_of_drifted_columns"])

    time_str = (base_time + timedelta(days=i)).strftime("%Y-%m-%dT%H:%M:%S")

    rows.append(
        {
            "name": "data_drift",
            "score": share_drift,
            "time": time_str,
            "feature_pid": str(uuid.uuid4()),
        }
    )

df_out = pd.DataFrame(rows)
df_out.to_csv(out_path, index=False)

print(f"Written {len(df_out)} rows to {out_path}")
print("First rows:")
print(df_out.head())
