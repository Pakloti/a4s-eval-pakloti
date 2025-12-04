import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parents[2]

ref_path = ROOT / "tests" / "data" / "lcld_v2_train_800.csv"
curr_path = ROOT / "tests" / "data" / "lcld_v2_test_400.csv"

reference_df = pd.read_csv(ref_path)
current_df_base = pd.read_csv(curr_path)

TARGET = "charged_off"

if TARGET not in reference_df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in dataset.")

def select_numeric_features(df):
    return [
        col for col in df.columns
        if col != TARGET and pd.api.types.is_numeric_dtype(df[col])
    ]

numeric_cols = select_numeric_features(reference_df)
if not numeric_cols:
    raise RuntimeError("No numeric columns found for the models.")


scaler = StandardScaler()
X_train = scaler.fit_transform(reference_df[numeric_cols])
y_train = reference_df[TARGET].values

X_test_base = current_df_base[numeric_cols].values
X_test_base = scaler.transform(X_test_base)
y_test = current_df_base[TARGET].values


models = {
    "logreg": LogisticRegression(max_iter=200),
    "randomforest": RandomForestClassifier(n_estimators=100),
    "svm": SVC(kernel="rbf")
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)

np.random.seed(42)
n_runs = 50
base_time = datetime(2016, 3, 1)

drift_scores = []
logreg_results = []
rf_results = []
svm_results = []

for i in range(n_runs):

    # Create a copy to use as drifted data
    curr_df = current_df_base.copy()

    n_cols_to_drift = np.random.randint(1, max(2, len(numeric_cols) // 2 + 1))
    drift_cols = np.random.choice(numeric_cols, size=n_cols_to_drift, replace=False)

    for col in drift_cols:
        factor = np.random.uniform(1.1, 1.8)
        curr_df[col] = curr_df[col] * factor

    # Compute drift score
    drift_score = len(drift_cols) / len(numeric_cols)

    X_test_drifted = scaler.transform(curr_df[numeric_cols])
    y_test_drifted = curr_df[TARGET].values

    # Evaluate every model
    acc_logreg = accuracy_score(y_test_drifted, models["logreg"].predict(X_test_drifted))
    acc_rf = accuracy_score(y_test_drifted, models["randomforest"].predict(X_test_drifted))
    acc_svm = accuracy_score(y_test_drifted, models["svm"].predict(X_test_drifted))

    timestamp = (base_time + timedelta(days=i)).strftime("%Y-%m-%dT%H:%M:%S")

    drift_scores.append([timestamp, drift_score])
    logreg_results.append([timestamp, acc_logreg])
    rf_results.append([timestamp, acc_rf])
    svm_results.append([timestamp, acc_svm])

out_dir = ROOT / "tests" / "data" / "measures"
out_dir.mkdir(exist_ok=True, parents=True)

pd.DataFrame(logreg_results, columns=["time", "accuracy"]).to_csv(out_dir / "results_logreg.csv", index=False)
pd.DataFrame(rf_results, columns=["time", "accuracy"]).to_csv(out_dir / "results_randomforest.csv", index=False)
pd.DataFrame(svm_results, columns=["time", "accuracy"]).to_csv(out_dir / "results_svm.csv", index=False)

print("All CSV files written to:", out_dir)