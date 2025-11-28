import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

CSV_PATH = "tests/data/measures/data_drift.csv"

df = pd.read_csv(CSV_PATH)

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")

os.makedirs("metric_visuals", exist_ok=True)


plt.figure(figsize=(10, 5))
plt.plot(df["time"], df["score"], marker="o", linewidth=2)
plt.title("Data Drift Over Time")
plt.xlabel("Date")
plt.ylabel("Drift Score")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("metric_visuals/drift_timeseries.png", dpi=300)
plt.show()


df["rolling"] = df["score"].rolling(window=5, min_periods=1).mean()

plt.figure(figsize=(10, 5))
plt.plot(df["time"], df["score"], marker="o", alpha=0.6, label="Daily drift")
plt.plot(df["time"], df["rolling"], linewidth=3, label="Rolling avg (5d)")
plt.title("Drift Score with Trend (Rolling Average)")
plt.xlabel("Date")
plt.ylabel("Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("metric_visuals/drift_timeseries_rolling.png", dpi=300)
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(df["score"], kde=True)
plt.title("Distribution of Drift Scores")
plt.xlabel("Drift Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("metric_visuals/drift_score_distribution.png", dpi=300)
plt.show()


threshold = 0.25
high_drift = (df["score"] > threshold).sum()
normal = len(df) - high_drift

plt.figure(figsize=(6, 6))
plt.pie(
    [high_drift, normal],
    labels=["High Drift", "Normal"],
    colors=["#C0392B", "#2980B9"],
    autopct="%1.1f%%",
    startangle=140,
    wedgeprops={"width": 0.3},
)
plt.title(f"Drift Severity (threshold={threshold})")
plt.savefig("metric_visuals/drift_donut.png", dpi=300)
plt.show()

print("\n✓ Graphs generated in metric_visuals/")
