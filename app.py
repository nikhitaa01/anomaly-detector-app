import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.title("Simple Anomaly Detection on Revenue Data")

# Generate example data
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", periods=100)
revenue = np.random.normal(loc=10000, scale=800, size=100)
revenue[25] = 4000   # anomaly low
revenue[70] = 18000  # anomaly high

df = pd.DataFrame({"Date": dates, "Revenue": revenue})
df.set_index("Date", inplace=True)

# Calculate revenue difference for anomaly detection
df["Revenue_diff"] = df["Revenue"].diff().fillna(0)

# Fit Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
df["anomaly"] = model.fit_predict(df[["Revenue_diff"]])

st.write("### Revenue Data with Detected Anomalies")
st.write(df)

# Plot revenue and anomalies
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df["Revenue"], label="Revenue")
ax.scatter(df[df["anomaly"] == -1].index, df[df["anomaly"] == -1]["Revenue"], color="red", label="Anomaly")
ax.set_xlabel("Date")
ax.set_ylabel("Revenue")
ax.legend()
st.pyplot(fig)
