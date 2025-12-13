import pandas as pd
import numpy as np

print("=== TASK 3: FEATURE ENGINEERING ===")

# Load data
df = pd.read_csv("../data/raw/data.csv")
print(f"Loaded {len(df)} rows")

# Extract numeric IDs
df['AccountId_numeric'] = df['AccountId'].str.extract(r'(\d+)$').astype(float)

# RFM Calculation
snapshot_date = pd.to_datetime('2019-02-28')
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime']).dt.tz_localize(None)

# Group by customer
customer_rfm = df.groupby('AccountId_numeric').agg(
    Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
    Frequency=('TransactionId', 'count'),
    Monetary_Total=('Amount', 'sum'),
    Monetary_Avg=('Amount', 'mean')
).reset_index()

print(f"\nRFM calculated for {len(customer_rfm)} customers")
print(customer_rfm.head())

print("\n✓ TASK 3 COMPLETE: Features engineered")