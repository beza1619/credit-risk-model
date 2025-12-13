"""
Task 3: Feature Engineering - FIXED VERSION
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath="data/raw/data.csv"):
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)
    id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 
               'CustomerId', 'ProviderId', 'ProductId', 'ChannelId']
    for col in id_cols:
        if col in df.columns:
            df[f'{col}_numeric'] = df[col].str.extract(r'(\d+)$').astype(float)
    return df

class RFMFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, snapshot_date='2019-02-28'):
        self.snapshot_date = pd.to_datetime(snapshot_date).tz_localize(None)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime']).dt.tz_localize(None)
        
        customer_stats = X.groupby('AccountId_numeric').agg({
            'TransactionStartTime': lambda x: (self.snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': ['sum', 'mean', 'std', 'min', 'max']
        })
        
        customer_stats.columns = [
            'Recency', 'Frequency', 'Monetary_Total', 
            'Monetary_Avg', 'Monetary_Std', 'Monetary_Min', 'Monetary_Max'
        ]
        
        X = X.merge(customer_stats, left_on='AccountId_numeric', 
                    right_index=True, how='left')
        return X

def main():
    print("=== Feature Engineering Test ===")
    df = load_and_prepare_data()
    print(f"✓ Loaded {len(df)} rows")
    
    rfm = RFMFeatureEngineer()
    df_rfm = rfm.fit_transform(df)
    
    print(f"✓ RFM features added. New columns:")
    print(f"  - {list(df_rfm.columns[-7:])}")
    
    output_path = "data/processed/features_data.csv"
    df_rfm.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}")
    print("Task 3 Complete!")

if __name__ == "__main__":
    main()