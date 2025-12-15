"""
Data Processing Module for Credit Risk Model
Task 3: Feature Engineering - FIXED VERSION
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


class RFMCalculator(BaseEstimator, TransformerMixin):
    """Calculate RFM (Recency, Frequency, Monetary) features - FIXED"""
    
    def __init__(self, customer_col='CustomerId', date_col='TransactionStartTime', 
                 amount_col='Amount', snapshot_date=None):
        self.customer_col = customer_col
        self.date_col = date_col
        self.amount_col = amount_col
        self.snapshot_date = snapshot_date
        self.rfm_features_ = None
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Calculate RFM features for each customer"""
        print("Calculating RFM features...")
        
        # Make a copy
        df = X.copy()
        
        # Convert date column
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # Set snapshot date (max date in data + 1 day)
        if self.snapshot_date is None:
            self.snapshot_date = df[self.date_col].max() + pd.Timedelta(days=1)
        
        # Calculate RFM - FIXED: Don't count customer_col
        rfm = df.groupby(self.customer_col).agg({
            self.date_col: lambda x: (self.snapshot_date - x.max()).days,  # Recency
            self.amount_col: ['count', 'sum']  # Frequency and Monetary
        }).reset_index()
        
        # Flatten the multi-level columns
        rfm.columns = [self.customer_col, 'recency', 'frequency', 'monetary']
        
        # Calculate additional features
        rfm['avg_transaction_value'] = rfm['monetary'] / rfm['frequency']
        rfm['transaction_std'] = df.groupby(self.customer_col)[self.amount_col].std().values
        rfm['transaction_std'].fillna(0, inplace=True)
        
        self.rfm_features_ = rfm
        print(f"RFM features calculated for {len(rfm)} customers")
        
        return rfm


class DataProcessor:  # Changed from FeatureEngineer to DataProcessor for simplicity
    """Main feature engineering class - SIMPLIFIED"""
    
    def __init__(self):
        self.features_df = None
        
    def extract_time_features(self, df):
        """Extract time-based features"""
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            
            # Time features
            df['transaction_hour'] = df['TransactionStartTime'].dt.hour
            df['transaction_day'] = df['TransactionStartTime'].dt.day
            df['transaction_month'] = df['TransactionStartTime'].dt.month
            df['transaction_year'] = df['TransactionStartTime'].dt.year
            df['transaction_dayofweek'] = df['TransactionStartTime'].dt.dayofweek
            df['transaction_weekend'] = df['transaction_dayofweek'].isin([5, 6]).astype(int)
            
            print("✅ Time features extracted")
        return df
    
    def create_customer_features(self, df):
        """Create all customer-level features"""
        print("\n" + "="*60)
        print("CREATING CUSTOMER FEATURES")
        print("="*60)
        
        # 1. Basic aggregates
        print("\n1. Basic aggregates...")
        agg_features = df.groupby('CustomerId').agg({
            'Amount': ['count', 'sum', 'mean', 'std', 'min', 'max'],
            'TransactionId': 'nunique'
        }).reset_index()
        
        agg_features.columns = [
            'customer_id', 'transaction_count', 'total_amount', 'avg_amount',
            'std_amount', 'min_amount', 'max_amount', 'unique_transactions'
        ]
        
        # 2. RFM features
        print("2. RFM features...")
        rfm_calc = RFMCalculator()
        rfm_features = rfm_calc.transform(df)
        rfm_features.rename(columns={'CustomerId': 'customer_id'}, inplace=True)
        
        # 3. Diversity features
        print("3. Diversity features...")
        diversity_features = []
        
        if 'ProviderId' in df.columns:
            provider_div = df.groupby('CustomerId')['ProviderId'].nunique().reset_index()
            provider_div.columns = ['customer_id', 'provider_diversity']
            diversity_features.append(provider_div)
            
        if 'ProductCategory' in df.columns:
            product_div = df.groupby('CustomerId')['ProductCategory'].nunique().reset_index()
            product_div.columns = ['customer_id', 'product_diversity']
            diversity_features.append(product_div)
            
        if 'ChannelId' in df.columns:
            channel_div = df.groupby('CustomerId')['ChannelId'].nunique().reset_index()
            channel_div.columns = ['customer_id', 'channel_diversity']
            diversity_features.append(channel_div)
        
        # 4. Merge all features
        print("4. Merging all features...")
        
        # Start with aggregate features
        all_features = agg_features
        
        # Merge RFM
        all_features = pd.merge(all_features, rfm_features, on='customer_id', how='left')
        
        # Merge diversity features
        for div_df in diversity_features:
            all_features = pd.merge(all_features, div_df, on='customer_id', how='left')
        
        # 5. Calculate derived features
        print("5. Derived features...")
        all_features['amount_range'] = all_features['max_amount'] - all_features['min_amount']
        all_features['monetary_per_day'] = all_features['monetary'] / (all_features['recency'] + 1)
        
        # Fill NaN values
        all_features.fillna(0, inplace=True)
        
        print(f"\n✅ Feature engineering completed!")
        print(f"   Total customers: {len(all_features)}")
        print(f"   Total features: {all_features.shape[1]}")
        
        self.features_df = all_features
        return all_features
    
    def save_features(self, output_path):
        """Save features to CSV"""
        if self.features_df is not None:
            self.features_df.to_csv(output_path, index=False)
            print(f"✅ Features saved to: {output_path}")
        else:
            print("❌ No features to save. Run create_customer_features first.")


# Simplified test function
def create_features(input_path, output_path):
    """One function to create all features"""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Data shape: {df.shape}")
    
    processor = DataProcessor()
    
    # Extract time features
    df = processor.extract_time_features(df)
    
    # Create customer features
    features = processor.create_customer_features(df)
    
    # Save
    features.to_csv(output_path, index=False)
    print(f"\n✅ Features saved to {output_path}")
    return features


# Run if executed directly
if __name__ == "__main__":
    print("="*60)
    print("RUNNING FEATURE ENGINEERING")
    print("="*60)
    
    input_path = '../data/raw/data.csv'
    output_path = '../data/processed/features_data.csv'
    
    features = create_features(input_path, output_path)
    
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    print(f"Features created: {features.shape[1]}")
    print(f"Customers: {features.shape[0]}")
    print("\nFirst 5 customers:")
    print(features.head())