"""
Feature Engineering Module
Creates realistic features for model training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def engineer_features(df):
    """Create realistic features from cleaned data"""
    
    df = df.copy()
    
    # Feature 1: Locality encoding
    le_locality = LabelEncoder()
    df['Locality_Encoded'] = le_locality.fit_transform(df['Locality'])
    
    # Feature 2: Furnishing encoding
    if 'Furnishing' in df.columns:
        le_furnishing = LabelEncoder()
        df['Furnishing_Encoded'] = le_furnishing.fit_transform(df['Furnishing'])
    else:
        df['Furnishing_Encoded'] = 0
    
    # Feature 3: Locality Tier (Premium/High-End/Mid-Range/Budget)
    df = add_locality_tier(df)
    
    # Feature 4: BHK  Locality Tier Interaction
    df['BHK_Tier_Interaction'] = df['BHK'] * df['Locality_Tier_Encoded']
    
    return df

def add_locality_tier(df):
    """Categorize localities into tiers based on average price"""
    
    locality_avg_price = df.groupby('Locality')['Price_Lakhs'].mean()
    
    q75 = locality_avg_price.quantile(0.75)
    q50 = locality_avg_price.quantile(0.50)
    q25 = locality_avg_price.quantile(0.25)
    
    def assign_tier(locality):
        avg_price = locality_avg_price.get(locality, locality_avg_price.median())
        if avg_price >= q75:
            return 'Premium'
        elif avg_price >= q50:
            return 'High-End'
        elif avg_price >= q25:
            return 'Mid-Range'
        else:
            return 'Budget'
    
    df['Locality_Tier'] = df['Locality'].apply(assign_tier)
    
    le_tier = LabelEncoder()
    df['Locality_Tier_Encoded'] = le_tier.fit_transform(df['Locality_Tier'])
    
    return df

def remove_outliers(df, column='Price_Lakhs'):
    """Remove outliers using IQR method"""
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    
    removed = len(df) - len(df_clean)
    print(f" Removed {removed} outliers ({removed/len(df)*100:.1f}%)")
    
    return df_clean

def get_feature_columns():
    """Return list of feature columns for modeling"""
    return [
        'Area_SqFt',
        'BHK',
        'Locality_Encoded',
        'Furnishing_Encoded',
        'Locality_Tier_Encoded',
        'BHK_Tier_Interaction'
    ]
