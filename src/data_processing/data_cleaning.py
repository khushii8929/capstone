"""
Data Cleaning Module
Handles loading and cleaning raw real estate data
"""

import pandas as pd
import numpy as np

def load_raw_data(filepath):
    """Load raw data from CSV"""
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """Clean and preprocess raw real estate data"""
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna(subset=['Price', 'Area', 'BHK', 'Locality'])
    
    # Parse price (remove , Cr, Lac)
    df['Price_Lakhs'] = df['Price'].apply(parse_price)
    
    # Parse area (extract numeric sq.ft)
    df['Area_SqFt'] = df['Area'].apply(parse_area)
    
    # Clean BHK
    df['BHK'] = pd.to_numeric(df['BHK'], errors='coerce')
    
    # Fill bathrooms with BHK if missing
    if 'Bathrooms' in df.columns:
        df['Bathrooms'] = df['Bathrooms'].fillna(df['BHK'])
    
    # Drop rows with invalid values
    df = df.dropna(subset=['Price_Lakhs', 'Area_SqFt', 'BHK'])
    
    # Remove unrealistic values
    df = df[
        (df['Price_Lakhs'] > 0) & 
        (df['Area_SqFt'] > 100) & 
        (df['BHK'] >= 1) & (df['BHK'] <= 10)
    ]
    
    return df

def parse_price(price_str):
    """Convert price string to lakhs"""
    try:
        price_str = str(price_str).replace('', '').replace(',', '').strip()
        if 'Cr' in price_str:
            return float(price_str.replace('Cr', '').strip()) * 100
        elif 'Lac' in price_str:
            return float(price_str.replace('Lac', '').strip())
        else:
            return float(price_str)
    except:
        return np.nan

def parse_area(area_str):
    """Extract square feet from area string"""
    try:
        area_str = str(area_str)
        if 'sqft' in area_str.lower():
            return float(area_str.lower().replace('sqft', '').replace(',', '').strip())
        return float(area_str)
    except:
        return np.nan

def save_cleaned_data(df, output_path):
    """Save cleaned data to CSV"""
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f" Cleaned data saved: {output_path}")
    print(f"  Properties: {len(df)}")
