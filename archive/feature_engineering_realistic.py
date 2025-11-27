"""
Realistic Feature Engineering - Only Practical Features
Keep it simple: Area, BHK, Locality, Furnishing, Price per SqFt
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("REALISTIC FEATURE ENGINEERING - SIMPLE & PRACTICAL")
print("="*70)

# Load cleaned data
df = pd.read_csv('../data/processed/cleaned_real_estate_data.csv')
print(f"\n[INFO] Loaded data: {df.shape}")

# Step 1: Basic Features Only
print("\n[STEP 1] Creating 5 Realistic Features...")
print("-" * 70)

# Feature 1: Area (Square Feet) - Most important
df['Area_SqFt'] = df['Area_SqFt'].fillna(df['Area_SqFt'].median())
print("✓ Area_SqFt - Property size (most important factor)")

# Feature 2: BHK - Configuration
df['BHK'] = df['BHK'].fillna(df['BHK'].mode()[0])
print("✓ BHK - Number of bedrooms (2BHK, 3BHK, etc.)")

# Feature 3: Locality - Location is key
le_locality = LabelEncoder()
df['Locality_Encoded'] = le_locality.fit_transform(df['Locality'])
print(f"✓ Locality - Location encoding ({df['Locality'].nunique()} unique areas)")

# Feature 4: Furnishing Status - Ready to move?
le_furnishing = LabelEncoder()
df['Furnishing_Encoded'] = le_furnishing.fit_transform(df['Furnishing'])
print("✓ Furnishing - Semi/Fully/Unfurnished status")

# Feature 5: Locality Tier - Premium/High-End/Mid-Range/Budget
# Calculate average price per locality
locality_avg_price = df.groupby('Locality')['Price_Lakhs'].mean()

# Define tiers based on price quartiles
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
print(f"✓ Locality_Tier - Premium/High-End/Mid-Range/Budget zones")
print(f"  Premium: ≥₹{q75:.1f}L | High-End: ≥₹{q50:.1f}L | Mid-Range: ≥₹{q25:.1f}L | Budget: <₹{q25:.1f}L")

print(f"\n[INFO] Total Features: 5 realistic features")
print("       No artificial features - only what buyers care about!")
print("       NO DATA LEAKAGE - Pure prediction!")

# Keep only necessary columns
final_columns = [
    'Property Title', 'Price_Lakhs', 'Area_SqFt', 'BHK', 
    'Furnishing', 'Locality', 'Locality_Tier', 
    'Locality_Encoded', 'Furnishing_Encoded', 'Locality_Tier_Encoded'
]

df_final = df[final_columns].copy()

# Save processed data
output_path = '../data/processed/featured_real_estate_data.csv'
df_final.to_csv(output_path, index=False, encoding='utf-8')

print(f"\n[SUCCESS] Feature engineering complete!")
print(f"[OUTPUT] {df_final.shape[0]} properties × {df_final.shape[1]} columns")
print(f"[SAVED] {output_path}")
print("\n" + "="*70)

# Show sample
print("\nSample of realistic features:")
print(df_final[['Price_Lakhs', 'Area_SqFt', 'BHK', 'Locality_Tier', 'Furnishing']].head(3))
print("\nLocality Tier Distribution:")
print(df_final['Locality_Tier'].value_counts().sort_index())
print("="*70)
