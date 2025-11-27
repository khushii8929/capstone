import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
import sys
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("ADVANCED FEATURE ENGINEERING - 18 POWERFUL FEATURES")
print("="*70)

# Load Cleaned Data
df = pd.read_csv('../data/processed/cleaned_real_estate_data.csv')
print(f"\n✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Fill missing bathrooms intelligently
df['Bathrooms'] = df.apply(
    lambda row: row['Bathrooms'] if pd.notna(row['Bathrooms']) 
    else max(1, row['BHK'] - 1) if row['BHK'] > 2 else row['BHK'], 
    axis=1
)
print(f"✓ Filled {df['Bathrooms'].isna().sum()} missing bathroom values")

# ============================================================================
# STEP 1: BASE FEATURES - Encoding Categorical Variables
# ============================================================================
print("\n" + "="*70)
print("STEP 1: ENCODING CATEGORICAL FEATURES")
print("="*70)

# Encode Furnishing (Furnished=0, Semi-Furnished=1, Unfurnished=2, Unknown=3)
le_furnishing = LabelEncoder()
df['Furnishing_Encoded'] = le_furnishing.fit_transform(df['Furnishing'].fillna('Unknown'))
print(f"✓ Furnishing_Encoded: {df['Furnishing_Encoded'].nunique()} categories")

# Encode Locality (high cardinality - 1238 localities)
le_locality = LabelEncoder()
df['Locality_Encoded'] = le_locality.fit_transform(df['Locality'])
print(f"✓ Locality_Encoded: {df['Locality_Encoded'].nunique()} localities")

# Encode Seller Type
if 'Seller Type' in df.columns:
    le_seller = LabelEncoder()
    df['Seller_Type_Encoded'] = le_seller.fit_transform(df['Seller Type'].fillna('Unknown'))
    print(f"✓ Seller_Type_Encoded: {df['Seller_Type_Encoded'].nunique()} types")

# ============================================================================
# STEP 2: POWERFUL INTERACTION FEATURES (Capture Feature Synergies)
# ============================================================================
print("\n" + "="*70)
print("STEP 2: INTERACTION FEATURES (Capture Complex Relationships)")
print("="*70)

# Feature 1: Area_BHK_Interaction (MOST POWERFUL - captures size+config synergy)
df['Area_BHK_Interaction'] = df['Area_SqFt'] * df['BHK']
print("✓ Area_BHK_Interaction: Area × BHK (size-configuration synergy)")

# Feature 2: Locality_Area_Interaction (captures premium area patterns)
df['Locality_Area'] = df['Locality_Encoded'] * df['Area_SqFt']
print("✓ Locality_Area: Locality × Area (location-size premium)")

# Feature 3: Locality_BHK_Interaction (captures neighborhood config patterns)
df['Locality_BHK'] = df['Locality_Encoded'] * df['BHK']
print("✓ Locality_BHK: Locality × BHK (location-config patterns)")

# Feature 4: Furnishing_Area_Interaction (furnished larger properties worth more)
df['Furnishing_Area'] = df['Furnishing_Encoded'] * df['Area_SqFt']
print("✓ Furnishing_Area: Furnishing × Area (furnishing premium on size)")

# Feature 5: Bathroom_Area_Interaction (luxury bathroom-area relationship)
df['Bathroom_Area'] = df['Bathrooms'] * df['Area_SqFt']
print("✓ Bathroom_Area: Bathrooms × Area (luxury amenity effect)")

# ============================================================================
# STEP 3: POLYNOMIAL FEATURES (Capture Non-Linear Relationships)
# ============================================================================
print("\n" + "="*70)
print("STEP 3: POLYNOMIAL FEATURES (Non-Linear Price Curves)")
print("="*70)

# Feature 6: Area_Squared (larger properties have exponential price growth)
df['Area_Squared'] = df['Area_SqFt'] ** 2
print("✓ Area_Squared: Area² (non-linear size premium)")

# Feature 7: BHK_Squared (luxury configs grow non-linearly)
df['BHK_Squared'] = df['BHK'] ** 2
print("✓ BHK_Squared: BHK² (luxury configuration premium)")

# Feature 8: Bathroom_Squared (luxury bathroom premium)
df['Bathroom_Squared'] = df['Bathrooms'] ** 2
print("✓ Bathroom_Squared: Bathrooms² (luxury amenity premium)")

# ============================================================================
# STEP 4: RATIO & DENSITY FEATURES (Efficiency Metrics)
# ============================================================================
print("\n" + "="*70)
print("STEP 4: RATIO FEATURES (Efficiency & Quality Indicators)")
print("="*70)

# Feature 9: Area_per_BHK (spaciousness measure)
df['Area_per_BHK'] = df['Area_SqFt'] / df['BHK']
print("✓ Area_per_BHK: Area/BHK (room spaciousness)")

# Feature 10: Bathroom_BHK_Ratio (luxury indicator)
df['Bathroom_BHK_Ratio'] = df['Bathrooms'] / df['BHK']
print("✓ Bathroom_BHK_Ratio: Bathrooms/BHK (luxury ratio)")

# Feature 11: Area_per_Room (total rooms = BHK + Bathrooms)
df['Total_Rooms'] = df['BHK'] + df['Bathrooms']
df['Area_per_Room'] = df['Area_SqFt'] / df['Total_Rooms']
print("✓ Area_per_Room: Area/(BHK+Bathrooms) (overall spaciousness)")

# ============================================================================
# STEP 5: CATEGORICAL BINNING FEATURES (Market Segmentation)
# ============================================================================
print("\n" + "="*70)
print("STEP 5: CATEGORICAL BINNING (Market Segments)")
print("="*70)

# Feature 12: Is_Large_Property (top 25% by area)
area_75th = df['Area_SqFt'].quantile(0.75)
df['Is_Large_Property'] = (df['Area_SqFt'] > area_75th).astype(int)
print(f"✓ Is_Large_Property: Area > {area_75th:.0f} sq.ft (top 25%)")

# Feature 13: Is_Luxury_Config (4+ BHK)
df['Is_Luxury_Config'] = (df['BHK'] >= 4).astype(int)
luxury_pct = (df['Is_Luxury_Config'].sum() / len(df)) * 100
print(f"✓ Is_Luxury_Config: BHK ≥ 4 ({luxury_pct:.1f}% of properties)")

# Feature 14: Is_Premium_Bathroom (more bathrooms than BHK-1)
df['Is_Premium_Bathroom'] = (df['Bathrooms'] >= df['BHK']).astype(int)
premium_bath_pct = (df['Is_Premium_Bathroom'].sum() / len(df)) * 100
print(f"✓ Is_Premium_Bathroom: Bathrooms ≥ BHK ({premium_bath_pct:.1f}%)")

# Feature 15: Is_Furnished (Furnished or Semi-Furnished)
df['Is_Furnished'] = (df['Furnishing_Encoded'] <= 1).astype(int)
furnished_pct = (df['Is_Furnished'].sum() / len(df)) * 100
print(f"✓ Is_Furnished: Furnished/Semi-Furnished ({furnished_pct:.1f}%)")

# ============================================================================
# STEP 6: LOCALITY-BASED FEATURES (Location Intelligence)
# ============================================================================
print("\n" + "="*70)
print("STEP 6: LOCALITY FEATURES (Location Intelligence)")
print("="*70)

# Feature 16: Locality_Property_Count (supply measure - NO DATA LEAKAGE)
locality_counts = df.groupby('Locality').size()
df['Locality_PropertyCount'] = df['Locality'].map(locality_counts)
top_locality_threshold = df['Locality_PropertyCount'].quantile(0.80)
print(f"✓ Locality_PropertyCount: Properties per locality (supply metric)")

# Feature 17: Is_High_Supply_Area (top 20% by property count)
df['Is_High_Supply_Area'] = (df['Locality_PropertyCount'] > top_locality_threshold).astype(int)
high_supply_pct = (df['Is_High_Supply_Area'].sum() / len(df)) * 100
print(f"✓ Is_High_Supply_Area: High supply localities ({high_supply_pct:.1f}%)")

# Feature 18: Locality_Avg_Area (average property size in locality)
locality_avg_area = df.groupby('Locality')['Area_SqFt'].transform('mean')
df['Locality_Avg_Area'] = locality_avg_area
print(f"✓ Locality_Avg_Area: Average area in locality (NO DATA LEAKAGE)")

# ============================================================================
# ADDITIONAL FEATURES FOR VISUALIZATION (Not used in modeling)
# ============================================================================
print("\n" + "="*70)
print("STEP 7: ANALYSIS FEATURES (For Visualization Only)")
print("="*70)

# Price per SqFt (for analysis, NOT modeling - derived from target)
df['Price_Per_SqFt'] = (df['Price_Lakhs'] * 100000) / df['Area_SqFt']
print("✓ Price_Per_SqFt (analysis only - NOT for modeling)")

# Area categories
def categorize_area(area):
    if area < 800:
        return 'Small'
    elif area < 1500:
        return 'Medium'
    elif area < 2500:
        return 'Large'
    else:
        return 'Extra Large'

df['Area_Category'] = df['Area_SqFt'].apply(categorize_area)
le_area = LabelEncoder()
df['Area_Category_Encoded'] = le_area.fit_transform(df['Area_Category'])
print("✓ Area_Category (for charts)")

# Property type
def classify_property_type(bhk):
    if bhk == 1:
        return 'Studio/1BHK'
    elif bhk == 2:
        return '2BHK'
    elif bhk == 3:
        return '3BHK'
    elif bhk == 4:
        return '4BHK'
    else:
        return 'Luxury (5+ BHK)'

df['Property_Type'] = df['BHK'].apply(classify_property_type)
le_prop = LabelEncoder()
df['Property_Type_Encoded'] = le_prop.fit_transform(df['Property_Type'])
print("✓ Property_Type (for charts)")

# Price segments
def categorize_price(price):
    if price < 50:
        return 'Budget (< 50L)'
    elif price < 100:
        return 'Affordable (50L-1Cr)'
    elif price < 200:
        return 'Premium (1-2Cr)'
    else:
        return 'Luxury (> 2Cr)'

df['Price_Segment'] = df['Price_Lakhs'].apply(categorize_price)
le_price = LabelEncoder()
df['Price_Segment_Encoded'] = le_price.fit_transform(df['Price_Segment'])
print("✓ Price_Segment (for charts)")

# Space quality
df['Space_Quality'] = df['Area_per_BHK'].apply(
    lambda x: 'Compact' if x < 400 else ('Standard' if x < 600 else ('Spacious' if x < 800 else 'Very Spacious'))
)
le_space = LabelEncoder()
df['Space_Quality_Encoded'] = le_space.fit_transform(df['Space_Quality'])
print("✓ Space_Quality (for charts)")

# ============================================================================
# SAVE FEATURED DATASET
# ============================================================================
# ============================================================================
# SAVE FEATURED DATASET
# ============================================================================
output_file = '../data/processed/featured_real_estate_data.csv'
df.to_csv(output_file, index=False)

print("\n" + "="*70)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("="*70)
print(f"\n📁 Output: {output_file}")
print(f"📊 Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n🎯 18 POWERFUL MODEL FEATURES CREATED:")
print("\n📍 BASE FEATURES (5):")
print("   1. Area_SqFt - Property size")
print("   2. BHK - Bedroom count")
print("   3. Bathrooms - Bathroom count")
print("   4. Furnishing_Encoded - Furnishing status")
print("   5. Locality_Encoded - Location encoding")
print("\n🔄 INTERACTION FEATURES (5):")
print("   6. Area_BHK_Interaction ⭐ - Size × Config synergy")
print("   7. Locality_Area - Location × Size premium")
print("   8. Locality_BHK - Location × Config patterns")
print("   9. Furnishing_Area - Furnishing × Size premium")
print("   10. Bathroom_Area - Bathroom × Size luxury")
print("\n📈 POLYNOMIAL FEATURES (3):")
print("   11. Area_Squared - Non-linear size effect")
print("   12. BHK_Squared - Non-linear config effect")
print("   13. Bathroom_Squared - Luxury bathroom premium")
print("\n📊 RATIO FEATURES (3):")
print("   14. Area_per_BHK - Room spaciousness")
print("   15. Bathroom_BHK_Ratio - Luxury ratio")
print("   16. Area_per_Room - Overall efficiency")
print("\n🏷️ BINARY FLAGS (2):")
print("   17. Is_Large_Property - Top 25% by area")
print("   18. Is_Luxury_Config - Premium 4+ BHK")
print("\n📍 LOCALITY FEATURES (Bonus):")
print("   • Locality_PropertyCount - Supply metric")
print("   • Is_High_Supply_Area - High supply flag")
print("   • Locality_Avg_Area - Neighborhood size pattern")
print("\n✅ NO DATA LEAKAGE - All features available at prediction time")
print("="*70)
