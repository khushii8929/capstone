"""
Ahmedabad Real Estate Analytics - Complete Pipeline Executor
This script runs all notebooks in sequence and generates a comprehensive report
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime
import pickle

print("="*80)
print("AHMEDABAD REAL ESTATE ANALYTICS - END-TO-END PIPELINE")
print("="*80)
print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# STEP 1: DATA CLEANING
# ============================================================================
print("\n" + "="*80)
print("STEP 1: DATA CLEANING")
print("="*80)

print("\n📊 Loading raw data...")
df = pd.read_csv('ahmedabad_real_estate_data.csv')
print(f"Initial records: {len(df):,}")

# Remove duplicates
print("\n🔄 Removing duplicates...")
df = df.drop_duplicates(subset=['Property Title', 'Price', 'Area'], keep='first')
print(f"After removing duplicates: {len(df):,}")

# Clean Price column
print("\n💰 Cleaning Price column...")
def clean_price(price_str):
    if pd.isna(price_str) or price_str == 'N/A':
        return np.nan
    price_str = str(price_str).replace('₹', '').replace(',', '').strip()
    
    # Extract first numeric price if multiple prices exist
    import re
    match = re.search(r'([\d.]+)\s*(Cr|L|Lacs|Lakhs?)', price_str, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        if 'cr' in unit:
            return value * 100  # Convert Cr to Lakhs
        else:
            return value  # Already in Lakhs
    return np.nan

df['Price_Lakhs'] = df['Price'].apply(clean_price)
print(f"Valid prices: {df['Price_Lakhs'].notna().sum():,}")

# Clean Area column
print("\n📏 Cleaning Area column...")
def clean_area(area_str):
    if pd.isna(area_str) or area_str == 'N/A':
        return np.nan
    area_str = str(area_str).replace(',', '').strip()
    
    import re
    match = re.search(r'([\d.]+)\s*(sq\.?\s*ft|sqft|sq\.?\s*yd|sqyd)', area_str, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        if 'yd' in unit:
            return value * 9  # Convert sq yard to sq ft
        return value
    return np.nan

df['Area_SqFt'] = df['Area'].apply(clean_area)
print(f"Valid areas: {df['Area_SqFt'].notna().sum():,}")

# Clean BHK
print("\n🏠 Standardizing BHK...")
def standardize_bhk(bhk_str):
    if pd.isna(bhk_str) or bhk_str == 'N/A':
        return np.nan
    import re
    match = re.search(r'(\d+)', str(bhk_str))
    if match:
        return int(match.group(1))
    return np.nan

df['BHK'] = df['BHK'].apply(standardize_bhk)
print(f"Valid BHK: {df['BHK'].notna().sum():,}")

# Clean Bathrooms
def clean_bathrooms(bath_str):
    if pd.isna(bath_str) or bath_str == 'N/A':
        return np.nan
    import re
    match = re.search(r'(\d+)', str(bath_str))
    if match:
        return int(match.group(1))
    return np.nan

df['Bathroom'] = df['Bathrooms'].apply(clean_bathrooms)

# Remove outliers
print("\n🎯 Removing outliers...")
initial_count = len(df)
df = df[(df['Price_Lakhs'].notna()) & (df['Area_SqFt'].notna()) & (df['BHK'].notna())]

# IQR method for price
Q1_price = df['Price_Lakhs'].quantile(0.01)
Q3_price = df['Price_Lakhs'].quantile(0.99)
df = df[(df['Price_Lakhs'] >= Q1_price) & (df['Price_Lakhs'] <= Q3_price)]

# IQR method for area
Q1_area = df['Area_SqFt'].quantile(0.01)
Q3_area = df['Area_SqFt'].quantile(0.99)
df = df[(df['Area_SqFt'] >= Q1_area) & (df['Area_SqFt'] <= Q3_area)]

print(f"Removed {initial_count - len(df):,} outliers")
print(f"Final clean records: {len(df):,}")

# Fill missing bathrooms
df['Bathroom'] = df['Bathroom'].fillna(df['BHK'])

# Save cleaned data
df.to_csv('cleaned_real_estate_data.csv', index=False)
print("\n✅ Cleaned data saved: cleaned_real_estate_data.csv")

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("STEP 2: FEATURE ENGINEERING")
print("="*80)

print("\n🔧 Creating derived features...")

# Price per sqft
df['Price_Per_SqFt'] = df['Price_Lakhs'] * 100000 / df['Area_SqFt']

# Area categories
df['Area_Category'] = pd.cut(df['Area_SqFt'], 
                               bins=[0, 600, 1000, 1500, 10000],
                               labels=['Small', 'Medium', 'Large', 'XL'])

# Property type
def categorize_property(title):
    title = str(title).lower()
    if 'villa' in title or 'bungalow' in title:
        return 'Villa'
    elif 'builder floor' in title or 'floor' in title:
        return 'Builder Floor'
    else:
        return 'Apartment'

df['Property_Type'] = df['Property Title'].apply(categorize_property)

# Price segments
df['Price_Segment'] = pd.cut(df['Price_Lakhs'],
                               bins=[0, 50, 100, 200, 1000],
                               labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])

# Locality-based features
locality_stats = df.groupby('Locality')['Price_Lakhs'].agg(['mean', 'median', 'count']).reset_index()
locality_stats.columns = ['Locality', 'Locality_Avg_Price', 'Locality_Median_Price', 'Locality_Count']
df = df.merge(locality_stats, on='Locality', how='left')

# Value Score (price relative to locality average)
df['Value_Score'] = (df['Locality_Avg_Price'] - df['Price_Lakhs']) / df['Locality_Avg_Price'] * 100

# Prime locality indicator
top_localities = df.groupby('Locality')['Price_Lakhs'].mean().nlargest(10).index
df['Is_Prime_Locality'] = df['Locality'].isin(top_localities).astype(int)

# BHK to Bathroom ratio
df['BHK_Bath_Ratio'] = df['BHK'] / df['Bathroom']

# Age group (based on possession)
df['Furnishing_Status'] = df['Furnishing'].fillna('Unfurnished')
df['Seller_Type'] = df['Seller Type'].fillna('Agent')

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

le_locality = LabelEncoder()
df['Locality_Encoded'] = le_locality.fit_transform(df['Locality'].astype(str))

le_furnishing = LabelEncoder()
df['Furnishing_Encoded'] = le_furnishing.fit_transform(df['Furnishing_Status'])

le_seller = LabelEncoder()
df['Seller_Encoded'] = le_seller.fit_transform(df['Seller_Type'])

le_property = LabelEncoder()
df['Property_Type_Encoded'] = le_property.fit_transform(df['Property_Type'])

print(f"✅ Created 15+ new features")
print(f"Total columns: {len(df.columns)}")

# Save featured data
df.to_csv('featured_real_estate_data.csv', index=False)
print("\n✅ Featured data saved: featured_real_estate_data.csv")

# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("="*80)

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')

print("\n📊 Generating visualizations...")

# 1. Price Distribution
plt.figure(figsize=(10, 6))
plt.hist(df['Price_Lakhs'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Price (Lakhs)')
plt.ylabel('Frequency')
plt.title('Price Distribution - Ahmedabad Real Estate')
plt.savefig('01_price_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 01_price_distribution.png")

# 2. Area vs Price
plt.figure(figsize=(10, 6))
plt.scatter(df['Area_SqFt'], df['Price_Lakhs'], alpha=0.5)
plt.xlabel('Area (Sq.Ft)')
plt.ylabel('Price (Lakhs)')
plt.title('Area vs Price Relationship')
plt.savefig('02_area_vs_price.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 02_area_vs_price.png")

# 3. Top Localities
plt.figure(figsize=(12, 6))
top_10_localities = df.groupby('Locality')['Price_Lakhs'].mean().nlargest(10)
top_10_localities.plot(kind='barh')
plt.xlabel('Average Price (Lakhs)')
plt.title('Top 10 Localities by Average Price')
plt.tight_layout()
plt.savefig('03_top_localities.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 03_top_localities.png")

# 4. BHK Distribution
plt.figure(figsize=(10, 6))
df['BHK'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('BHK')
plt.ylabel('Count')
plt.title('BHK Distribution')
plt.savefig('04_bhk_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 04_bhk_distribution.png")

# 5. Furnishing Impact
plt.figure(figsize=(10, 6))
df.groupby('Furnishing_Status')['Price_Lakhs'].mean().plot(kind='bar')
plt.xlabel('Furnishing Status')
plt.ylabel('Average Price (Lakhs)')
plt.title('Impact of Furnishing on Price')
plt.xticks(rotation=45)
plt.savefig('05_furnishing_impact.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 05_furnishing_impact.png")

print(f"\n✅ Generated 5 key visualizations")

# Additional EDA visualizations
print("\n📊 Creating additional EDA visualizations...")

# 6. Price per sqft by locality (top 10)
plt.figure(figsize=(12, 6))
top_10_price_sqft = df.groupby('Locality')['Price_Per_SqFt'].mean().nlargest(10)
top_10_price_sqft.plot(kind='barh', color='purple', alpha=0.7)
plt.xlabel('Average Price per Sq.Ft (₹)')
plt.title('Top 10 Localities by Price per Sq.Ft')
plt.tight_layout()
plt.savefig('06_price_per_sqft_localities.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 06_price_per_sqft_localities.png")

# 7. Correlation heatmap
plt.figure(figsize=(12, 10))
numeric_cols = ['Price_Lakhs', 'Area_SqFt', 'BHK', 'Bathroom', 'Price_Per_SqFt', 
                'Locality_Avg_Price', 'Value_Score', 'BHK_Bath_Ratio']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('07_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 07_correlation_heatmap.png")

print(f"\n✅ Generated 7 total EDA visualizations")

# ============================================================================
# STEP 4: MACHINE LEARNING MODELS
# ============================================================================
print("\n" + "="*80)
print("STEP 4: MACHINE LEARNING MODELS")
print("="*80)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print("\n🤖 Preparing features for modeling...")

# Select features for modeling
feature_columns = [
    'Area_SqFt', 'BHK', 'Bathroom', 'Price_Per_SqFt',
    'Locality_Encoded', 'Furnishing_Encoded', 'Seller_Encoded',
    'Property_Type_Encoded', 'Locality_Avg_Price', 'Value_Score',
    'Is_Prime_Locality', 'BHK_Bath_Ratio'
]

X = df[feature_columns].copy()
y = df['Price_Lakhs'].copy()

# Handle any remaining NaN values
X = X.fillna(X.median())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples: {len(X_test):,}")

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, min_samples_split=20, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
}

results = []
print("\n🔄 Training and evaluating models...")

for name, model in models.items():
    print(f"\n   Training {name}...")
    
    # Train
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Evaluate
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    results.append({
        'Model': name,
        'R² Score': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    })
    
    print(f"      R² Score: {r2:.4f} ({r2*100:.2f}%)")
    print(f"      RMSE: ₹{rmse:.2f} Lakhs")
    print(f"      MAE: ₹{mae:.2f} Lakhs")
    print(f"      MAPE: {mape:.2f}%")

# Save results
results_df = pd.DataFrame(results).sort_values('R² Score', ascending=False)
results_df.to_csv('model_comparison_results.csv', index=False)
print("\n✅ Model comparison saved: model_comparison_results.csv")

# Create model comparison visualizations
print("\n📊 Creating model comparison visualizations...")

# 1. Model Performance Comparison - Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# R² Score comparison
axes[0, 0].barh(results_df['Model'], results_df['R² Score'], color='steelblue')
axes[0, 0].set_xlabel('R² Score')
axes[0, 0].set_title('Model Accuracy (R² Score)')
axes[0, 0].set_xlim(0, 1)
for i, v in enumerate(results_df['R² Score']):
    axes[0, 0].text(v + 0.01, i, f'{v:.4f}', va='center')

# MAE comparison
axes[0, 1].barh(results_df['Model'], results_df['MAE'], color='coral')
axes[0, 1].set_xlabel('MAE (Lakhs)')
axes[0, 1].set_title('Mean Absolute Error (Lower is Better)')
for i, v in enumerate(results_df['MAE']):
    axes[0, 1].text(v + 0.2, i, f'₹{v:.2f}L', va='center')

# RMSE comparison
axes[1, 0].barh(results_df['Model'], results_df['RMSE'], color='lightcoral')
axes[1, 0].set_xlabel('RMSE (Lakhs)')
axes[1, 0].set_title('Root Mean Square Error (Lower is Better)')
for i, v in enumerate(results_df['RMSE']):
    axes[1, 0].text(v + 0.3, i, f'₹{v:.2f}L', va='center')

# MAPE comparison
axes[1, 1].barh(results_df['Model'], results_df['MAPE'], color='lightgreen')
axes[1, 1].set_xlabel('MAPE (%)')
axes[1, 1].set_title('Mean Absolute Percentage Error (Lower is Better)')
for i, v in enumerate(results_df['MAPE']):
    axes[1, 1].text(v + 0.3, i, f'{v:.2f}%', va='center')

plt.tight_layout()
plt.savefig('06_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 06_model_comparison.png")

# Save best model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

model_filename = f"best_model_{best_model_name.replace(' ', '')}.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)

with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

model_info = {
    'best_model_name': best_model_name,
    'test_r2_score': results_df.iloc[0]['R² Score'],
    'test_rmse': results_df.iloc[0]['RMSE'],
    'test_mae': results_df.iloc[0]['MAE'],
    'test_mape': results_df.iloc[0]['MAPE']
}
with open('model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print(f"\n✅ Best model saved: {model_filename}")
print(f"   Model: {best_model_name}")
print(f"   Accuracy: {model_info['test_r2_score']*100:.2f}%")

# Create best model visualizations
print("\n📊 Creating best model performance visualizations...")

# Get predictions for best model
if best_model_name == 'Linear Regression':
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

# 2. Actual vs Predicted scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5, s=30)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price (Lakhs)')
plt.ylabel('Predicted Price (Lakhs)')
plt.title(f'{best_model_name}: Actual vs Predicted Prices')
plt.legend()
plt.grid(True, alpha=0.3)
# Add R² annotation
plt.text(0.05, 0.95, f'R² = {model_info["test_r2_score"]:.4f}\nMAE = ₹{model_info["test_mae"]:.2f}L', 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.savefig('07_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 07_actual_vs_predicted.png")

# 3. Residual plot
residuals = y_test - y_pred_best
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_best, residuals, alpha=0.5, s=30)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price (Lakhs)')
plt.ylabel('Residuals (Lakhs)')
plt.title(f'{best_model_name}: Residual Plot (Error Distribution)')
plt.grid(True, alpha=0.3)
plt.savefig('08_residual_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 08_residual_plot.png")

# 4. Residual distribution histogram
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
plt.xlabel('Residuals (Lakhs)')
plt.ylabel('Frequency')
plt.title(f'{best_model_name}: Residual Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
# Add statistics
plt.text(0.02, 0.98, f'Mean: ₹{residuals.mean():.2f}L\nStd Dev: ₹{residuals.std():.2f}L', 
         transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
plt.savefig('09_residual_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 09_residual_distribution.png")

# 5. Prediction error percentage
error_pct = np.abs((y_test - y_pred_best) / y_test) * 100
plt.figure(figsize=(10, 6))
plt.hist(error_pct, bins=50, edgecolor='black', alpha=0.7, color='orange')
plt.axvline(x=error_pct.mean(), color='r', linestyle='--', lw=2, label=f'Mean: {error_pct.mean():.2f}%')
plt.xlabel('Prediction Error (%)')
plt.ylabel('Frequency')
plt.title(f'{best_model_name}: Prediction Error Percentage Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('10_error_percentage.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 10_error_percentage.png")

# 6. Feature Importance (for tree-based models)
if best_model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'{best_model_name}: Feature Importance Analysis')
    plt.tight_layout()
    plt.savefig('11_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: 11_feature_importance.png")
    
    # Save feature importance to CSV
    feature_importance.to_csv('feature_importance.csv', index=False)
    print("   ✓ Saved: feature_importance.csv")

# 7. Error by price range
price_ranges = pd.cut(y_test, bins=[0, 50, 100, 200, 1000], labels=['<50L', '50-100L', '100-200L', '>200L'])
error_by_range = pd.DataFrame({
    'Price_Range': price_ranges,
    'Absolute_Error': np.abs(residuals)
})
error_summary = error_by_range.groupby('Price_Range')['Absolute_Error'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(error_summary['Price_Range'], error_summary['Absolute_Error'], color='teal', alpha=0.7)
plt.xlabel('Price Range')
plt.ylabel('Average Absolute Error (Lakhs)')
plt.title(f'{best_model_name}: Average Prediction Error by Price Range')
plt.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(error_summary['Absolute_Error']):
    plt.text(i, v + 0.5, f'₹{v:.2f}L', ha='center', va='bottom')
plt.savefig('12_error_by_price_range.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 12_error_by_price_range.png")

print(f"\n✅ Generated 7 additional model visualizations")

# ============================================================================
# STEP 5: BUSINESS INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("STEP 5: BUSINESS INSIGHTS & USE CASES")
print("="*80)

print("\n💡 Generating business insights...")

# Use Case 1: Affordable Housing Zones
print("\n📊 Use Case 1: Affordable Housing Development Zones")
affordable = df[df['BHK'].isin([1, 2])].copy()
affordable_localities = affordable.groupby('Locality').agg({
    'Price_Lakhs': 'mean',
    'Property Title': 'count'
}).rename(columns={'Property Title': 'Supply'}).sort_values('Supply', ascending=False).head(5)
print(f"   Top 5 affordable housing zones identified")
print(affordable_localities)

# Use Case 2: Undervalued Properties
print("\n📊 Use Case 2: Undervalued Properties")
if best_model_name == 'Linear Regression':
    df['Predicted_Price'] = best_model.predict(scaler.transform(df[feature_columns].fillna(df[feature_columns].median())))
else:
    df['Predicted_Price'] = best_model.predict(df[feature_columns].fillna(df[feature_columns].median()))

df['Value_Opportunity'] = ((df['Predicted_Price'] - df['Price_Lakhs']) / df['Price_Lakhs']) * 100
undervalued = df[df['Value_Opportunity'] > 10].sort_values('Value_Opportunity', ascending=False)
print(f"   Found {len(undervalued)} undervalued properties")
print(f"   Average opportunity: {undervalued['Value_Opportunity'].mean():.1f}%")

# Use Case 3: Premium Investment Zones
print("\n📊 Use Case 3: Premium Investment Zones")
premium = df[df['Price_Lakhs'] > df['Price_Lakhs'].quantile(0.75)]
premium_localities = premium.groupby('Locality').agg({
    'Price_Lakhs': ['mean', 'count']
}).sort_values(('Price_Lakhs', 'count'), ascending=False).head(5)
print(f"   Top 5 premium investment zones identified")

# Use Case 4: Price Prediction Statistics
print("\n📊 Use Case 4: Market Statistics")
print(f"   Total properties analyzed: {len(df):,}")
print(f"   Average price: ₹{df['Price_Lakhs'].mean():.2f} Lakhs")
print(f"   Median price: ₹{df['Price_Lakhs'].median():.2f} Lakhs")
print(f"   Average area: {df['Area_SqFt'].mean():.0f} sq.ft")
print(f"   Most common BHK: {df['BHK'].mode()[0]}")
print(f"   Unique localities: {df['Locality'].nunique()}")

# Save final dataset with predictions
df.to_csv('final_analysis_with_predictions.csv', index=False)
print("\n✅ Final analysis saved: final_analysis_with_predictions.csv")

# ============================================================================
# FINAL REPORT SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL EXECUTION REPORT")
print("="*80)

print(f"\n📊 DATA PROCESSING:")
print(f"   Raw records: 2,991")
print(f"   Cleaned records: {len(df):,}")
print(f"   Features engineered: 15+")
print(f"   EDA visualizations: 7")
print(f"   Model visualizations: 7")
print(f"   Total visualizations: 14")

print(f"\n🤖 MODEL PERFORMANCE:")
for idx, row in results_df.iterrows():
    print(f"   {row['Model']}:")
    print(f"      R² Score: {row['R² Score']:.4f} ({row['R² Score']*100:.2f}%)")
    print(f"      MAE: ₹{row['MAE']:.2f} Lakhs")

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {model_info['test_r2_score']*100:.2f}%")
print(f"   Average Error: ±₹{model_info['test_mae']:.2f} Lakhs")

print(f"\n💼 BUSINESS INSIGHTS:")
print(f"   ✓ Affordable housing zones: 5 identified")
print(f"   ✓ Undervalued properties: {len(undervalued):,} found")
print(f"   ✓ Premium investment zones: 5 identified")
print(f"   ✓ Market predictions: Generated for all properties")

print(f"\n📁 GENERATED FILES:")
files = [
    'cleaned_real_estate_data.csv',
    'featured_real_estate_data.csv',
    'model_comparison_results.csv',
    'final_analysis_with_predictions.csv',
    model_filename,
    'feature_scaler.pkl',
    'feature_columns.pkl',
    'model_info.pkl'
]
for i, file in enumerate(files, 1):
    print(f"   {i}. {file}")

print(f"\n" + "="*80)
print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
