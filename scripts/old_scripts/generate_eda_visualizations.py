"""
Generate EDA Visualizations with Clean Features (No Data Leakage)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENERATING EDA VISUALIZATIONS (CLEAN DATA)")
print("="*80)

# Load clean featured data
df = pd.read_csv('../data/processed/featured_real_estate_data.csv')
print(f"\n[INFO] Dataset loaded: {df.shape}")
print(f"[INFO] Features: {list(df.columns)}")

# Create visualizations directory if needed
import os
viz_dir = '../visualizations/eda'
os.makedirs(viz_dir, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')

print(f"\n[INFO] Generating visualizations...")

# 1. Price Distribution
print("  [1/10] Price distribution...")
plt.figure(figsize=(12, 6))
plt.hist(df['Price_Lakhs'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('Price (Lakhs)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Price Distribution - Ahmedabad Real Estate', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig(f'{viz_dir}/01_price_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Area vs Price
print("  [2/10] Area vs Price scatter...")
plt.figure(figsize=(12, 6))
plt.scatter(df['Area_SqFt'], df['Price_Lakhs'], alpha=0.5, s=30, c='coral')
plt.xlabel('Area (Sq.Ft)', fontsize=12)
plt.ylabel('Price (Lakhs)', fontsize=12)
plt.title('Area vs Price Relationship', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig(f'{viz_dir}/02_area_vs_price.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Top Localities by Average Price
print("  [3/10] Top localities...")
plt.figure(figsize=(14, 8))
top_10_localities = df.groupby('Locality')['Price_Lakhs'].mean().nlargest(10)
top_10_localities.plot(kind='barh', color='purple', alpha=0.7)
plt.xlabel('Average Price (Lakhs)', fontsize=12)
plt.ylabel('Locality', fontsize=12)
plt.title('Top 10 Localities by Average Price', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{viz_dir}/03_top_localities.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. BHK Distribution
print("  [4/10] BHK distribution...")
plt.figure(figsize=(10, 6))
bhk_counts = df['BHK'].value_counts().sort_index()
plt.bar(bhk_counts.index, bhk_counts.values, color='teal', alpha=0.7, edgecolor='black')
plt.xlabel('BHK', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('BHK Distribution', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.savefig(f'{viz_dir}/04_bhk_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Furnishing Impact
print("  [5/10] Furnishing impact...")
plt.figure(figsize=(10, 6))
furn_avg = df.groupby('Furnishing')['Price_Lakhs'].mean()
plt.bar(furn_avg.index, furn_avg.values, color='orange', alpha=0.7, edgecolor='black')
plt.xlabel('Furnishing Status', fontsize=12)
plt.ylabel('Average Price (Lakhs)', fontsize=12)
plt.title('Impact of Furnishing on Price', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{viz_dir}/05_furnishing_impact.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Price per sqft by locality (top 10)
print("  [6/10] Price per sqft by locality...")
plt.figure(figsize=(14, 8))
top_10_price_sqft = df.groupby('Locality')['Price_Per_SqFt'].mean().nlargest(10)
plt.barh(range(len(top_10_price_sqft)), top_10_price_sqft.values, color='lightcoral', alpha=0.7)
plt.yticks(range(len(top_10_price_sqft)), top_10_price_sqft.index)
plt.xlabel('Average Price per Sq.Ft (Rs.)', fontsize=12)
plt.ylabel('Locality', fontsize=12)
plt.title('Top 10 Localities by Price per Sq.Ft', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{viz_dir}/06_price_per_sqft_localities.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Correlation heatmap (CLEAN FEATURES ONLY)
print("  [7/10] Correlation heatmap...")
plt.figure(figsize=(10, 8))
numeric_cols = ['Price_Lakhs', 'Area_SqFt', 'BHK', 'Furnishing_Encoded', 'Locality_Encoded']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap (4 Main Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{viz_dir}/07_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. BHK vs Price boxplot
print("  [8/10] BHK vs Price boxplot...")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='BHK', y='Price_Lakhs', palette='Set2')
plt.xlabel('BHK', fontsize=12)
plt.ylabel('Price (Lakhs)', fontsize=12)
plt.title('Price Distribution by BHK', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.savefig(f'{viz_dir}/08_bhk_price_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. Property Type Distribution
print("  [9/10] Property type distribution...")
plt.figure(figsize=(10, 6))
type_counts = df['Property_Type'].value_counts()
plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', 
        colors=['lightblue', 'lightgreen', 'lightcoral'], startangle=90)
plt.title('Property Type Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{viz_dir}/09_property_type_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 10. Area Category Distribution
print("  [10/10] Area category distribution...")
plt.figure(figsize=(10, 6))
cat_counts = df['Area_Category'].value_counts()
plt.bar(cat_counts.index, cat_counts.values, color='skyblue', alpha=0.7, edgecolor='black')
plt.xlabel('Area Category', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Area Category Distribution', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{viz_dir}/10_area_category_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[OK] All 10 EDA visualizations generated successfully!")
print(f"[OK] Saved to: {viz_dir}/")
print("="*80)
