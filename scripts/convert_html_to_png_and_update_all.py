"""
Convert HTML visualizations to PNG and regenerate all visualizations with updated data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

# Paths
DATA_PATH = Path('../data/processed/featured_real_estate_data.csv')
MODEL_DIR = Path('../models')
VIZ_DIR = Path('../visualizations')

# Load data
print("Loading data and models...")
df = pd.read_csv(DATA_PATH)

# Load best model
with open(MODEL_DIR / 'best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)
with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open(MODEL_DIR / 'feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Load comparison results for accurate metrics
comparison_df = pd.read_csv('../data/processed/model_comparison_results.csv')
best_result = comparison_df.iloc[0]
r2 = best_result['Test_R2']  # Use test set R2, not full dataset
mae = best_result['MAE_Lakhs']
model_name = best_result['Model']

# For visualizations, use full dataset predictions
X = df[feature_columns]
y = df['Price_Lakhs']
X_scaled = scaler.transform(X)
y_pred = best_model.predict(X_scaled)

print(f"Model: {model_name} - {r2*100:.2f}% accurate (Test Set)")
print(f"Dataset: {len(df)} properties")

# ============================================================================
# PART 1: Convert HTML to PNG (Advanced Visualizations)
# ============================================================================
print("\n" + "="*70)
print("PART 1: Converting HTML Advanced Visualizations to PNG")
print("="*70)

advanced_dir = VIZ_DIR / 'advanced'

# Since we can't directly convert HTML to PNG without browser automation,
# let's recreate these visualizations as static PNG using matplotlib/seaborn

print("\n[1/10] Creating price distribution...")
fig, ax = plt.subplots(figsize=(14, 8))
ax.hist(df['Price_Lakhs'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
ax.axvline(df['Price_Lakhs'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ₹{df["Price_Lakhs"].mean():.1f}L')
ax.axvline(df['Price_Lakhs'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ₹{df["Price_Lakhs"].median():.1f}L')
ax.set_xlabel('Price (₹ Lakhs)', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax.set_title('Real Estate Price Distribution\nAhmedabad Market Analysis', fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(advanced_dir / '01_price_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 01_price_distribution.png")

print("[2/10] Creating 3D-style scatter (Area vs Price vs BHK)...")
fig, ax = plt.subplots(figsize=(14, 10))
scatter = ax.scatter(df['Area_SqFt'], df['Price_Lakhs'], c=df['BHK'], 
                    s=100, alpha=0.6, cmap='viridis', edgecolor='black', linewidth=0.5)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('BHK', fontsize=12, fontweight='bold')
ax.set_xlabel('Area (Sq.Ft)', fontsize=14, fontweight='bold')
ax.set_ylabel('Price (₹ Lakhs)', fontsize=14, fontweight='bold')
ax.set_title('Property Size vs Price (Colored by BHK)\n3D Relationship Visualization', fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(advanced_dir / '02_area_price_bhk_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 02_area_price_bhk_scatter.png")

print("[3/10] Creating locality price heatmap...")
# Map tier names if they're already strings
if df['Locality_Tier'].dtype == 'object':
    tier_name_to_num = {'Budget': 0, 'Mid-Range': 1, 'High-End': 2, 'Premium': 3}
    df['Locality_Tier_Num'] = df['Locality_Tier'].map(tier_name_to_num).fillna(df['Locality_Tier'])
    group_col = 'Locality_Tier'
else:
    group_col = 'Locality_Tier'

locality_prices = df.groupby(group_col)['Price_Lakhs'].agg(['mean', 'median', 'count']).reset_index()
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(locality_prices))
width = 0.35
bars1 = ax.bar(x - width/2, locality_prices['mean'], width, label='Mean Price', color='#3498db', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, locality_prices['median'], width, label='Median Price', color='#e74c3c', alpha=0.8, edgecolor='black')
ax.set_xlabel('Locality Tier', fontsize=14, fontweight='bold')
ax.set_ylabel('Price (₹ Lakhs)', fontsize=14, fontweight='bold')
ax.set_title('Price Analysis by Locality Tier', fontsize=18, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(locality_prices[group_col])
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'₹{height:.0f}L', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(advanced_dir / '03_locality_tier_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 03_locality_tier_analysis.png")

print("[4/10] Creating BHK price hierarchy...")
bhk_stats = df.groupby('BHK')['Price_Lakhs'].agg(['mean', 'median', 'min', 'max', 'count']).reset_index()
# Filter out any non-numeric or extreme BHK values
bhk_stats = bhk_stats[(bhk_stats['BHK'] >= 1) & (bhk_stats['BHK'] <= 10)]
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Mean prices
ax = axes[0]
bars = ax.barh(bhk_stats['BHK'].astype(str), bhk_stats['mean'], color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
for i, (bar, val, cnt) in enumerate(zip(bars, bhk_stats['mean'], bhk_stats['count'])):
    ax.text(val + 2, bar.get_y() + bar.get_height()/2, f'₹{val:.1f}L\n({cnt} props)', 
            va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Average Price (₹ Lakhs)', fontsize=13, fontweight='bold')
ax.set_ylabel('BHK', fontsize=13, fontweight='bold')
ax.set_title('Average Price by Property Size', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Price range
ax = axes[1]
for i, row in bhk_stats.iterrows():
    ax.barh(i, row['max'] - row['min'], left=row['min'], color='#9b59b6', alpha=0.6, edgecolor='black', linewidth=1.5)
    ax.plot(row['mean'], i, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=1.5, label='Mean' if i == 0 else '')
ax.set_yticks(range(len(bhk_stats)))
ax.set_yticklabels(bhk_stats['BHK'].astype(str))
ax.set_xlabel('Price Range (₹ Lakhs)', fontsize=13, fontweight='bold')
ax.set_ylabel('BHK', fontsize=13, fontweight='bold')
ax.set_title('Price Range by Property Size', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.suptitle('BHK Property Analysis - Price Hierarchy', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(advanced_dir / '04_bhk_hierarchy.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 04_bhk_hierarchy.png")

print("[5/10] Creating feature correlation parallel coordinates...")
features_to_plot = ['Area_SqFt', 'BHK', 'Price_Lakhs']  # Only numeric features
fig, ax = plt.subplots(figsize=(14, 8))
sample_df = df[features_to_plot].sample(min(500, len(df)))
# Ensure all columns are numeric
sample_df = sample_df.apply(pd.to_numeric, errors='coerce').dropna()
normalized = (sample_df - sample_df.min()) / (sample_df.max() - sample_df.min())
for i in range(len(normalized)):
    ax.plot(range(len(features_to_plot)), normalized.iloc[i], alpha=0.1, color='blue')
ax.set_xticks(range(len(features_to_plot)))
ax.set_xticklabels(features_to_plot, fontsize=12, fontweight='bold')
ax.set_ylabel('Normalized Value', fontsize=13, fontweight='bold')
ax.set_title('Feature Relationships - Parallel Coordinates\nProperty Characteristics Patterns', fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(advanced_dir / '05_parallel_coordinates.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 05_parallel_coordinates.png")

print("[6/10] Creating BHK evolution analysis...")
bhk_evolution = df.groupby('BHK').agg({
    'Price_Lakhs': ['mean', 'count'],
    'Area_SqFt': 'mean'
}).reset_index()
bhk_evolution.columns = ['BHK', 'Avg_Price', 'Count', 'Avg_Area']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Price evolution
ax = axes[0]
ax.plot(bhk_evolution['BHK'], bhk_evolution['Avg_Price'], 'o-', linewidth=3, markersize=10, color='#3498db', markeredgecolor='black', markeredgewidth=2)
ax.set_xlabel('BHK', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Price (₹L)', fontsize=12, fontweight='bold')
ax.set_title('Price Evolution by BHK', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Count
ax = axes[1]
bars = ax.bar(bhk_evolution['BHK'], bhk_evolution['Count'], color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
for bar, val in zip(bars, bhk_evolution['Count']):
    ax.text(bar.get_x() + bar.get_width()/2, val + 10, str(int(val)), ha='center', fontsize=11, fontweight='bold')
ax.set_xlabel('BHK', fontsize=12, fontweight='bold')
ax.set_ylabel('Property Count', fontsize=12, fontweight='bold')
ax.set_title('Market Supply by BHK', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Area evolution
ax = axes[2]
ax.plot(bhk_evolution['BHK'], bhk_evolution['Avg_Area'], 's-', linewidth=3, markersize=10, color='#e74c3c', markeredgecolor='black', markeredgewidth=2)
ax.set_xlabel('BHK', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Area (Sq.Ft)', fontsize=12, fontweight='bold')
ax.set_title('Size Evolution by BHK', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle('BHK Market Evolution Analysis', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(advanced_dir / '06_bhk_evolution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 06_bhk_evolution.png")

print("[7/10] Creating comprehensive boxplots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Create tier name mapping if not exists
tier_map = {0: 'Budget', 1: 'Mid', 2: 'High', 3: 'Premium'}
if 'Tier_Name' not in df.columns:
    if df['Locality_Tier'].dtype in ['int64', 'float64']:
        df['Tier_Name'] = df['Locality_Tier'].map(tier_map)
    else:
        df['Tier_Name'] = df['Locality_Tier']

# Price by BHK
ax = axes[0, 0]
df_bhk = df[df['BHK'].between(1, 10)]  # Filter valid BHK values
df_bhk.boxplot(column='Price_Lakhs', by='BHK', ax=ax, patch_artist=True)
ax.set_xlabel('BHK', fontsize=12, fontweight='bold')
ax.set_ylabel('Price (₹ Lakhs)', fontsize=12, fontweight='bold')
ax.set_title('Price Distribution by BHK', fontsize=14, fontweight='bold')
plt.sca(ax)
plt.xticks(rotation=0)

# Price by Tier
ax = axes[0, 1]
df.boxplot(column='Price_Lakhs', by='Tier_Name', ax=ax, patch_artist=True)
ax.set_xlabel('Locality Tier', fontsize=12, fontweight='bold')
ax.set_ylabel('Price (₹ Lakhs)', fontsize=12, fontweight='bold')
ax.set_title('Price Distribution by Tier', fontsize=14, fontweight='bold')

# Area by BHK
ax = axes[1, 0]
df_bhk.boxplot(column='Area_SqFt', by='BHK', ax=ax, patch_artist=True)
ax.set_xlabel('BHK', fontsize=12, fontweight='bold')
ax.set_ylabel('Area (Sq.Ft)', fontsize=12, fontweight='bold')
ax.set_title('Area Distribution by BHK', fontsize=14, fontweight='bold')

# Price per sqft by Tier
ax = axes[1, 1]
df['Price_Per_SqFt'] = df['Price_Lakhs'] * 100000 / df['Area_SqFt']
df.boxplot(column='Price_Per_SqFt', by='Tier_Name', ax=ax, patch_artist=True)
ax.set_xlabel('Locality Tier', fontsize=12, fontweight='bold')
ax.set_ylabel('Price per Sq.Ft (₹)', fontsize=12, fontweight='bold')
ax.set_title('Price/SqFt by Tier', fontsize=14, fontweight='bold')

plt.suptitle('Comprehensive Distribution Analysis', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(advanced_dir / '07_comprehensive_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 07_comprehensive_boxplots.png")

print("[8/10] Creating feature correlation matrix...")
corr_features = ['Area_SqFt', 'BHK', 'Locality_Tier_Encoded', 'Furnishing_Encoded', 'BHK_Tier_Interaction', 'Price_Lakhs']
corr_matrix = df[corr_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_features)))
ax.set_yticks(range(len(corr_features)))
feature_names = ['Area', 'BHK', 'Tier', 'Furnishing', 'BHK×Tier', 'Price']
ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=12, fontweight='bold')
ax.set_yticklabels(feature_names, fontsize=12, fontweight='bold')

# Add correlation values
for i in range(len(corr_features)):
    for j in range(len(corr_features)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=11, fontweight='bold')

plt.colorbar(im, ax=ax, label='Correlation Coefficient')
ax.set_title('Feature Correlation Matrix\nInteractive Relationships Analysis', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(advanced_dir / '08_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 08_correlation_matrix.png")

print("[9/10] Creating market composition treemap...")
market_comp = df.groupby(['Locality_Tier', 'BHK']).size().reset_index(name='Count')
market_comp['Tier_Name'] = market_comp['Locality_Tier'].map(tier_map)

fig, ax = plt.subplots(figsize=(14, 10))
tier_colors = {'Budget': '#95a5a6', 'Mid': '#f39c12', 'High': '#3498db', 'Premium': '#2ecc71'}

y_pos = 0
for tier in ['Budget', 'Mid', 'High', 'Premium']:
    tier_data = market_comp[market_comp['Tier_Name'] == tier]
    tier_total = tier_data['Count'].sum()
    
    x_pos = 0
    for _, row in tier_data.iterrows():
        width = row['Count'] / market_comp['Count'].sum()
        height = tier_total / market_comp['Count'].sum()
        
        rect = plt.Rectangle((x_pos, y_pos), width, height, 
                            facecolor=tier_colors[tier], alpha=0.7, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        ax.text(x_pos + width/2, y_pos + height/2, 
               f"{tier}\n{int(row['BHK'])} BHK\n{row['Count']} props",
               ha='center', va='center', fontsize=9, fontweight='bold')
        
        x_pos += width
    
    y_pos += height

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Market Composition Treemap\nProperty Distribution by Tier and BHK', 
            fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(advanced_dir / '09_market_treemap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 09_market_treemap.png")

print("[10/10] Creating comprehensive dashboard...")
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Overall stats
ax = fig.add_subplot(gs[0, :])
ax.axis('off')
stats_text = f"""
AHMEDABAD REAL ESTATE MARKET - COMPREHENSIVE DASHBOARD
{'='*100}

Dataset: {len(df):,} Properties | Best Model: Gradient Boosting | Accuracy: {r2*100:.2f}% | MAE: ±₹{mae:.2f}L

Price Range: ₹{df['Price_Lakhs'].min():.1f}L - ₹{df['Price_Lakhs'].max():.1f}L | Average: ₹{df['Price_Lakhs'].mean():.1f}L | Median: ₹{df['Price_Lakhs'].median():.1f}L
"""
ax.text(0.5, 0.5, stats_text, transform=ax.transAxes, fontsize=12, fontfamily='monospace',
       ha='center', va='center', bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2))

# Small visualizations
positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
titles = ['Price Distribution', 'BHK Distribution', 'Tier Distribution', 
         'Area Distribution', 'Price by BHK', 'Price by Tier']

for pos, title in zip(positions, titles):
    ax = fig.add_subplot(gs[pos[0], pos[1]])
    
    if title == 'Price Distribution':
        ax.hist(df['Price_Lakhs'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    elif title == 'BHK Distribution':
        bhk_counts = df['BHK'].value_counts().sort_index()
        ax.bar(bhk_counts.index, bhk_counts.values, color='#2ecc71', alpha=0.7, edgecolor='black')
    elif title == 'Tier Distribution':
        tier_counts = df['Tier_Name'].value_counts()
        ax.bar(range(len(tier_counts)), tier_counts.values, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(tier_counts)))
        ax.set_xticklabels(tier_counts.index, rotation=45)
    elif title == 'Area Distribution':
        ax.hist(df['Area_SqFt'], bins=30, color='#f39c12', alpha=0.7, edgecolor='black')
    elif title == 'Price by BHK':
        df.boxplot(column='Price_Lakhs', by='BHK', ax=ax)
        plt.sca(ax)
        plt.xticks(rotation=0)
    elif title == 'Price by Tier':
        df.boxplot(column='Price_Lakhs', by='Tier_Name', ax=ax)
        plt.sca(ax)
        plt.xticks(rotation=45)
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.savefig(advanced_dir / '10_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 10_comprehensive_dashboard.png")

print("\n" + "="*70)
print("✅ ALL ADVANCED VISUALIZATIONS CONVERTED TO PNG!")
print("="*70)
print(f"\n10 PNG files created in: {advanced_dir}")
print("\nAll visualizations are now in PNG format and updated with latest data!")
print(f"Best Model: Gradient Boosting - {r2*100:.2f}% Accurate")
print("="*70)
