"""
Comprehensive EDA Visualizations for Ahmedabad Real Estate
Organized by Business-Relevant Categories

A. PRICE & DISTRIBUTION INSIGHTS (4 visualizations)
B. LOCATION-BASED INSIGHTS (5 visualizations) 
C. PROPERTY FEATURES & COMPARISONS (5 visualizations)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

print("="*100)
print("COMPREHENSIVE EDA VISUALIZATIONS - AHMEDABAD REAL ESTATE ANALYSIS")
print("="*100)

# Load featured data
df = pd.read_csv('../data/processed/featured_real_estate_data.csv')
print(f"\n[INFO] Dataset loaded: {df.shape}")
print(f"[INFO] Total Properties: {len(df):,}")
print(f"[INFO] Price Range: ₹{df['Price_Lakhs'].min():.2f}L - ₹{df['Price_Lakhs'].max():.2f}L")
print(f"[INFO] Average Price: ₹{df['Price_Lakhs'].mean():.2f} Lakhs")

# Create visualizations directory
viz_dir = '../visualizations/eda'
os.makedirs(viz_dir, exist_ok=True)

# Set styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("\n" + "="*100)
print("A. PRICE & DISTRIBUTION INSIGHTS")
print("="*100)

# ============================================================================
# 1. PRICE DISTRIBUTION HISTOGRAM
# ============================================================================
print("\n[1/14] 📊 Price Distribution Histogram - Shows how property prices vary")

fig, ax = plt.subplots(figsize=(14, 7))
n, bins, patches = ax.hist(df['Price_Lakhs'], bins=60, edgecolor='black', alpha=0.7, color='#3498db')

# Add mean and median lines
mean_price = df['Price_Lakhs'].mean()
median_price = df['Price_Lakhs'].median()
ax.axvline(mean_price, color='red', linestyle='--', linewidth=2.5, label=f'Mean: ₹{mean_price:.2f}L')
ax.axvline(median_price, color='green', linestyle='--', linewidth=2.5, label=f'Median: ₹{median_price:.2f}L')

ax.set_xlabel('Price (Lakhs)', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency (Number of Properties)', fontsize=14, fontweight='bold')
ax.set_title('Price Distribution - Ahmedabad Real Estate Market\nHelps identify pricing trends and market segments', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)

# Add statistics box
stats_text = f'Total Properties: {len(df):,}\nStd Dev: ₹{df["Price_Lakhs"].std():.2f}L\nMin: ₹{df["Price_Lakhs"].min():.2f}L\nMax: ₹{df["Price_Lakhs"].max():.2f}L'
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{viz_dir}/01_price_distribution_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 01_price_distribution_histogram.png")

# ============================================================================
# 2. AREA DISTRIBUTION HISTOGRAM
# ============================================================================
print("\n[2/14] 📐 Area Distribution Histogram - Visualizes small vs medium vs luxury-size homes")

fig, ax = plt.subplots(figsize=(14, 7))
colors = ['#52B788' if x < 1000 else '#74C69D' if x < 2000 else '#95D5B2' if x < 3000 else '#B7E4C7' 
          for x in df['Area_SqFt']]
n, bins, patches = ax.hist(df['Area_SqFt'], bins=60, edgecolor='black', alpha=0.7, color='#2ECC71')

# Color code by size category
for i, patch in enumerate(patches):
    if bins[i] < 1000:
        patch.set_facecolor('#52B788')  # Small
    elif bins[i] < 2000:
        patch.set_facecolor('#74C69D')  # Medium
    elif bins[i] < 3000:
        patch.set_facecolor('#95D5B2')  # Large
    else:
        patch.set_facecolor('#B7E4C7')  # Extra Large

mean_area = df['Area_SqFt'].mean()
median_area = df['Area_SqFt'].median()
ax.axvline(mean_area, color='red', linestyle='--', linewidth=2.5, label=f'Mean: {mean_area:.0f} sqft')
ax.axvline(median_area, color='blue', linestyle='--', linewidth=2.5, label=f'Median: {median_area:.0f} sqft')

ax.set_xlabel('Area (Square Feet)', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency (Number of Properties)', fontsize=14, fontweight='bold')
ax.set_title('Area Distribution by Property Size\nGreen: Small (<1000) | Teal: Medium (1000-2000) | Blue: Large (2000-3000) | Light: Luxury (>3000)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Add category counts
small = len(df[df['Area_SqFt'] < 1000])
medium = len(df[(df['Area_SqFt'] >= 1000) & (df['Area_SqFt'] < 2000)])
large = len(df[(df['Area_SqFt'] >= 2000) & (df['Area_SqFt'] < 3000)])
xlarge = len(df[df['Area_SqFt'] >= 3000])
stats_text = f'Small (<1000): {small}\nMedium (1000-2000): {medium}\nLarge (2000-3000): {large}\nLuxury (>3000): {xlarge}'
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{viz_dir}/02_area_distribution_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 02_area_distribution_histogram.png")

# ============================================================================
# 3. PRICE PER SQFT DISTRIBUTION
# ============================================================================
print("\n[3/14] 💎 Price per Sqft Distribution - Helps identify overpriced vs underpriced properties")

fig, ax = plt.subplots(figsize=(14, 7))
n, bins, patches = ax.hist(df['Price_Per_SqFt'], bins=60, edgecolor='black', alpha=0.7, color='#9B59B6')

mean_pps = df['Price_Per_SqFt'].mean()
median_pps = df['Price_Per_SqFt'].median()
ax.axvline(mean_pps, color='red', linestyle='--', linewidth=2.5, label=f'Mean: ₹{mean_pps:.0f}/sqft')
ax.axvline(median_pps, color='orange', linestyle='--', linewidth=2.5, label=f'Median: ₹{median_pps:.0f}/sqft')

# Add affordability zones
q25 = df['Price_Per_SqFt'].quantile(0.25)
q75 = df['Price_Per_SqFt'].quantile(0.75)
ax.axvspan(0, q25, alpha=0.2, color='green', label=f'Underpriced (<₹{q25:.0f})')
ax.axvspan(q75, df['Price_Per_SqFt'].max(), alpha=0.2, color='red', label=f'Overpriced (>₹{q75:.0f})')

ax.set_xlabel('Price per Sq.Ft (₹)', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency (Number of Properties)', fontsize=14, fontweight='bold')
ax.set_title('Price per Sq.Ft Distribution\nGreen zone: Best value | Red zone: Premium pricing', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

# Add value analysis
underpriced = len(df[df['Price_Per_SqFt'] < q25])
fair = len(df[(df['Price_Per_SqFt'] >= q25) & (df['Price_Per_SqFt'] <= q75)])
overpriced = len(df[df['Price_Per_SqFt'] > q75])
stats_text = f'Best Value: {underpriced} ({underpriced/len(df)*100:.1f}%)\nFair Priced: {fair} ({fair/len(df)*100:.1f}%)\nPremium: {overpriced} ({overpriced/len(df)*100:.1f}%)'
ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{viz_dir}/03_price_per_sqft_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 03_price_per_sqft_distribution.png")

# ============================================================================
# 4. LOG-SCALED PRICE DISTRIBUTION
# ============================================================================
print("\n[4/14] 📉 Log-scaled Price Distribution - Removes skewness due to extreme values")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Normal scale
ax1.hist(df['Price_Lakhs'], bins=50, edgecolor='black', alpha=0.7, color='#E74C3C')
ax1.set_xlabel('Price (Lakhs)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Normal Scale - Shows skewness', fontsize=14, fontweight='bold')
ax1.axvline(df['Price_Lakhs'].mean(), color='blue', linestyle='--', linewidth=2, label='Mean')
ax1.axvline(df['Price_Lakhs'].median(), color='green', linestyle='--', linewidth=2, label='Median')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Log scale
ax2.hist(np.log10(df['Price_Lakhs']), bins=50, edgecolor='black', alpha=0.7, color='#3498DB')
ax2.set_xlabel('Price (Log₁₀ of Lakhs)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('Log Scale - Normalized distribution', fontsize=14, fontweight='bold')
ax2.axvline(np.log10(df['Price_Lakhs'].mean()), color='blue', linestyle='--', linewidth=2, label='Mean')
ax2.axvline(np.log10(df['Price_Lakhs'].median()), color='green', linestyle='--', linewidth=2, label='Median')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add log scale labels
log_ticks = [10, 50, 100, 200, 500]
ax2.set_xticks([np.log10(x) for x in log_ticks])
ax2.set_xticklabels([f'₹{x}L' for x in log_ticks])

fig.suptitle('Price Distribution: Normal vs Log Scale\nLog scale reveals true price distribution patterns', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(f'{viz_dir}/04_log_price_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 04_log_price_distribution.png")

print("\n" + "="*100)
print("B. LOCATION-BASED INSIGHTS")
print("="*100)

# ============================================================================
# 5. AVERAGE PRICE PER LOCALITY (TOP 20)
# ============================================================================
print("\n[5/14] 🏙️ Average Price per Locality - Most important chart for buyers & builders")

# Get top 20 localities by average price
top_20_localities = df.groupby('Locality').agg({
    'Price_Lakhs': ['mean', 'count']
}).round(2)
top_20_localities.columns = ['Avg_Price', 'Count']
top_20_localities = top_20_localities[top_20_localities['Count'] >= 3]  # At least 3 properties
top_20_localities = top_20_localities.sort_values('Avg_Price', ascending=True).tail(20)

fig, ax = plt.subplots(figsize=(14, 10))
bars = ax.barh(range(len(top_20_localities)), top_20_localities['Avg_Price'], 
               color='coral', edgecolor='black', alpha=0.8)

# Color code by price range
for i, (idx, row) in enumerate(top_20_localities.iterrows()):
    if row['Avg_Price'] > 200:
        bars[i].set_color('#E74C3C')  # Red for luxury
    elif row['Avg_Price'] > 150:
        bars[i].set_color('#F39C12')  # Orange for premium
    elif row['Avg_Price'] > 100:
        bars[i].set_color('#3498DB')  # Blue for mid-range
    else:
        bars[i].set_color('#2ECC71')  # Green for affordable

ax.set_yticks(range(len(top_20_localities)))
ax.set_yticklabels(top_20_localities.index, fontsize=11)
ax.set_xlabel('Average Price (Lakhs)', fontsize=14, fontweight='bold')
ax.set_ylabel('Locality', fontsize=14, fontweight='bold')
ax.set_title('Top 20 Localities by Average Price\nGreen: Affordable | Blue: Mid-range | Orange: Premium | Red: Luxury', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (idx, row) in enumerate(top_20_localities.iterrows()):
    ax.text(row['Avg_Price'] + 3, i, f"₹{row['Avg_Price']:.1f}L ({int(row['Count'])} props)", 
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{viz_dir}/05_avg_price_per_locality_top20.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 05_avg_price_per_locality_top20.png")

# ============================================================================
# 6. LOCALITY-WISE PRICE PER SQFT (BOX PLOT)
# ============================================================================
print("\n[6/14] 📦 Locality-wise Price per Sqft Box Plot - Shows price variability & investment risk")

# Get top 15 localities by property count
top_15_locs = df['Locality'].value_counts().head(15).index
df_top15 = df[df['Locality'].isin(top_15_locs)]

# Sort localities by median price per sqft
locality_order = df_top15.groupby('Locality')['Price_Per_SqFt'].median().sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(16, 10))
bp = ax.boxplot([df_top15[df_top15['Locality'] == loc]['Price_Per_SqFt'].values for loc in locality_order],
                 labels=locality_order, vert=False, patch_artist=True,
                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))

ax.set_xlabel('Price per Sq.Ft (₹)', fontsize=14, fontweight='bold')
ax.set_ylabel('Locality', fontsize=14, fontweight='bold')
ax.set_title('Price per Sq.Ft Distribution by Top 15 Localities\nBox width shows price variability (risk/opportunity)', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')

# Color boxes by risk level (variability)
for i, loc in enumerate(locality_order):
    loc_data = df_top15[df_top15['Locality'] == loc]['Price_Per_SqFt']
    cv = (loc_data.std() / loc_data.mean()) * 100  # Coefficient of variation
    if cv > 30:
        bp['boxes'][i].set_facecolor('#E74C3C')  # High risk
    elif cv > 20:
        bp['boxes'][i].set_facecolor('#F39C12')  # Medium risk
    else:
        bp['boxes'][i].set_facecolor('#2ECC71')  # Low risk

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ECC71', label='Low Risk (CV<20%)'),
                   Patch(facecolor='#F39C12', label='Medium Risk (CV 20-30%)'),
                   Patch(facecolor='#E74C3C', label='High Risk (CV>30%)')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig(f'{viz_dir}/06_locality_price_sqft_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 06_locality_price_sqft_boxplot.png")

# ============================================================================
# 7. TOP 10 MOST EXPENSIVE LOCALITIES
# ============================================================================
print("\n[7/14] 💰 Top 10 Most Expensive Localities - Useful for premium developers")

top_10_expensive = df.groupby('Locality').agg({
    'Price_Lakhs': ['mean', 'count'],
    'Price_Per_SqFt': 'mean'
}).round(2)
top_10_expensive.columns = ['Avg_Price', 'Count', 'Avg_PPS']
top_10_expensive = top_10_expensive[top_10_expensive['Count'] >= 2]
top_10_expensive = top_10_expensive.sort_values('Avg_Price', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(top_10_expensive))
bars = ax.bar(x, top_10_expensive['Avg_Price'], color='#8E44AD', edgecolor='black', alpha=0.8)

# Add gradient effect
for i, bar in enumerate(bars):
    bar.set_color(plt.cm.Reds(0.4 + (i / len(bars)) * 0.6))

ax.set_xticks(x)
ax.set_xticklabels(top_10_expensive.index, rotation=45, ha='right', fontsize=11)
ax.set_xlabel('Locality', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Price (Lakhs)', fontsize=14, fontweight='bold')
ax.set_title('Top 10 Most Expensive Localities in Ahmedabad\nPremium development opportunities', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (idx, row) in enumerate(top_10_expensive.iterrows()):
    ax.text(i, row['Avg_Price'] + 5, f"₹{row['Avg_Price']:.1f}L\n₹{row['Avg_PPS']:.0f}/sqft\n({int(row['Count'])} props)", 
            ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{viz_dir}/07_top10_expensive_localities.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 07_top10_expensive_localities.png")

# ============================================================================
# 8. TOP 10 MOST AFFORDABLE LOCALITIES
# ============================================================================
print("\n[8/14] 🏠 Top 10 Most Affordable Localities - Useful for affordable housing planning")

top_10_affordable = df.groupby('Locality').agg({
    'Price_Lakhs': ['mean', 'count'],
    'Price_Per_SqFt': 'mean'
}).round(2)
top_10_affordable.columns = ['Avg_Price', 'Count', 'Avg_PPS']
top_10_affordable = top_10_affordable[top_10_affordable['Count'] >= 3]
top_10_affordable = top_10_affordable.sort_values('Avg_Price', ascending=True).head(10)

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(top_10_affordable))
bars = ax.bar(x, top_10_affordable['Avg_Price'], color='#27AE60', edgecolor='black', alpha=0.8)

# Add gradient effect
for i, bar in enumerate(bars):
    bar.set_color(plt.cm.Greens(0.4 + (i / len(bars)) * 0.6))

ax.set_xticks(x)
ax.set_xticklabels(top_10_affordable.index, rotation=45, ha='right', fontsize=11)
ax.set_xlabel('Locality', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Price (Lakhs)', fontsize=14, fontweight='bold')
ax.set_title('Top 10 Most Affordable Localities in Ahmedabad\nBest value for money & affordable housing potential', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (idx, row) in enumerate(top_10_affordable.iterrows()):
    ax.text(i, row['Avg_Price'] + 2, f"₹{row['Avg_Price']:.1f}L\n₹{row['Avg_PPS']:.0f}/sqft\n({int(row['Count'])} props)", 
            ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{viz_dir}/08_top10_affordable_localities.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 08_top10_affordable_localities.png")

# ============================================================================
# 9. GEOSPATIAL HEATMAP (LOCALITY PRICE INTENSITY)
# ============================================================================
print("\n[9/14] 🗺️ Geospatial Heatmap - Price intensity across localities")

# Create a comprehensive locality analysis
locality_stats = df.groupby('Locality').agg({
    'Price_Lakhs': ['mean', 'count'],
    'Price_Per_SqFt': 'mean',
    'Area_SqFt': 'mean'
}).round(2)
locality_stats.columns = ['Avg_Price', 'Property_Count', 'Avg_PPS', 'Avg_Area']
locality_stats = locality_stats[locality_stats['Property_Count'] >= 2].sort_values('Avg_Price', ascending=False)

# Create heatmap-style visualization
fig, ax = plt.subplots(figsize=(18, 12))

# Top 30 localities for better visualization
top_30 = locality_stats.head(30)
data_matrix = top_30[['Avg_Price', 'Avg_PPS', 'Avg_Area', 'Property_Count']].T

# Normalize data for heatmap
data_normalized = (data_matrix - data_matrix.min(axis=1).values.reshape(-1, 1)) / \
                  (data_matrix.max(axis=1) - data_matrix.min(axis=1)).values.reshape(-1, 1)

im = ax.imshow(data_normalized, cmap='YlOrRd', aspect='auto')

# Set ticks and labels
ax.set_xticks(np.arange(len(top_30)))
ax.set_yticks(np.arange(len(data_matrix)))
ax.set_xticklabels(top_30.index, rotation=90, ha='right', fontsize=9)
ax.set_yticklabels(['Avg Price (L)', 'Price/SqFt (₹)', 'Avg Area (sqft)', 'Property Count'], fontsize=12)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Normalized Intensity (0-1)', rotation=270, labelpad=25, fontsize=12, fontweight='bold')

# Add values to cells
for i in range(len(data_matrix)):
    for j in range(len(top_30)):
        if i == 0:  # Avg Price
            text = ax.text(j, i, f'{data_matrix.iloc[i, j]:.0f}L',
                          ha="center", va="center", color="black", fontsize=8, fontweight='bold')
        elif i == 1:  # Price per sqft
            text = ax.text(j, i, f'₹{data_matrix.iloc[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=8, fontweight='bold')
        elif i == 2:  # Avg Area
            text = ax.text(j, i, f'{data_matrix.iloc[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=8, fontweight='bold')
        else:  # Count
            text = ax.text(j, i, f'{int(data_matrix.iloc[i, j])}',
                          ha="center", va="center", color="black", fontsize=8, fontweight='bold')

ax.set_title('Geospatial Price Intensity Heatmap - Top 30 Localities\nDarker = Higher value | Best for identifying premium vs affordable areas', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{viz_dir}/09_geospatial_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 09_geospatial_heatmap.png")

print("\n" + "="*100)
print("C. PROPERTY FEATURES & COMPARISONS")
print("="*100)

# ============================================================================
# 10. FURNISHED VS UNFURNISHED PRICE COMPARISON
# ============================================================================
print("\n[10/14] 🪑 Furnished vs Unfurnished Price Comparison - Shows how furnishing affects pricing")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Price comparison
furn_price = df.groupby('Furnishing')['Price_Lakhs'].mean().sort_values(ascending=False)
bars = axes[0].bar(range(len(furn_price)), furn_price.values, 
                   color=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'], 
                   edgecolor='black', alpha=0.8)
axes[0].set_xticks(range(len(furn_price)))
axes[0].set_xticklabels(furn_price.index, rotation=15, fontsize=12)
axes[0].set_ylabel('Average Price (Lakhs)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Furnishing Status', fontsize=13, fontweight='bold')
axes[0].set_title('Average Price by Furnishing Status', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels and % difference
baseline = furn_price.iloc[-1]  # Unfurnished as baseline
for i, (furn, price) in enumerate(furn_price.items()):
    pct_diff = ((price - baseline) / baseline) * 100
    axes[0].text(i, price + 3, f'₹{price:.1f}L\n(+{pct_diff:.1f}%)' if pct_diff > 0 else f'₹{price:.1f}L', 
                ha='center', fontsize=10, fontweight='bold')

# Distribution by furnishing
furn_counts = df['Furnishing'].value_counts()
axes[1].pie(furn_counts.values, labels=furn_counts.index, autopct='%1.1f%%',
           colors=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'],
           startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Market Share by Furnishing Status', fontsize=14, fontweight='bold')

fig.suptitle('Furnishing Impact Analysis\nShows premium for furnished properties', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(f'{viz_dir}/10_furnished_vs_unfurnished.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 10_furnished_vs_unfurnished.png")

# ============================================================================
# 11. BHK VS AVERAGE PRICE
# ============================================================================
print("\n[11/14] 🏠 BHK vs Average Price - Useful for homebuyers and rental investors")

bhk_stats = df.groupby('BHK').agg({
    'Price_Lakhs': ['mean', 'count', 'std'],
    'Area_SqFt': 'mean'
}).round(2)
bhk_stats.columns = ['Avg_Price', 'Count', 'Std_Price', 'Avg_Area']
bhk_stats = bhk_stats[bhk_stats['Count'] >= 5]

fig, ax = plt.subplots(figsize=(14, 8))
x = bhk_stats.index
bars = ax.bar(x, bhk_stats['Avg_Price'], yerr=bhk_stats['Std_Price'], 
              color=['#3498DB', '#2ECC71', '#F39C12', '#E74C3C', '#9B59B6'][:len(x)],
              edgecolor='black', alpha=0.8, capsize=10)

ax.set_xlabel('BHK Configuration', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Price (Lakhs)', fontsize=14, fontweight='bold')
ax.set_title('Average Price by BHK Configuration\nError bars show price variability (±1 std dev)', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(x)
ax.set_xticklabels([f'{int(bhk)} BHK' for bhk in x], fontsize=12)

# Add detailed labels
for i, bhk in enumerate(x):
    price = bhk_stats.loc[bhk, 'Avg_Price']
    count = int(bhk_stats.loc[bhk, 'Count'])
    area = bhk_stats.loc[bhk, 'Avg_Area']
    ax.text(bhk, price + 15, f'₹{price:.1f}L\n{count} props\n{area:.0f} sqft avg', 
            ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{viz_dir}/11_bhk_vs_avg_price.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 11_bhk_vs_avg_price.png")

# ============================================================================
# 12. BHK VS PRICE PER SQFT (BOX PLOT)
# ============================================================================
print("\n[12/14] 📊 BHK vs Price per Sqft Box Plot - Shows how price-per-unit changes with larger BHK")

# Filter BHKs with enough data
bhk_filter = df['BHK'].value_counts()
valid_bhks = bhk_filter[bhk_filter >= 10].index.sort_values()
df_bhk = df[df['BHK'].isin(valid_bhks)]

fig, ax = plt.subplots(figsize=(14, 8))
bp = ax.boxplot([df_bhk[df_bhk['BHK'] == bhk]['Price_Per_SqFt'].values for bhk in valid_bhks],
                 labels=[f'{int(bhk)} BHK' for bhk in valid_bhks],
                 patch_artist=True,
                 boxprops=dict(alpha=0.7),
                 medianprops=dict(color='red', linewidth=2.5),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))

# Color boxes
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(valid_bhks)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.set_xlabel('BHK Configuration', fontsize=14, fontweight='bold')
ax.set_ylabel('Price per Sq.Ft (₹)', fontsize=14, fontweight='bold')
ax.set_title('Price per Sq.Ft Distribution by BHK\nShows economies of scale for larger configurations', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='y')

# Add median values
for i, bhk in enumerate(valid_bhks):
    median_val = df_bhk[df_bhk['BHK'] == bhk]['Price_Per_SqFt'].median()
    ax.text(i + 1, median_val, f'₹{median_val:.0f}', ha='center', va='bottom', 
            fontsize=10, fontweight='bold', color='darkred')

plt.tight_layout()
plt.savefig(f'{viz_dir}/12_bhk_vs_price_per_sqft_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 12_bhk_vs_price_per_sqft_boxplot.png")

# ============================================================================
# 13. BATHROOM COUNT VS PRICE
# ============================================================================
print("\n[13/14] 🚿 Bathroom Count vs Price - Shows relationship of amenities to pricing")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Scatter plot
scatter = axes[0].scatter(df['Bathroom'], df['Price_Lakhs'], 
                         c=df['BHK'], cmap='viridis', alpha=0.6, s=80, edgecolors='black')
axes[0].set_xlabel('Number of Bathrooms', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Price (Lakhs)', fontsize=13, fontweight='bold')
axes[0].set_title('Bathroom Count vs Property Price\nColored by BHK configuration', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=axes[0])
cbar.set_label('BHK', rotation=270, labelpad=20, fontweight='bold')

# Add trend line
z = np.polyfit(df['Bathroom'].dropna(), df['Price_Lakhs'][df['Bathroom'].notna()], 1)
p = np.poly1d(z)
axes[0].plot(df['Bathroom'].unique(), p(df['Bathroom'].unique()), 
            "r--", linewidth=2.5, label=f'Trend: y={z[0]:.1f}x+{z[1]:.1f}')
axes[0].legend(fontsize=11)

# Average price by bathroom count
bath_price = df.groupby('Bathroom').agg({
    'Price_Lakhs': ['mean', 'count']
}).round(2)
bath_price.columns = ['Avg_Price', 'Count']
bath_price = bath_price[bath_price['Count'] >= 5]

bars = axes[1].bar(bath_price.index, bath_price['Avg_Price'], 
                   color=plt.cm.plasma(np.linspace(0.2, 0.9, len(bath_price))),
                   edgecolor='black', alpha=0.8)
axes[1].set_xlabel('Number of Bathrooms', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Average Price (Lakhs)', fontsize=13, fontweight='bold')
axes[1].set_title('Average Price by Bathroom Count', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels
for bath, row in bath_price.iterrows():
    axes[1].text(bath, row['Avg_Price'] + 5, f"₹{row['Avg_Price']:.1f}L\n({int(row['Count'])} props)", 
                ha='center', fontsize=9, fontweight='bold')

fig.suptitle('Bathroom Amenity Impact on Property Pricing', fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(f'{viz_dir}/13_bathroom_vs_price.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 13_bathroom_vs_price.png")

# ============================================================================
# 14. SELLER TYPE PRICE DIFFERENCE
# ============================================================================
print("\n[14/14] 👤 Seller Type Price Difference - Displays negotiation opportunity for buyers")

seller_stats = df.groupby('Seller_Type').agg({
    'Price_Lakhs': ['mean', 'median', 'count', 'std'],
    'Price_Per_SqFt': 'mean'
}).round(2)
seller_stats.columns = ['Avg_Price', 'Median_Price', 'Count', 'Std_Price', 'Avg_PPS']
seller_stats = seller_stats[seller_stats['Count'] >= 5].sort_values('Avg_Price', ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Average price comparison
x = np.arange(len(seller_stats))
bars1 = axes[0, 0].bar(x, seller_stats['Avg_Price'], 
                       color=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'][:len(x)],
                       edgecolor='black', alpha=0.8)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(seller_stats.index, fontsize=12)
axes[0, 0].set_ylabel('Average Price (Lakhs)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Seller Type', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Average Price by Seller Type', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Add value labels with % difference from lowest
min_price = seller_stats['Avg_Price'].min()
for i, (seller, row) in enumerate(seller_stats.iterrows()):
    pct_diff = ((row['Avg_Price'] - min_price) / min_price) * 100
    axes[0, 0].text(i, row['Avg_Price'] + 5, f"₹{row['Avg_Price']:.1f}L\n+{pct_diff:.1f}%", 
                   ha='center', fontsize=10, fontweight='bold')

# 2. Property count distribution
axes[0, 1].pie(seller_stats['Count'], labels=seller_stats.index, autopct='%1.1f%%',
              colors=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'][:len(seller_stats)],
              startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[0, 1].set_title('Market Share by Seller Type', fontsize=14, fontweight='bold')

# 3. Price per sqft comparison
bars2 = axes[1, 0].bar(x, seller_stats['Avg_PPS'], 
                       color=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'][:len(x)],
                       edgecolor='black', alpha=0.8)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(seller_stats.index, fontsize=12)
axes[1, 0].set_ylabel('Avg Price per Sq.Ft (₹)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Seller Type', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Price per Sq.Ft by Seller Type', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, (seller, row) in enumerate(seller_stats.iterrows()):
    axes[1, 0].text(i, row['Avg_PPS'] + 150, f"₹{row['Avg_PPS']:.0f}", 
                   ha='center', fontsize=10, fontweight='bold')

# 4. Price variability (negotiation opportunity)
bars3 = axes[1, 1].bar(x, seller_stats['Std_Price'], 
                       color=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'][:len(x)],
                       edgecolor='black', alpha=0.8)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(seller_stats.index, fontsize=12)
axes[1, 1].set_ylabel('Price Std Deviation (Lakhs)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Seller Type', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Price Variability (Negotiation Opportunity)\nHigher = More room for negotiation', 
                     fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, (seller, row) in enumerate(seller_stats.iterrows()):
    cv = (row['Std_Price'] / row['Avg_Price']) * 100
    axes[1, 1].text(i, row['Std_Price'] + 2, f"±₹{row['Std_Price']:.1f}L\nCV: {cv:.1f}%", 
                   ha='center', fontsize=9, fontweight='bold')

fig.suptitle('Seller Type Analysis: Pricing & Negotiation Opportunities\nOwner-direct typically offers better negotiation potential', 
             fontsize=16, fontweight='bold', y=1.00)

plt.tight_layout()
plt.savefig(f'{viz_dir}/14_seller_type_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 14_seller_type_analysis.png")

print("\n" + "="*100)
print("COMPREHENSIVE EDA VISUALIZATION COMPLETE!")
print("="*100)

# Generate summary report
print("\n📊 GENERATION SUMMARY:")
print("   ✓ A. PRICE & DISTRIBUTION INSIGHTS: 4 visualizations")
print("   ✓ B. LOCATION-BASED INSIGHTS: 5 visualizations")
print("   ✓ C. PROPERTY FEATURES & COMPARISONS: 5 visualizations")
print(f"\n   📁 Total visualizations created: 14")
print(f"   📂 Saved to: {os.path.abspath(viz_dir)}")

print("\n🎯 KEY INSIGHTS:")
print(f"   • Average property price: ₹{df['Price_Lakhs'].mean():.2f} Lakhs")
print(f"   • Most common BHK: {df['BHK'].mode()[0]} BHK")
print(f"   • Average price per sqft: ₹{df['Price_Per_SqFt'].mean():.0f}")
print(f"   • Total localities analyzed: {df['Locality'].nunique()}")
print(f"   • Furnished premium: ~{((df[df['Furnishing']=='Furnished']['Price_Lakhs'].mean() / df[df['Furnishing']=='Unfurnished']['Price_Lakhs'].mean() - 1) * 100):.1f}%")

print("\n✅ ALL VISUALIZATIONS SUCCESSFULLY GENERATED!")
print("="*100)
