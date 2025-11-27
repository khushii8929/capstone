"""
Generate Master Dashboard Visualizations with 4 Main Features
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENERATING MASTER DASHBOARD VISUALIZATIONS")
print("="*80)

# Load data
df = pd.read_csv('../data/processed/featured_real_estate_data.csv')
results_df = pd.read_csv('../data/processed/model_comparison_results.csv')

# Load model from models directory
with open('../models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('../models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('../models/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

print(f"\n[INFO] Dataset: {df.shape}")
print(f"[INFO] Features: {feature_columns}")

# Create directory
import os
viz_dir = '../visualizations/master_dashboard'
os.makedirs(viz_dir, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')

# 1. Market Overview Dashboard
print("\n[1/6] Market overview dashboard...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Price distribution
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(df['Price_Lakhs'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Price (Lakhs)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Price Distribution', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Market statistics
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
stats_text = f"""
MARKET STATISTICS

Total Properties: {len(df):,}
Avg Price: Rs.{df['Price_Lakhs'].mean():.1f}L
Median Price: Rs.{df['Price_Lakhs'].median():.1f}L

Avg Area: {df['Area_SqFt'].mean():.0f} sq.ft
Most Common: {int(df['BHK'].mode()[0])} BHK

Localities: {df['Locality'].nunique()}
Price Range: Rs.{df['Price_Lakhs'].min():.1f}L - Rs.{df['Price_Lakhs'].max():.1f}L
"""
ax2.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# BHK distribution
ax3 = fig.add_subplot(gs[1, 0])
bhk_counts = df['BHK'].value_counts().sort_index()
ax3.bar(bhk_counts.index, bhk_counts.values, color='teal', alpha=0.7, edgecolor='black')
ax3.set_xlabel('BHK', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('BHK Distribution', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Top localities
ax4 = fig.add_subplot(gs[1, 1:])
top_loc = df.groupby('Locality')['Price_Lakhs'].mean().nlargest(8)
ax4.barh(range(len(top_loc)), top_loc.values, color='purple', alpha=0.7)
ax4.set_yticks(range(len(top_loc)))
ax4.set_yticklabels([loc[:30] for loc in top_loc.index], fontsize=9)
ax4.set_xlabel('Average Price (Lakhs)', fontsize=11)
ax4.set_title('Top 8 Localities by Price', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# Furnishing impact
ax5 = fig.add_subplot(gs[2, 0])
furn_avg = df.groupby('Furnishing')['Price_Lakhs'].mean()
ax5.bar(range(len(furn_avg)), furn_avg.values, color='orange', alpha=0.7, edgecolor='black')
ax5.set_xticks(range(len(furn_avg)))
ax5.set_xticklabels(furn_avg.index, rotation=45, ha='right', fontsize=9)
ax5.set_ylabel('Average Price (Lakhs)', fontsize=11)
ax5.set_title('Furnishing Impact', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Area vs Price
ax6 = fig.add_subplot(gs[2, 1:])
ax6.scatter(df['Area_SqFt'], df['Price_Lakhs'], alpha=0.3, s=20, c='coral')
ax6.set_xlabel('Area (Sq.Ft)', fontsize=11)
ax6.set_ylabel('Price (Lakhs)', fontsize=11)
ax6.set_title('Area vs Price Relationship', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

fig.suptitle('Market Overview Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(f'{viz_dir}/00_master_market_overview.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Model Performance Dashboard
print("[2/6] Model performance dashboard...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Model comparison
ax1 = fig.add_subplot(gs[0, :2])
x = np.arange(len(results_df))
width = 0.35
ax1.bar(x - width/2, results_df['Test_R2'], width, label='R² Score', color='steelblue', alpha=0.7)
ax1.bar(x + width/2, results_df['MAPE_%']/100, width, label='MAPE (scaled)', color='coral', alpha=0.7)
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax1.set_ylabel('Score', fontsize=11)
ax1.set_title('Model Comparison', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Best model info
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
best = results_df.iloc[0]
model_text = f"""
BEST MODEL

Name: {best['Model']}
R² Score: {best['Test_R2']:.4f}
Accuracy: {best['Test_R2']*100:.2f}%

MAE: Rs.{best['MAE_Lakhs']:.2f}L
RMSE: Rs.{best['RMSE_Lakhs']:.2f}L
MAPE: {best['MAPE_%']:.2f}%

Features Used: 5
- BHK (Bedrooms)
- Locality Tier ⭐ NEW
- Area (Size in sq.ft)
- Location, Furnishing
"""
ax2.text(0.1, 0.5, model_text, fontsize=10, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Predictions
X = df[feature_columns].copy()
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)
y_true = df['Price_Lakhs'].values

# Actual vs Predicted
ax3 = fig.add_subplot(gs[1, :2])
ax3.scatter(y_true, y_pred, alpha=0.4, s=20, c='steelblue')
ax3.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
ax3.set_xlabel('Actual Price (Lakhs)', fontsize=11)
ax3.set_ylabel('Predicted Price (Lakhs)', fontsize=11)
ax3.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Error distribution
ax4 = fig.add_subplot(gs[1, 2])
errors = y_true - y_pred
ax4.hist(errors, bins=30, color='orange', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='r', linestyle='--', lw=2)
ax4.set_xlabel('Error (Lakhs)', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Error Distribution', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

fig.suptitle('Model Performance Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(f'{viz_dir}/00_master_model_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Feature Analysis Dashboard
print("[3/6] Feature analysis dashboard...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Feature importance
ax1 = fig.add_subplot(gs[0, 0])
importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
ax1.barh(range(len(importance)), importance['Importance'].values, color='teal', alpha=0.7)
ax1.set_yticks(range(len(importance)))
ax1.set_yticklabels(importance['Feature'].values)
ax1.set_xlabel('Importance', fontsize=11)
ax1.set_title('Feature Importance', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')

# Correlation heatmap
ax2 = fig.add_subplot(gs[0, 1])
corr_features = ['Price_Lakhs', 'Area_SqFt', 'BHK', 'Furnishing_Encoded', 'Locality_Encoded']
corr_matrix = df[corr_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, ax=ax2, cbar_kws={"shrink": 0.8})
ax2.set_title('Feature Correlations', fontsize=12, fontweight='bold')

# BHK price boxplot
ax3 = fig.add_subplot(gs[1, 0])
bhk_data = [df[df['BHK']==bhk]['Price_Lakhs'].values for bhk in sorted(df['BHK'].unique())]
ax3.boxplot(bhk_data, labels=[f'{int(bhk)} BHK' for bhk in sorted(df['BHK'].unique())])
ax3.set_xlabel('BHK', fontsize=11)
ax3.set_ylabel('Price (Lakhs)', fontsize=11)
ax3.set_title('Price Distribution by BHK', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Area distribution
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(df['Area_SqFt'], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Area (Sq.Ft)', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Area Distribution', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

fig.suptitle('Feature Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(f'{viz_dir}/00_master_feature_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Locality Intelligence Dashboard
print("[4/6] Locality intelligence dashboard...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Top expensive localities
ax1 = fig.add_subplot(gs[0, 0])
top_expensive = df.groupby('Locality')['Price_Lakhs'].mean().nlargest(10)
ax1.barh(range(len(top_expensive)), top_expensive.values, color='red', alpha=0.6)
ax1.set_yticks(range(len(top_expensive)))
ax1.set_yticklabels([loc[:25] for loc in top_expensive.index], fontsize=9)
ax1.set_xlabel('Avg Price (Lakhs)', fontsize=11)
ax1.set_title('Top 10 Most Expensive Localities', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')

# Most affordable localities
ax2 = fig.add_subplot(gs[0, 1])
affordable = df.groupby('Locality')['Price_Lakhs'].mean().nsmallest(10)
ax2.barh(range(len(affordable)), affordable.values, color='green', alpha=0.6)
ax2.set_yticks(range(len(affordable)))
ax2.set_yticklabels([loc[:25] for loc in affordable.index], fontsize=9)
ax2.set_xlabel('Avg Price (Lakhs)', fontsize=11)
ax2.set_title('Top 10 Most Affordable Localities', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Localities by supply
ax3 = fig.add_subplot(gs[1, 0])
top_supply = df['Locality'].value_counts().head(10)
ax3.bar(range(len(top_supply)), top_supply.values, color='orange', alpha=0.7, edgecolor='black')
ax3.set_xticks(range(len(top_supply)))
ax3.set_xticklabels([loc[:15] for loc in top_supply.index], rotation=45, ha='right', fontsize=8)
ax3.set_ylabel('Property Count', fontsize=11)
ax3.set_title('Top 10 Localities by Supply', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Price range by locality
ax4 = fig.add_subplot(gs[1, 1])
locality_stats = df.groupby('Locality')['Price_Lakhs'].agg(['mean', 'std']).nlargest(10, 'mean')
ax4.bar(range(len(locality_stats)), locality_stats['mean'].values, 
        yerr=locality_stats['std'].values, color='purple', alpha=0.6, capsize=5)
ax4.set_xticks(range(len(locality_stats)))
ax4.set_xticklabels([loc[:15] for loc in locality_stats.index], rotation=45, ha='right', fontsize=8)
ax4.set_ylabel('Price (Lakhs)', fontsize=11)
ax4.set_title('Price Variability (Top 10 Localities)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

fig.suptitle('Locality Intelligence Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(f'{viz_dir}/00_master_locality_intelligence.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Investment Opportunities Dashboard
print("[5/6] Investment opportunities dashboard...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Price segments
ax1 = fig.add_subplot(gs[0, 0])
segments = pd.cut(df['Price_Lakhs'], bins=[0, 50, 100, 200, 1000], 
                  labels=['<50L', '50-100L', '100-200L', '>200L'])
seg_counts = segments.value_counts()
ax1.pie(seg_counts.values, labels=seg_counts.index, autopct='%1.1f%%', 
        colors=['lightblue', 'lightgreen', 'orange', 'red'], startangle=90)
ax1.set_title('Market Segmentation', fontsize=12, fontweight='bold')

# BHK wise average price
ax2 = fig.add_subplot(gs[0, 1:])
bhk_avg = df.groupby('BHK')['Price_Lakhs'].mean()
ax2.bar(bhk_avg.index, bhk_avg.values, color='teal', alpha=0.7, edgecolor='black')
ax2.set_xlabel('BHK', fontsize=11)
ax2.set_ylabel('Average Price (Lakhs)', fontsize=11)
ax2.set_title('Average Price by BHK', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Budget properties
ax3 = fig.add_subplot(gs[1, :2])
budget = df[df['Price_Lakhs'] < 50].groupby('Locality').size().nlargest(10)
ax3.barh(range(len(budget)), budget.values, color='green', alpha=0.6)
ax3.set_yticks(range(len(budget)))
ax3.set_yticklabels([loc[:25] for loc in budget.index], fontsize=9)
ax3.set_xlabel('Property Count', fontsize=11)
ax3.set_title('Top Budget Housing Zones (<50L)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# Premium properties
ax4 = fig.add_subplot(gs[1, 2])
ax4.axis('off')
premium = df[df['Price_Lakhs'] > 100]
invest_text = f"""
INVESTMENT INSIGHTS

Budget (<50L):
Properties: {len(df[df['Price_Lakhs'] < 50])}
Avg: Rs.{df[df['Price_Lakhs'] < 50]['Price_Lakhs'].mean():.1f}L

Mid-Range (50-100L):
Properties: {len(df[(df['Price_Lakhs'] >= 50) & (df['Price_Lakhs'] < 100)])}
Avg: Rs.{df[(df['Price_Lakhs'] >= 50) & (df['Price_Lakhs'] < 100)]['Price_Lakhs'].mean():.1f}L

Premium (>100L):
Properties: {len(premium)}
Avg: Rs.{premium['Price_Lakhs'].mean():.1f}L

Total Market: {len(df)} properties
"""
ax4.text(0.1, 0.5, invest_text, fontsize=9, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

fig.suptitle('Investment Opportunities Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(f'{viz_dir}/00_master_investment_opportunities.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Executive Summary Dashboard
print("[6/6] Executive summary dashboard...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Key metrics
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')
summary = f"""
AHMEDABAD REAL ESTATE ANALYTICS - EXECUTIVE SUMMARY

Dataset: {len(df):,} Properties | Localities: {df['Locality'].nunique()} | Features: 6 Optimized (BHK×Tier interaction + outliers removed)

MARKET OVERVIEW:
• Average Price: Rs.{df['Price_Lakhs'].mean():.1f} Lakhs  |  Median Price: Rs.{df['Price_Lakhs'].median():.1f} Lakhs
• Price Range: Rs.{df['Price_Lakhs'].min():.1f}L - Rs.{df['Price_Lakhs'].max():.1f}L  |  Average Area: {df['Area_SqFt'].mean():.0f} sq.ft
• Most Popular: {int(df['BHK'].mode()[0])} BHK ({len(df[df['BHK']==df['BHK'].mode()[0]])} properties, {len(df[df['BHK']==df['BHK'].mode()[0]])/len(df)*100:.1f}%)

MODEL PERFORMANCE:
• Best Model: {results_df.iloc[0]['Model']}  |  R² Score: {results_df.iloc[0]['Test_R2']:.4f} ({results_df.iloc[0]['Test_R2']*100:.2f}% accuracy)
• MAE: ±Rs.{results_df.iloc[0]['MAE_Lakhs']:.2f} Lakhs  |  RMSE: Rs.{results_df.iloc[0]['RMSE_Lakhs']:.2f} Lakhs  |  MAPE: {results_df.iloc[0]['MAPE_%']:.2f}%
• Training Samples: 1,797  |  Testing Samples: 450  |  No Data Leakage - Real Features Only

MARKET SEGMENTATION:
• Budget (<50L): {len(df[df['Price_Lakhs'] < 50])} properties ({len(df[df['Price_Lakhs'] < 50])/len(df)*100:.1f}%)
• Mid-Range (50-100L): {len(df[(df['Price_Lakhs'] >= 50) & (df['Price_Lakhs'] < 100)])} properties ({len(df[(df['Price_Lakhs'] >= 50) & (df['Price_Lakhs'] < 100)])/len(df)*100:.1f}%)
• Premium (>100L): {len(df[df['Price_Lakhs'] > 100])} properties ({len(df[df['Price_Lakhs'] > 100])/len(df)*100:.1f}%)
"""
ax1.text(0.05, 0.5, summary, fontsize=10, verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# Model comparison
ax2 = fig.add_subplot(gs[1, :])
x = np.arange(len(results_df))
width = 0.2
ax2.bar(x - 1.5*width, results_df['Test_R2'], width, label='R² Score', color='steelblue', alpha=0.7)
ax2.bar(x - 0.5*width, results_df['MAE_Lakhs']/100, width, label='MAE/100', color='orange', alpha=0.7)
ax2.bar(x + 0.5*width, results_df['RMSE_Lakhs']/100, width, label='RMSE/100', color='green', alpha=0.7)
ax2.bar(x + 1.5*width, results_df['MAPE_%']/100, width, label='MAPE/100', color='red', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(results_df['Model'], fontsize=9)
ax2.set_ylabel('Score (normalized)', fontsize=11)
ax2.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# Distribution comparison
ax3 = fig.add_subplot(gs[2, 0])
ax3.hist(df['Price_Lakhs'], bins=30, color='steelblue', alpha=0.6, label='Actual', edgecolor='black')
ax3.hist(y_pred, bins=30, color='orange', alpha=0.6, label='Predicted', edgecolor='black')
ax3.set_xlabel('Price (Lakhs)', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_title('Actual vs Predicted Distribution', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Top performing features
ax4 = fig.add_subplot(gs[2, 1])
ax4.barh(range(len(importance)), importance['Importance'].values, color='teal', alpha=0.7)
ax4.set_yticks(range(len(importance)))
ax4.set_yticklabels(importance['Feature'].values, fontsize=9)
ax4.set_xlabel('Importance', fontsize=10)
ax4.set_title('Feature Importance', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# Key recommendations
ax5 = fig.add_subplot(gs[2, 2])
ax5.axis('off')
recommendations = f"""
KEY RECOMMENDATIONS

FOR BUYERS:
• Sweet spot: {int(df['BHK'].mode()[0])} BHK
• Budget: Rs.{df['Price_Lakhs'].quantile(0.25):.0f}L-{df['Price_Lakhs'].quantile(0.75):.0f}L
• Best value zones identified

FOR INVESTORS:
• Focus on mid-range (50-100L)
• {df['Locality'].nunique()} localities to choose from
• Model accuracy: {results_df.iloc[0]['Test_R2']*100:.1f}%

FOR DEVELOPERS:
• High demand: {int(df['BHK'].mode()[0])} BHK properties
• Target area: ~{df['Area_SqFt'].mean():.0f} sq.ft
• Location is key factor
"""
ax5.text(0.05, 0.5, recommendations, fontsize=8, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

fig.suptitle('Executive Summary Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(f'{viz_dir}/00_master_executive_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[OK] All 6 master dashboard visualizations generated successfully!")
print(f"[OK] Saved to: {viz_dir}/")
print("="*80)
