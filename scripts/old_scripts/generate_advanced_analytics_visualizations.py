import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import pickle
from scipy import stats
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Get project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("GENERATING ADVANCED ANALYTICS VISUALIZATIONS (30 Charts)")
print("="*80)

# Create output directories
viz_dir = os.path.join(project_root, 'visualizations', 'advanced_analytics')
html_dir = os.path.join(viz_dir, 'html')
os.makedirs(viz_dir, exist_ok=True)
os.makedirs(html_dir, exist_ok=True)

# Load data
data_path = os.path.join(project_root, 'data', 'processed', 'featured_real_estate_data.csv')
df = pd.read_csv(data_path)
print(f"\n✓ Dataset loaded: {df.shape[0]} properties, {df.shape[1]} features")

# Load model and feature importance
try:
    model_path = os.path.join(project_root, 'notebooks', 'best_model_random_forest.pkl')
    feature_importance_path = os.path.join(project_root, 'data', 'processed', 'feature_importance.csv')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    feature_importance = pd.read_csv(feature_importance_path)
    model_loaded = True
    print("✓ Model and feature importance loaded")
except Exception as e:
    model_loaded = False
    print(f"⚠ Model not found - some ML visuals will be skipped")

# ============================================================================
# D. AREA, SIZE & STRUCTURE
# ============================================================================
print("\n" + "="*80)
print("SECTION D: AREA, SIZE & STRUCTURE (3 Charts)")
print("="*80)

# Chart 15: Area vs Price Scatter with Trendline
print("\n[15/30] Area vs Price Scatter with Trendline...")
fig = px.scatter(df, x='Area_SqFt', y='Price_Lakhs',
                 color='BHK', size='Bathrooms',
                 hover_data=['Locality', 'Property_Type'],
                 title='Area vs Price: Linear & Non-Linear Behavior',
                 labels={'Area_SqFt': 'Area (sq.ft)', 'Price_Lakhs': 'Price (Lakhs)'},
                 trendline="lowess")
fig.update_layout(height=600)
fig.write_html('../visualizations/advanced_analytics/html/15_area_vs_price_trendline.html')
print("   ✓ Saved: 15_area_vs_price_trendline.html")

# Chart 16: Area Category vs Price Box Plot
print("[16/30] Area Category vs Price Distribution...")
fig, ax = plt.subplots(figsize=(14, 8))
order = ['Small', 'Medium', 'Large', 'Extra Large']
sns.boxplot(data=df, x='Area_Category', y='Price_Lakhs', order=order, ax=ax)
sns.stripplot(data=df, x='Area_Category', y='Price_Lakhs', order=order, 
              color='red', alpha=0.3, size=3, ax=ax)
ax.set_title('Price Distribution by Area Category\n(Buyer Segment Analysis)', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Area Category', fontsize=13)
ax.set_ylabel('Price (Lakhs)', fontsize=13)
for i, category in enumerate(order):
    median = df[df['Area_Category']==category]['Price_Lakhs'].median()
    ax.text(i, median, f'₹{median:.1f}L', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('../visualizations/advanced_analytics/16_area_category_price_segments.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 16_area_category_price_segments.png")

# Chart 17: Area per BHK vs Price (Spaciousness Analysis)
print("[17/30] Spaciousness (Area per BHK) vs Price Analysis...")
fig = px.scatter(df, x='Area_per_BHK', y='Price_Lakhs',
                 color='Property_Type', size='Bathrooms',
                 hover_data=['Locality', 'Area_SqFt', 'BHK'],
                 title='Spaciousness Analysis: Area per BHK vs Price',
                 labels={'Area_per_BHK': 'Area per BHK (sq.ft)', 'Price_Lakhs': 'Price (Lakhs)'})
fig.add_vline(x=400, line_dash="dash", line_color="red", 
              annotation_text="Compact/Standard threshold")
fig.add_vline(x=600, line_dash="dash", line_color="green", 
              annotation_text="Standard/Spacious threshold")
fig.update_layout(height=600)
fig.write_html('../visualizations/advanced_analytics/html/17_spaciousness_analysis.html')
print("   ✓ Saved: 17_spaciousness_analysis.html")

# ============================================================================
# F. ADVANCED ML & FEATURE ANALYTICS
# ============================================================================
print("\n" + "="*80)
print("SECTION F: ADVANCED ML & FEATURE ANALYTICS (4 Charts)")
print("="*80)

# Chart 20: Correlation Heatmap
print("\n[20/30] Correlation Heatmap...")
numeric_cols = ['Price_Lakhs', 'Area_SqFt', 'BHK', 'Bathrooms', 'Area_BHK_Interaction',
                'Bathroom_Area', 'Area_Squared', 'BHK_Squared', 'Area_per_BHK', 
                'Bathroom_BHK_Ratio', 'Locality_PropertyCount']
corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Heatmap\n(Price-Driving Features)', 
             fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../visualizations/advanced_analytics/20_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 20_correlation_heatmap.png")

# Chart 21: Feature Importance
if model_loaded:
    print("[21/30] Feature Importance Analysis...")
    fig, ax = plt.subplots(figsize=(12, 10))
    top_features = feature_importance.head(15)
    colors = ['red' if x > 0.10 else 'orange' if x > 0.05 else 'skyblue' 
              for x in top_features['Importance']]
    ax.barh(top_features['Feature'], top_features['Importance']*100, color=colors)
    ax.set_xlabel('Importance (%)', fontsize=13)
    ax.set_title('Top 15 Feature Importance (Random Forest)\nWhat Drives Property Prices?', 
                 fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (feat, imp) in enumerate(zip(top_features['Feature'], top_features['Importance'])):
        ax.text(imp*100 + 0.5, i, f'{imp*100:.2f}%', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('../visualizations/advanced_analytics/21_feature_importance_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: 21_feature_importance_detailed.png")

# Chart 22: Actual vs Predicted Price (if predictions exist)
if model_loaded:
    print("[22/30] Actual vs Predicted Price Comparison...")
    feature_cols = ['Area_SqFt', 'BHK', 'Bathrooms', 'Furnishing_Encoded', 'Locality_Encoded',
                    'Area_BHK_Interaction', 'Locality_Area', 'Locality_BHK', 'Furnishing_Area',
                    'Bathroom_Area', 'Area_Squared', 'BHK_Squared', 'Bathroom_Squared',
                    'Area_per_BHK', 'Bathroom_BHK_Ratio', 'Area_per_Room', 
                    'Is_Large_Property', 'Is_Luxury_Config']
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y_pred = model.predict(X)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Scatter plot
    ax1.scatter(df['Price_Lakhs'], y_pred, alpha=0.5, s=20)
    ax1.plot([df['Price_Lakhs'].min(), df['Price_Lakhs'].max()], 
             [df['Price_Lakhs'].min(), df['Price_Lakhs'].max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Price (Lakhs)', fontsize=12)
    ax1.set_ylabel('Predicted Price (Lakhs)', fontsize=12)
    ax1.set_title('Actual vs Predicted Price\nModel Validation', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Error distribution
    errors = y_pred - df['Price_Lakhs']
    ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Prediction Error (Lakhs)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
    ax2.text(0.02, 0.98, f'Mean Error: ₹{errors.mean():.2f}L\nStd Dev: ₹{errors.std():.2f}L',
             transform=ax2.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../visualizations/advanced_analytics/22_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: 22_actual_vs_predicted.png")

# Chart 23: Residual Plot
if model_loaded:
    print("[23/30] Residual Plot (Model Quality Check)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    residuals = errors
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Price (Lakhs)', fontsize=12)
    ax1.set_ylabel('Residuals (Lakhs)', fontsize=12)
    ax1.set_title('Residual Plot\nChecking Model Assumptions', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot\nNormality Check', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../visualizations/advanced_analytics/23_residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: 23_residual_analysis.png")

# ============================================================================
# G. UNIQUE BUSINESS VISUALS
# ============================================================================
print("\n" + "="*80)
print("SECTION G: UNIQUE BUSINESS VISUALS (7 Charts)")
print("="*80)

# Chart 24: Investment Risk vs Return (Bubble Chart)
print("\n[24/30] Investment Risk vs Return Analysis...")
locality_stats = df.groupby('Locality').agg({
    'Price_Per_SqFt': ['mean', 'std'],
    'Price_Lakhs': 'count',
    'Area_SqFt': 'mean'
}).reset_index()
locality_stats.columns = ['Locality', 'Avg_Price_Per_SqFt', 'Price_Volatility', 
                          'Property_Count', 'Avg_Area']
locality_stats = locality_stats[locality_stats['Property_Count'] >= 5].head(30)

fig = px.scatter(locality_stats, x='Price_Volatility', y='Avg_Price_Per_SqFt',
                 size='Property_Count', hover_data=['Locality'],
                 title='Investment Risk vs Return Analysis<br>(Top 30 Localities with 5+ Properties)',
                 labels={'Price_Volatility': 'Risk (Price Volatility)', 
                        'Avg_Price_Per_SqFt': 'Return (Avg Price/SqFt)'})
fig.add_vline(x=locality_stats['Price_Volatility'].median(), line_dash="dash", 
              annotation_text="Median Risk")
fig.add_hline(y=locality_stats['Avg_Price_Per_SqFt'].median(), line_dash="dash", 
              annotation_text="Median Return")
fig.update_layout(height=600)
fig.write_html('../visualizations/advanced_analytics/html/24_investment_risk_return.html')
print("   ✓ Saved: 24_investment_risk_return.html")

# Chart 25: Premium Index (Radar Chart)
print("[25/30] Premium Index Analysis...")
# Calculate premium factors for top localities
top_localities = df['Locality'].value_counts().head(10).index
premium_data = []

for locality in top_localities:
    loc_data = df[df['Locality'] == locality]
    premium_data.append({
        'Locality': locality,
        'Avg_Price': loc_data['Price_Per_SqFt'].mean() / df['Price_Per_SqFt'].mean(),
        'Luxury_Config': (loc_data['Is_Luxury_Config'].sum() / len(loc_data)) / 
                        (df['Is_Luxury_Config'].sum() / len(df)),
        'Furnishing': (loc_data['Furnishing_Encoded'].mean()) / df['Furnishing_Encoded'].mean(),
        'Spaciousness': loc_data['Area_per_BHK'].mean() / df['Area_per_BHK'].mean(),
        'Bathroom_Luxury': loc_data['Bathroom_BHK_Ratio'].mean() / df['Bathroom_BHK_Ratio'].mean()
    })

premium_df = pd.DataFrame(premium_data)

fig = go.Figure()
for idx, row in premium_df.iterrows():
    fig.add_trace(go.Scatterpolar(
        r=[row['Avg_Price'], row['Luxury_Config'], row['Furnishing'], 
           row['Spaciousness'], row['Bathroom_Luxury']],
        theta=['Price Premium', 'Luxury Config', 'Furnishing', 'Spaciousness', 'Bathroom Quality'],
        fill='toself',
        name=row['Locality'][:15]
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 2])),
    showlegend=True,
    title='Premium Index by Locality<br>(Relative to Market Average = 1.0)',
    height=700
)
fig.write_html('../visualizations/advanced_analytics/html/25_premium_index_radar.html')
print("   ✓ Saved: 25_premium_index_radar.html")

# Chart 26: Affordability Score Heatmap
print("[26/30] Affordability Score Analysis...")
# Create affordability matrix
bhk_types = [1, 2, 3, 4, 5]
top_15_localities = df['Locality'].value_counts().head(15).index

affordability_matrix = []
for locality in top_15_localities:
    row = []
    for bhk in bhk_types:
        subset = df[(df['Locality'] == locality) & (df['BHK'] == bhk)]
        if len(subset) > 0:
            avg_price = subset['Price_Lakhs'].mean()
            row.append(avg_price)
        else:
            row.append(np.nan)
    affordability_matrix.append(row)

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(affordability_matrix, cmap='RdYlGn_r', aspect='auto')

ax.set_xticks(np.arange(len(bhk_types)))
ax.set_yticks(np.arange(len(top_15_localities)))
ax.set_xticklabels([f'{b} BHK' for b in bhk_types])
ax.set_yticklabels([loc[:25] for loc in top_15_localities])

# Annotate cells
for i in range(len(top_15_localities)):
    for j in range(len(bhk_types)):
        if not np.isnan(affordability_matrix[i][j]):
            text = ax.text(j, i, f'₹{affordability_matrix[i][j]:.0f}L',
                          ha="center", va="center", color="black", fontsize=8)

ax.set_title('Affordability Matrix: Price by Locality & BHK Type\n(Top 15 Localities)', 
             fontsize=16, fontweight='bold', pad=20)
plt.colorbar(im, ax=ax, label='Price (Lakhs)')
plt.tight_layout()
plt.savefig('../visualizations/advanced_analytics/26_affordability_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 26_affordability_heatmap.png")

# Chart 27: Locality Competitiveness Score
print("[27/30] Locality Competitiveness Score...")
comp_localities = df['Locality'].value_counts().head(20).index
competitiveness = []

for locality in comp_localities:
    loc_data = df[df['Locality'] == locality]
    competitiveness.append({
        'Locality': locality,
        'Affordability': 100 - (loc_data['Price_Per_SqFt'].mean() / df['Price_Per_SqFt'].max() * 100),
        'Supply': (len(loc_data) / len(df)) * 100,
        'Price_Stability': 100 - (loc_data['Price_Per_SqFt'].std() / loc_data['Price_Per_SqFt'].mean() * 100),
        'Avg_Space': (loc_data['Area_per_BHK'].mean() / df['Area_per_BHK'].max()) * 100
    })

comp_df = pd.DataFrame(competitiveness)
comp_df['Overall_Score'] = comp_df[['Affordability', 'Supply', 'Price_Stability', 'Avg_Space']].mean(axis=1)
comp_df = comp_df.sort_values('Overall_Score', ascending=False)

fig = go.Figure()
fig.add_trace(go.Bar(name='Affordability', x=comp_df['Locality'], y=comp_df['Affordability']))
fig.add_trace(go.Bar(name='Supply', x=comp_df['Locality'], y=comp_df['Supply']))
fig.add_trace(go.Bar(name='Price Stability', x=comp_df['Locality'], y=comp_df['Price_Stability']))
fig.add_trace(go.Bar(name='Spaciousness', x=comp_df['Locality'], y=comp_df['Avg_Space']))

fig.update_layout(
    barmode='group',
    title='Locality Competitiveness Score<br>(Top 20 Localities)',
    xaxis_title='Locality',
    yaxis_title='Score (0-100)',
    height=600,
    xaxis={'tickangle': -45}
)
fig.write_html('../visualizations/advanced_analytics/html/27_locality_competitiveness.html')
print("   ✓ Saved: 27_locality_competitiveness.html")

# Chart 28: Buyer Persona Analysis - Family Preferred Areas
print("[28/30] Buyer Persona Analysis...")
# Family Segment: 3-4 BHK, Medium to Large area, Good bathroom ratio
family_friendly = df[(df['BHK'].isin([3, 4])) & (df['Area_per_BHK'] >= 450)].copy()
family_localities = family_friendly['Locality'].value_counts().head(15)

# Working Professional: 1-2 BHK, affordable, furnished
professional_friendly = df[(df['BHK'].isin([1, 2])) & 
                          (df['Price_Lakhs'] < df['Price_Lakhs'].quantile(0.6))].copy()
professional_localities = professional_friendly['Locality'].value_counts().head(15)

# Luxury Segment: 4+ BHK, Large area, Premium locations
luxury_segment = df[(df['BHK'] >= 4) & (df['Is_Large_Property'] == 1)].copy()
luxury_localities = luxury_segment['Locality'].value_counts().head(15)

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('Family Preferred Areas<br>(3-4 BHK, Spacious)',
                    'Working Professionals<br>(1-2 BHK, Affordable)',
                    'Luxury Segment<br>(4+ BHK, Premium)'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
)

fig.add_trace(go.Bar(x=family_localities.values, y=family_localities.index, 
                     orientation='h', name='Family', marker_color='lightblue'), row=1, col=1)
fig.add_trace(go.Bar(x=professional_localities.values, y=professional_localities.index, 
                     orientation='h', name='Professional', marker_color='lightgreen'), row=1, col=2)
fig.add_trace(go.Bar(x=luxury_localities.values, y=luxury_localities.index, 
                     orientation='h', name='Luxury', marker_color='gold'), row=1, col=3)

fig.update_layout(height=800, showlegend=False, title_text='Buyer Persona Locality Preferences')
fig.update_xaxes(title_text="Property Count")
fig.write_html('../visualizations/advanced_analytics/html/28_buyer_persona_analysis.html')
print("   ✓ Saved: 28_buyer_persona_analysis.html")

# Chart 29: Property Type Distribution by Price Segment
print("[29/30] Property Type Distribution Analysis...")
fig = px.sunburst(df, path=['Price_Segment', 'Property_Type', 'Space_Quality'],
                  values='Price_Lakhs',
                  title='Property Distribution: Price Segment → Type → Space Quality',
                  color='Price_Lakhs',
                  color_continuous_scale='RdYlGn')
fig.update_layout(height=700)
fig.write_html('../visualizations/advanced_analytics/html/29_property_type_distribution.html')
print("   ✓ Saved: 29_property_type_distribution.html")

# Chart 30: Market Segmentation Dashboard
print("[30/30] Market Segmentation Dashboard...")
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Price Distribution by Segment', 
                    'Locality Supply Distribution',
                    'Furnishing Premium Analysis',
                    'Luxury Config Distribution'),
    specs=[[{'type': 'box'}, {'type': 'pie'}],
           [{'type': 'bar'}, {'type': 'scatter'}]]
)

# Price by segment
for segment in df['Price_Segment'].unique():
    segment_data = df[df['Price_Segment'] == segment]
    fig.add_trace(go.Box(y=segment_data['Price_Lakhs'], name=segment), row=1, col=1)

# Supply distribution
supply_dist = df['Area_Category'].value_counts()
fig.add_trace(go.Pie(labels=supply_dist.index, values=supply_dist.values, 
                     name='Supply'), row=1, col=2)

# Furnishing premium
furn_premium = df.groupby('Furnishing')['Price_Per_SqFt'].mean().sort_values()
fig.add_trace(go.Bar(x=furn_premium.index, y=furn_premium.values, 
                     marker_color='coral'), row=2, col=1)

# Luxury config scatter
fig.add_trace(go.Scatter(x=df['BHK'], y=df['Price_Lakhs'], 
                         mode='markers', marker=dict(size=5, opacity=0.5),
                         name='Properties'), row=2, col=2)

fig.update_layout(height=900, showlegend=True, 
                  title_text='Market Segmentation Dashboard')
fig.write_html('../visualizations/advanced_analytics/html/30_market_segmentation_dashboard.html')
print("   ✓ Saved: 30_market_segmentation_dashboard.html")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("✅ ADVANCED ANALYTICS VISUALIZATIONS COMPLETE!")
print("="*80)

static_count = len([f for f in os.listdir('../visualizations/advanced_analytics') if f.endswith('.png')])
interactive_count = len([f for f in os.listdir('../visualizations/advanced_analytics/html') if f.endswith('.html')])

print(f"\n📊 Generated Visualizations:")
print(f"   Static PNG Charts:      {static_count} files")
print(f"   Interactive HTML:       {interactive_count} files")
print(f"   Total:                  {static_count + interactive_count} visualizations")

print(f"\n📁 Output Locations:")
print(f"   Static:     visualizations/advanced_analytics/*.png")
print(f"   Interactive: visualizations/advanced_analytics/html/*.html")

print(f"\n🎯 Visualization Categories:")
print(f"   D. Area & Structure:        3 charts (15-17)")
print(f"   F. ML & Feature Analytics:  4 charts (20-23)")
print(f"   G. Business Insights:       7 charts (24-30)")
print(f"   Total:                      14 advanced charts")

print(f"\n💡 Key Insights Available:")
print(f"   ✓ Area vs Price trends (linear/non-linear)")
print(f"   ✓ Market segmentation by buyer type")
print(f"   ✓ Investment risk-return analysis")
print(f"   ✓ Premium index by locality")
print(f"   ✓ Affordability matrix")
print(f"   ✓ Locality competitiveness scores")
print(f"   ✓ Buyer persona preferences")
print(f"   ✓ Feature importance & correlations")
print(f"   ✓ Model validation (actual vs predicted)")

print("\n" + "="*80)
print("🎉 Advanced analytics generation successful!")
print("="*80)
