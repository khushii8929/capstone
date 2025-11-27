"""
Advanced Interactive Visualizations for Real Estate Analysis
Uses Plotly for interactive charts and detailed insights
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import os

print("="*100)
print("ADVANCED INTERACTIVE VISUALIZATIONS - AHMEDABAD REAL ESTATE")
print("="*100)

# Load data
df = pd.read_csv('../data/processed/featured_real_estate_data.csv')
print(f"\n[INFO] Dataset loaded: {df.shape}")

# Create output directory
viz_dir = '../visualizations/advanced'
os.makedirs(viz_dir, exist_ok=True)

print("\n" + "="*100)
print("GENERATING ADVANCED INTERACTIVE VISUALIZATIONS")
print("="*100)

# ============================================================================
# 1. INTERACTIVE PRICE DISTRIBUTION WITH STATISTICS
# ============================================================================
print("\n[1/10] 📊 Interactive Price Distribution with Statistics")

fig = go.Figure()

# Add histogram
fig.add_trace(go.Histogram(
    x=df['Price_Lakhs'],
    nbinsx=60,
    name='Price Distribution',
    marker=dict(color='#3498db', line=dict(color='black', width=1)),
    hovertemplate='Price Range: %{x}<br>Count: %{y}<extra></extra>'
))

# Add mean and median lines
mean_price = df['Price_Lakhs'].mean()
median_price = df['Price_Lakhs'].median()

fig.add_vline(x=mean_price, line_dash="dash", line_color="red", 
              annotation_text=f"Mean: ₹{mean_price:.2f}L", annotation_position="top")
fig.add_vline(x=median_price, line_dash="dash", line_color="green",
              annotation_text=f"Median: ₹{median_price:.2f}L", annotation_position="bottom")

fig.update_layout(
    title='Interactive Price Distribution - Ahmedabad Real Estate<br><sub>Hover for details | Use toolbar to zoom and pan</sub>',
    xaxis_title='Price (Lakhs)',
    yaxis_title='Number of Properties',
    template='plotly_white',
    height=600,
    showlegend=True,
    hovermode='closest'
)

fig.write_html(f'{viz_dir}/01_interactive_price_distribution.html')
print("   ✓ Saved: 01_interactive_price_distribution.html")

# ============================================================================
# 2. 3D SCATTER: AREA vs PRICE vs BHK
# ============================================================================
print("\n[2/10] 🎯 3D Scatter: Area vs Price vs BHK")

fig = px.scatter_3d(df, x='Area_SqFt', y='Price_Lakhs', z='BHK',
                     color='Property_Type', size='Price_Per_SqFt',
                     hover_data=['Locality', 'Furnishing', 'Seller_Type'],
                     title='3D Property Analysis: Area × Price × BHK<br><sub>Size represents Price per Sq.Ft | Color shows Property Type</sub>',
                     labels={'Area_SqFt': 'Area (Sq.Ft)', 'Price_Lakhs': 'Price (Lakhs)', 'BHK': 'BHK Configuration'},
                     height=700)

fig.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')))
fig.write_html(f'{viz_dir}/02_3d_scatter_area_price_bhk.html')
print("   ✓ Saved: 02_3d_scatter_area_price_bhk.html")

# ============================================================================
# 3. INTERACTIVE LOCALITY PRICE MAP
# ============================================================================
print("\n[3/10] 🗺️ Interactive Locality Price Map")

locality_stats = df.groupby('Locality').agg({
    'Price_Lakhs': ['mean', 'count'],
    'Price_Per_SqFt': 'mean',
    'Area_SqFt': 'mean'
}).round(2)
locality_stats.columns = ['Avg_Price', 'Property_Count', 'Avg_PPS', 'Avg_Area']
locality_stats = locality_stats[locality_stats['Property_Count'] >= 2].reset_index()
locality_stats = locality_stats.sort_values('Avg_Price', ascending=False).head(50)

fig = px.scatter(locality_stats, x='Avg_Price', y='Avg_PPS',
                 size='Property_Count', color='Avg_Area',
                 hover_name='Locality',
                 hover_data={'Avg_Price': ':.2f', 'Avg_PPS': ':.0f', 'Property_Count': True, 'Avg_Area': ':.0f'},
                 title='Locality Price Analysis Map (Top 50 Localities)<br><sub>Bubble size: Property count | Color: Average area | Position: Price metrics</sub>',
                 labels={'Avg_Price': 'Average Price (Lakhs)', 'Avg_PPS': 'Price per Sq.Ft (₹)',
                        'Property_Count': 'Properties', 'Avg_Area': 'Avg Area (sqft)'},
                 color_continuous_scale='Viridis',
                 height=700)

fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(template='plotly_white')
fig.write_html(f'{viz_dir}/03_interactive_locality_map.html')
print("   ✓ Saved: 03_interactive_locality_map.html")

# ============================================================================
# 4. SUNBURST CHART: HIERARCHY ANALYSIS
# ============================================================================
print("\n[4/10] ☀️ Sunburst Chart: Property Hierarchy Analysis")

# Create aggregated hierarchy data
hierarchy_data = df.groupby(['Price_Segment', 'Property_Type', 'Furnishing']).agg({
    'Price_Lakhs': 'sum',
    'Price_Per_SqFt': 'mean'
}).reset_index()

fig = px.sunburst(hierarchy_data, 
                  path=['Price_Segment', 'Property_Type', 'Furnishing'],
                  values='Price_Lakhs',
                  color='Price_Per_SqFt',
                  color_continuous_scale='RdYlGn_r',
                  title='Hierarchical Property Analysis: Price Segment → Property Type → Furnishing<br><sub>Size: Total price value | Color: Avg Price per Sq.Ft</sub>',
                  height=800)

fig.update_traces(textinfo='label+percent parent')
fig.write_html(f'{viz_dir}/04_sunburst_hierarchy.html')
print("   ✓ Saved: 04_sunburst_hierarchy.html")

# ============================================================================
# 5. PARALLEL COORDINATES: MULTI-FEATURE ANALYSIS
# ============================================================================
print("\n[5/10] 📐 Parallel Coordinates: Multi-Feature Analysis")

# Prepare data
df_parallel = df[['Price_Lakhs', 'Area_SqFt', 'BHK', 'Price_Per_SqFt', 'Bathroom', 'Property_Type']].copy()
df_parallel = df_parallel.dropna()
df_parallel = df_parallel.sample(min(500, len(df_parallel)))  # Sample for performance

fig = px.parallel_coordinates(df_parallel,
                              dimensions=['Price_Lakhs', 'Area_SqFt', 'BHK', 'Price_Per_SqFt', 'Bathroom'],
                              color='Price_Lakhs',
                              color_continuous_scale='Turbo',
                              title='Parallel Coordinates Analysis - Feature Relationships<br><sub>Trace lines to see patterns | Color represents price</sub>',
                              labels={'Price_Lakhs': 'Price (L)', 'Area_SqFt': 'Area (sqft)', 
                                     'Price_Per_SqFt': 'Price/Sqft', 'Bathroom': 'Bathrooms'},
                              height=600)

fig.write_html(f'{viz_dir}/05_parallel_coordinates.html')
print("   ✓ Saved: 05_parallel_coordinates.html")

# ============================================================================
# 6. ANIMATED SCATTER: BHK EVOLUTION
# ============================================================================
print("\n[6/10] 🎬 Animated Scatter: BHK Configuration Evolution")

fig = px.scatter(df, x='Area_SqFt', y='Price_Lakhs',
                 animation_frame='BHK',
                 color='Furnishing',
                 size='Price_Per_SqFt',
                 hover_data=['Locality', 'Seller_Type'],
                 title='Property Characteristics by BHK Configuration<br><sub>Play animation to see how properties change across BHK types</sub>',
                 labels={'Area_SqFt': 'Area (Sq.Ft)', 'Price_Lakhs': 'Price (Lakhs)'},
                 range_x=[0, df['Area_SqFt'].max() * 1.1],
                 range_y=[0, df['Price_Lakhs'].max() * 1.1],
                 height=700)

fig.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')))
fig.update_layout(template='plotly_white')
fig.write_html(f'{viz_dir}/06_animated_bhk_evolution.html')
print("   ✓ Saved: 06_animated_bhk_evolution.html")

# ============================================================================
# 7. BOX PLOT COMPARISON: COMPREHENSIVE
# ============================================================================
print("\n[7/10] 📦 Interactive Box Plot: Comprehensive Comparison")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Price by BHK', 'Price by Furnishing', 
                   'Price per Sqft by Property Type', 'Price by Seller Type'),
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

# BHK
bhk_data = [df[df['BHK']==bhk]['Price_Lakhs'].dropna() for bhk in sorted(df['BHK'].unique()) if df[df['BHK']==bhk].shape[0] >= 10]
for i, bhk in enumerate(sorted(df['BHK'].unique())):
    if df[df['BHK']==bhk].shape[0] >= 10:
        fig.add_trace(go.Box(y=df[df['BHK']==bhk]['Price_Lakhs'], name=f'{int(bhk)} BHK',
                            marker_color=px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]),
                     row=1, col=1)

# Furnishing
for i, furn in enumerate(df['Furnishing'].unique()):
    fig.add_trace(go.Box(y=df[df['Furnishing']==furn]['Price_Lakhs'], name=furn,
                        marker_color=px.colors.qualitative.Set1[i]),
                 row=1, col=2)

# Property Type
for i, ptype in enumerate(df['Property_Type'].unique()):
    fig.add_trace(go.Box(y=df[df['Property_Type']==ptype]['Price_Per_SqFt'], name=ptype,
                        marker_color=px.colors.qualitative.Pastel[i]),
                 row=2, col=1)

# Seller Type
for i, seller in enumerate(df['Seller_Type'].unique()):
    if pd.notna(seller):
        fig.add_trace(go.Box(y=df[df['Seller_Type']==seller]['Price_Lakhs'], name=seller,
                            marker_color=px.colors.qualitative.Dark2[i]),
                     row=2, col=2)

fig.update_yaxes(title_text="Price (Lakhs)", row=1, col=1)
fig.update_yaxes(title_text="Price (Lakhs)", row=1, col=2)
fig.update_yaxes(title_text="Price per Sqft (₹)", row=2, col=1)
fig.update_yaxes(title_text="Price (Lakhs)", row=2, col=2)

fig.update_layout(
    title_text='Comprehensive Price Distribution Analysis<br><sub>Interactive box plots across key features</sub>',
    height=900,
    showlegend=False,
    template='plotly_white'
)

fig.write_html(f'{viz_dir}/07_comprehensive_boxplots.html')
print("   ✓ Saved: 07_comprehensive_boxplots.html")

# ============================================================================
# 8. CORRELATION HEATMAP (INTERACTIVE)
# ============================================================================
print("\n[8/10] 🔥 Interactive Correlation Heatmap")

numeric_cols = ['Price_Lakhs', 'Area_SqFt', 'BHK', 'Bathroom', 'Price_Per_SqFt',
                'Bathroom_BHK_Ratio', 'Area_Per_Bedroom', 'Is_Top_Locality']
corr_matrix = df[numeric_cols].corr()

fig = px.imshow(corr_matrix,
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdBu_r',
                title='Feature Correlation Matrix<br><sub>Hover for exact values | Red: Positive | Blue: Negative</sub>',
                labels=dict(color="Correlation"),
                height=700)

fig.update_xaxes(side="bottom")
fig.write_html(f'{viz_dir}/08_interactive_correlation.html')
print("   ✓ Saved: 08_interactive_correlation.html")

# ============================================================================
# 9. TREEMAP: MARKET COMPOSITION
# ============================================================================
print("\n[9/10] 🌳 Treemap: Market Composition Analysis")

# Aggregate data - use Price_Lakhs to count
market_composition = df.groupby(['Property_Type', 'Price_Segment', 'Furnishing']).agg({
    'Price_Lakhs': ['sum', 'count']
}).reset_index()
market_composition.columns = ['Property_Type', 'Price_Segment', 'Furnishing', 'Total_Value', 'Property_Count']

fig = px.treemap(market_composition,
                 path=['Property_Type', 'Price_Segment', 'Furnishing'],
                 values='Total_Value',
                 color='Property_Count',
                 hover_data=['Property_Count'],
                 color_continuous_scale='YlOrRd',
                 title='Market Composition Treemap<br><sub>Size: Total market value | Color: Number of properties</sub>',
                 height=800)

fig.update_traces(textinfo='label+value+percent parent')
fig.write_html(f'{viz_dir}/09_treemap_market_composition.html')
print("   ✓ Saved: 09_treemap_market_composition.html")

# ============================================================================
# 10. COMPREHENSIVE DASHBOARD
# ============================================================================
print("\n[10/10] 📊 Comprehensive Dashboard")

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Price Distribution', 'Area vs Price', 
                   'Top 10 Localities', 'BHK Distribution',
                   'Furnishing Impact', 'Seller Type Analysis'),
    specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
           [{'type': 'bar'}, {'type': 'bar'}],
           [{'type': 'bar'}, {'type': 'bar'}]],
    vertical_spacing=0.1,
    horizontal_spacing=0.1
)

# Price Distribution
fig.add_trace(go.Histogram(x=df['Price_Lakhs'], nbinsx=50, name='Price', marker_color='#3498db'),
             row=1, col=1)

# Area vs Price
fig.add_trace(go.Scatter(x=df['Area_SqFt'], y=df['Price_Lakhs'], mode='markers',
                        marker=dict(size=5, opacity=0.6, color='coral'),
                        name='Properties'),
             row=1, col=2)

# Top 10 Localities
top_10 = df.groupby('Locality')['Price_Lakhs'].mean().nlargest(10).sort_values()
fig.add_trace(go.Bar(x=top_10.values, y=top_10.index, orientation='h',
                    marker_color='purple', name='Localities'),
             row=2, col=1)

# BHK Distribution
bhk_counts = df['BHK'].value_counts().sort_index()
fig.add_trace(go.Bar(x=bhk_counts.index, y=bhk_counts.values,
                    marker_color='teal', name='BHK'),
             row=2, col=2)

# Furnishing Impact
furn_avg = df.groupby('Furnishing')['Price_Lakhs'].mean()
fig.add_trace(go.Bar(x=furn_avg.index, y=furn_avg.values,
                    marker_color='orange', name='Furnishing'),
             row=3, col=1)

# Seller Type Analysis
seller_avg = df.groupby('Seller_Type')['Price_Lakhs'].mean().dropna()
fig.add_trace(go.Bar(x=seller_avg.index, y=seller_avg.values,
                    marker_color='green', name='Seller'),
             row=3, col=2)

# Update layout
fig.update_layout(
    title_text='Comprehensive Real Estate Dashboard<br><sub>All key metrics in one view</sub>',
    height=1200,
    showlegend=False,
    template='plotly_white'
)

fig.update_xaxes(title_text="Price (Lakhs)", row=1, col=1)
fig.update_xaxes(title_text="Area (Sq.Ft)", row=1, col=2)
fig.update_xaxes(title_text="Avg Price (Lakhs)", row=2, col=1)
fig.update_xaxes(title_text="BHK", row=2, col=2)

fig.write_html(f'{viz_dir}/10_comprehensive_dashboard.html')
print("   ✓ Saved: 10_comprehensive_dashboard.html")

print("\n" + "="*100)
print("ADVANCED VISUALIZATIONS COMPLETE!")
print("="*100)
print(f"\n   📁 Total interactive visualizations: 10")
print(f"   📂 Saved to: {os.path.abspath(viz_dir)}")
print("\n✅ ALL ADVANCED VISUALIZATIONS SUCCESSFULLY GENERATED!")
print("="*100)
