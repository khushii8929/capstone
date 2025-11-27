"""
Generate Model Performance Visualizations with Clean Features
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
print("GENERATING MODEL PERFORMANCE VISUALIZATIONS")
print("="*80)

# Create visualizations directory
import os
viz_dir = '../visualizations/model_performance'
os.makedirs(viz_dir, exist_ok=True)

# Load model comparison results
print("\n[INFO] Loading model results...")
results_df = pd.read_csv('../data/processed/model_comparison_results.csv')
print(f"[OK] Loaded {len(results_df)} model results")

# Load best model and make predictions
print("\n[INFO] Loading best model...")
with open('../notebooks/best_model_realistic.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../notebooks/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('../notebooks/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Load featured data
df = pd.read_csv('../data/processed/featured_real_estate_data.csv')
print(f"[OK] Dataset loaded: {df.shape}")

# Prepare data for predictions
X = df[feature_columns].copy()
X = X.fillna(X.median())
y = df['Price_Lakhs'].copy()

# Make predictions
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

plt.style.use('seaborn-v0_8-darkgrid')

print(f"\n[INFO] Generating visualizations...")

# 1. Model Performance Comparison - Bar Charts
print("  [1/7] Model comparison charts...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Performance Comparison (5 Realistic Features - No Leakage)', fontsize=16, fontweight='bold')

# R² Score comparison
axes[0, 0].barh(results_df['Model'], results_df['Test_R2'], color='steelblue', alpha=0.7)
axes[0, 0].set_xlabel('R² Score', fontsize=11)
axes[0, 0].set_title('Model Accuracy (R² Score)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlim(0, 1)
axes[0, 0].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(results_df['Test_R2']):
    axes[0, 0].text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10)

# MAE comparison
axes[0, 1].barh(results_df['Model'], results_df['MAE_Lakhs'], color='coral', alpha=0.7)
axes[0, 1].set_xlabel('MAE (Lakhs)', fontsize=11)
axes[0, 1].set_title('Mean Absolute Error (Lower is Better)', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(results_df['MAE_Lakhs']):
    axes[0, 1].text(v + 0.2, i, f'Rs.{v:.2f}L', va='center', fontsize=10)

# RMSE comparison
axes[1, 0].barh(results_df['Model'], results_df['RMSE_Lakhs'], color='lightcoral', alpha=0.7)
axes[1, 0].set_xlabel('RMSE (Lakhs)', fontsize=11)
axes[1, 0].set_title('Root Mean Square Error (Lower is Better)', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(results_df['RMSE_Lakhs']):
    axes[1, 0].text(v + 0.3, i, f'Rs.{v:.2f}L', va='center', fontsize=10)

# MAPE comparison
axes[1, 1].barh(results_df['Model'], results_df['MAPE_%'], color='lightgreen', alpha=0.7)
axes[1, 1].set_xlabel('MAPE (%)', fontsize=11)
axes[1, 1].set_title('Mean Absolute Percentage Error (Lower is Better)', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(results_df['MAPE_%']):
    axes[1, 1].text(v + 0.3, i, f'{v:.2f}%', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(f'{viz_dir}/01_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Actual vs Predicted scatter plot
print("  [2/7] Actual vs Predicted scatter...")
plt.figure(figsize=(12, 8))
plt.scatter(y, y_pred, alpha=0.5, s=30, c='steelblue', edgecolors='black', linewidths=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price (Lakhs)', fontsize=12)
plt.ylabel('Predicted Price (Lakhs)', fontsize=12)
plt.title('Random Forest: Actual vs Predicted Prices (5 Features, No Leakage)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Add R² annotation
best_r2 = results_df.iloc[0]['Test_R2']
best_mae = results_df.iloc[0]['MAE_Lakhs']
plt.text(0.05, 0.95, f'R² = {best_r2:.4f}\nMAE = Rs.{best_mae:.2f}L\nHonest Performance!', 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.tight_layout()
plt.savefig(f'{viz_dir}/02_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Residual plot
print("  [3/7] Residual plot...")
residuals = y - y_pred
plt.figure(figsize=(12, 6))
plt.scatter(y_pred, residuals, alpha=0.5, s=30, c='coral', edgecolors='black', linewidths=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price (Lakhs)', fontsize=12)
plt.ylabel('Residuals (Lakhs)', fontsize=12)
plt.title('Random Forest: Residual Plot (Error Distribution)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{viz_dir}/03_residual_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Residual distribution histogram
print("  [4/7] Residual distribution...")
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
plt.xlabel('Residuals (Lakhs)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Random Forest: Residual Distribution', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Add statistics
plt.text(0.02, 0.98, f'Mean: Rs.{residuals.mean():.2f}L\nStd Dev: Rs.{residuals.std():.2f}L', 
         transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
plt.tight_layout()
plt.savefig(f'{viz_dir}/04_residual_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Prediction error percentage
print("  [5/7] Error percentage distribution...")
error_pct = np.abs((y - y_pred) / y) * 100
plt.figure(figsize=(12, 6))
plt.hist(error_pct, bins=50, edgecolor='black', alpha=0.7, color='orange')
plt.axvline(x=error_pct.mean(), color='r', linestyle='--', lw=2, 
            label=f'Mean: {error_pct.mean():.2f}%')
plt.xlabel('Prediction Error (%)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Random Forest: Prediction Error Percentage Distribution', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{viz_dir}/05_error_percentage.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Feature Importance
print("  [6/7] Feature importance...")
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], 
         color='teal', alpha=0.7)
plt.xlabel('Feature Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Random Forest: Feature Importance Analysis (5 Realistic Features)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{viz_dir}/06_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Save feature importance to CSV
feature_importance.to_csv('../data/processed/feature_importance.csv', index=False)
print(f"  [OK] Feature importance saved to CSV")

# 7. Error by price range
print("  [7/7] Error by price range...")
price_ranges = pd.cut(y, bins=[0, 50, 100, 200, 1000], labels=['<50L', '50-100L', '100-200L', '>200L'])
error_by_range = pd.DataFrame({
    'Price_Range': price_ranges,
    'Absolute_Error': np.abs(residuals)
})
error_summary = error_by_range.groupby('Price_Range', observed=True)['Absolute_Error'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(error_summary['Price_Range'], error_summary['Absolute_Error'], 
        color='purple', alpha=0.7, edgecolor='black')
plt.xlabel('Price Range', fontsize=12)
plt.ylabel('Average Absolute Error (Lakhs)', fontsize=12)
plt.title('Random Forest: Average Prediction Error by Price Range', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(error_summary['Absolute_Error']):
    plt.text(i, v + 0.5, f'Rs.{v:.2f}L', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(f'{viz_dir}/07_error_by_price_range.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[OK] All 7 model performance visualizations generated successfully!")
print(f"[OK] Saved to: {viz_dir}/")
print("="*80)
