"""
Advanced Techniques to Boost Accuracy
1. Feature Engineering: Price per BHK (realistic)
2. Ensemble Stacking
3. XGBoost (if available)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ADVANCED ACCURACY BOOSTING TECHNIQUES")
print("="*70)

# Load data
df = pd.read_csv('../data/processed/featured_real_estate_data.csv')
print(f"\n[INFO] Dataset: {df.shape}")

# === NEW FEATURE: Area per BHK (realistic ratio) ===
print(f"\n[STEP 1] Adding realistic feature...")
df['Area_per_BHK'] = df['Area_SqFt'] / df['BHK']
print("✓ Area_per_BHK - Room spaciousness indicator")

# Features
feature_columns = [
    'Area_SqFt',
    'BHK', 
    'Locality_Encoded',
    'Furnishing_Encoded',
    'Locality_Tier_Encoded',
    'Area_per_BHK'  # NEW
]

X = df[feature_columns].fillna(df[feature_columns].median())
y = df['Price_Lakhs']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n[BASELINE] Current: 74.40%")
print(f"\n[STEP 2] Training advanced models...")

# === Test 1: Tuned Random Forest with new feature ===
print("\n  [Test 1] Random Forest + Area_per_BHK...")
rf_tuned = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_tuned.fit(X_train_scaled, y_train)
y_pred_rf = rf_tuned.predict(X_test_scaled)
rf_score = r2_score(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
print(f"    R²: {rf_score*100:.2f}% | MAE: ₹{rf_mae:.2f}L")

# === Test 2: Stacking Ensemble ===
print("\n  [Test 2] Stacking Ensemble (RF + GB + Ridge)...")
base_models = [
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42))
]
stacking = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=10),
    cv=5
)
stacking.fit(X_train_scaled, y_train)
y_pred_stack = stacking.predict(X_test_scaled)
stack_score = r2_score(y_test, y_pred_stack)
stack_mae = mean_absolute_error(y_test, y_pred_stack)
print(f"    R²: {stack_score*100:.2f}% | MAE: ₹{stack_mae:.2f}L")

# === Test 3: Try XGBoost if available ===
try:
    print("\n  [Test 3] XGBoost (if available)...")
    import xgboost as xgb
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    xgb_score = r2_score(y_test, y_pred_xgb)
    xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
    print(f"    R²: {xgb_score*100:.2f}% | MAE: ₹{xgb_mae:.2f}L")
    has_xgb = True
except ImportError:
    print("    XGBoost not installed, skipping...")
    xgb_score = 0
    has_xgb = False

# === Test 4: Weighted Ensemble (Average best models) ===
print("\n  [Test 4] Weighted Ensemble Averaging...")
if has_xgb:
    y_pred_ensemble = (0.4 * y_pred_rf + 0.3 * y_pred_stack + 0.3 * y_pred_xgb)
else:
    y_pred_ensemble = (0.6 * y_pred_rf + 0.4 * y_pred_stack)
ensemble_score = r2_score(y_test, y_pred_ensemble)
ensemble_mae = mean_absolute_error(y_test, y_pred_ensemble)
print(f"    R²: {ensemble_score*100:.2f}% | MAE: ₹{ensemble_mae:.2f}L")

# Find best model
results = {
    'Random Forest + Feature': (rf_score, rf_mae, rf_tuned),
    'Stacking Ensemble': (stack_score, stack_mae, stacking),
    'Weighted Ensemble': (ensemble_score, ensemble_mae, None)
}

if has_xgb:
    results['XGBoost'] = (xgb_score, xgb_mae, xgb_model)

best_name = max(results, key=lambda x: results[x][0])
best_score, best_mae, best_model = results[best_name]

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"Baseline:       74.40%")
print(f"\nBest Technique: {best_name}")
print(f"Accuracy:       {best_score*100:.2f}%")
print(f"MAE:            ₹{best_mae:.2f}L")
improvement = best_score * 100 - 74.40
if improvement > 0:
    print(f"Improvement:    +{improvement:.2f}% 🎉")
else:
    print(f"Change:         {improvement:.2f}%")

# If improvement, save the best model
if improvement > 0.5 and best_model is not None:
    print(f"\n[SAVING] New best model ({best_name})...")
    
    with open('best_model_realistic.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        importance_df.to_csv('../data/processed/feature_importance.csv', index=False)
        
        print(f"\n[FEATURE IMPORTANCE]")
        for idx, row in importance_df.iterrows():
            print(f"  {row['Feature']}: {row['Importance']*100:.2f}%")
    
    # Update comparison file
    y_train_pred = best_model.predict(X_train_scaled)
    train_r2 = r2_score(y_train, y_train_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf if best_name == 'Random Forest + Feature' else y_pred_stack))
    mape = np.mean(np.abs((y_test - (y_pred_rf if best_name == 'Random Forest + Feature' else y_pred_stack)) / y_test)) * 100
    
    results_data = [{
        'Model': best_name,
        'Train_R2': train_r2,
        'Test_R2': best_score,
        'RMSE_Lakhs': rmse,
        'MAE_Lakhs': best_mae,
        'MAPE_%': mape,
        'Overfitting_Gap': train_r2 - best_score
    }]
    
    # Add other models for comparison
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    
    other_models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=15, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    for name, model in other_models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_train_pred = model.predict(X_train_scaled)
        
        results_data.append({
            'Model': name,
            'Train_R2': r2_score(y_train, y_train_pred),
            'Test_R2': r2_score(y_test, y_pred),
            'RMSE_Lakhs': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE_Lakhs': mean_absolute_error(y_test, y_pred),
            'MAPE_%': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
            'Overfitting_Gap': r2_score(y_train, y_train_pred) - r2_score(y_test, y_pred)
        })
    
    results_df = pd.DataFrame(results_data).sort_values('Test_R2', ascending=False)
    results_df.to_csv('../data/processed/model_comparison_results.csv', index=False)
    
    print(f"[SAVED] Model and results updated!")
else:
    print(f"\n[INFO] No significant improvement. Keeping current model.")

print("\n" + "="*70)
print("ACCURACY OPTIMIZATION COMPLETE!")
print("="*70)
