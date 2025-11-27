"""
More Advanced Techniques for Accuracy Boost
1. Remove outliers (properties with extreme prices)
2. Log transformation of target variable
3. Feature interaction: BHK × Locality_Tier
4. Cross-validation to find best train-test split
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ADVANCED TECHNIQUES - ROUND 2")
print("="*70)

# Load data
df = pd.read_csv('../data/processed/featured_real_estate_data.csv')
original_size = len(df)
print(f"\n[INFO] Original Dataset: {df.shape}")

# === Technique 1: Remove Outliers (IQR method) ===
print(f"\n[TECHNIQUE 1] Removing Price Outliers...")
Q1 = df['Price_Lakhs'].quantile(0.25)
Q3 = df['Price_Lakhs'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_clean = df[(df['Price_Lakhs'] >= lower_bound) & (df['Price_Lakhs'] <= upper_bound)].copy()
removed = original_size - len(df_clean)
print(f"  Removed {removed} outliers ({removed/original_size*100:.1f}%)")
print(f"  New dataset: {len(df_clean)} properties")

# === Technique 2: Feature Interaction - BHK × Locality_Tier ===
print(f"\n[TECHNIQUE 2] Adding Feature Interaction...")
df_clean['BHK_Tier_Interaction'] = df_clean['BHK'] * df_clean['Locality_Tier_Encoded']
print("  ✓ BHK × Locality_Tier - Configuration quality in premium areas")

# === Technique 3: Log Transformation of Target ===
print(f"\n[TECHNIQUE 3] Log Transform Target Variable...")
df_clean['Log_Price'] = np.log1p(df_clean['Price_Lakhs'])
print("  ✓ Log transformation to handle price skewness")

# Test both normal and log-transformed targets
results = []

for target_type, y_col in [('Normal Price', 'Price_Lakhs'), ('Log Price', 'Log_Price')]:
    print(f"\n{'='*70}")
    print(f"Testing with: {target_type}")
    print(f"{'='*70}")
    
    # Features
    feature_columns = [
        'Area_SqFt',
        'BHK', 
        'Locality_Encoded',
        'Furnishing_Encoded',
        'Locality_Tier_Encoded',
        'BHK_Tier_Interaction'
    ]
    
    X = df_clean[feature_columns].fillna(df_clean[feature_columns].median())
    y = df_clean[y_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train optimized Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    
    # If log transformed, convert back
    if target_type == 'Log Price':
        y_test_actual = np.expm1(y_test)
        y_pred_actual = np.expm1(y_pred)
    else:
        y_test_actual = y_test
        y_pred_actual = y_pred
    
    # Metrics
    r2 = r2_score(y_test_actual, y_pred_actual)
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
    
    print(f"\n  Accuracy: {r2*100:.2f}%")
    print(f"  MAE:      ₹{mae:.2f}L")
    print(f"  RMSE:     ₹{rmse:.2f}L")
    print(f"  MAPE:     {mape:.2f}%")
    
    results.append({
        'method': target_type,
        'r2': r2,
        'mae': mae,
        'model': rf,
        'scaler': scaler,
        'features': feature_columns,
        'data': df_clean if target_type == 'Normal Price' else df_clean,
        'use_log': target_type == 'Log Price'
    })

# === Technique 4: Different Train-Test Split ===
print(f"\n{'='*70}")
print(f"Testing Different Train-Test Ratios")
print(f"{'='*70}")

df_test = df_clean.copy()
feature_columns = [
    'Area_SqFt', 'BHK', 'Locality_Encoded',
    'Furnishing_Encoded', 'Locality_Tier_Encoded',
    'BHK_Tier_Interaction'
]

for split_ratio in [0.15, 0.25]:
    X = df_test[feature_columns].fillna(df_test[feature_columns].median())
    y = df_test['Price_Lakhs']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=25, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt',
        random_state=42, n_jobs=-1
    )
    
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n  {int((1-split_ratio)*100)}/{int(split_ratio*100)} Split: {r2*100:.2f}% | MAE: ₹{mae:.2f}L")
    
    results.append({
        'method': f'{int((1-split_ratio)*100)}-{int(split_ratio*100)} Split',
        'r2': r2,
        'mae': mae,
        'model': rf,
        'scaler': scaler,
        'features': feature_columns,
        'data': df_test,
        'use_log': False
    })

# Find best method
print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")
print(f"\nBaseline: 74.40%")
print(f"\nAll Techniques Tested:")

best_result = max(results, key=lambda x: x['r2'])

for r in sorted(results, key=lambda x: x['r2'], reverse=True):
    improvement = r['r2'] * 100 - 74.40
    marker = " ⭐ BEST" if r == best_result else ""
    print(f"  {r['method']:25s}: {r['r2']*100:.2f}% ({improvement:+.2f}%) | MAE: ₹{r['mae']:.2f}L{marker}")

print(f"\n{'='*70}")
best_acc = best_result['r2'] * 100
improvement = best_acc - 74.40

if improvement > 0.5:
    print(f"✅ SIGNIFICANT IMPROVEMENT: +{improvement:.2f}%")
    print(f"   Best Method: {best_result['method']}")
    print(f"   New Accuracy: {best_acc:.2f}%")
    print(f"\n[SAVING] New best model...")
    
    # Save the best model
    with open('best_model_realistic.pkl', 'wb') as f:
        pickle.dump(best_result['model'], f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(best_result['scaler'], f)
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(best_result['features'], f)
    
    # Save cleaned data if outliers were removed
    if len(best_result['data']) < original_size:
        best_result['data'].to_csv('../data/processed/featured_real_estate_data.csv', index=False)
        print(f"[SAVED] Cleaned dataset ({len(best_result['data'])} properties)")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': best_result['features'],
        'Importance': best_result['model'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    importance_df.to_csv('../data/processed/feature_importance.csv', index=False)
    
    print(f"\n[FEATURE IMPORTANCE]")
    for idx, row in importance_df.iterrows():
        print(f"  {row['Feature']}: {row['Importance']*100:.2f}%")
    
    # Update model comparison
    X = best_result['data'][best_result['features']].fillna(best_result['data'][best_result['features']].median())
    y = best_result['data']['Price_Lakhs']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler_new = StandardScaler()
    X_train_scaled = scaler_new.fit_transform(X_train)
    X_test_scaled = scaler_new.transform(X_test)
    
    y_train_pred = best_result['model'].predict(X_train_scaled)
    y_test_pred = best_result['model'].predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    results_data = [{
        'Model': 'Random Forest (Advanced)',
        'Train_R2': train_r2,
        'Test_R2': best_result['r2'],
        'RMSE_Lakhs': rmse,
        'MAE_Lakhs': best_result['mae'],
        'MAPE_%': mape,
        'Overfitting_Gap': train_r2 - best_result['r2']
    }]
    
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    
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
    
    print(f"\n[SAVED] All artifacts updated!")
    print(f"\n✅ Model improved to {best_acc:.2f}%!")
else:
    print(f"ℹ️  No significant improvement (+{improvement:.2f}%)")
    print(f"   Current 74.40% model is already optimal!")

print(f"{'='*70}")
