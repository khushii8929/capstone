"""
Train Models with Realistic Features Only
Simple, practical, and easy to understand
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TRAINING MODELS - REALISTIC FEATURES ONLY")
print("="*70)

# Load data
df = pd.read_csv('../data/processed/featured_real_estate_data.csv')
print(f"\n[INFO] Dataset: {df.shape}")

# Define realistic features
feature_columns = [
    'Area_SqFt',
    'BHK', 
    'Locality_Encoded',
    'Furnishing_Encoded',
    'Locality_Tier_Encoded'
]

print(f"\n[FEATURES] Using 5 Realistic Features:")
for i, feat in enumerate(feature_columns, 1):
    print(f"  {i}. {feat}")

# Prepare data
X = df[feature_columns].copy()
y = df['Price_Lakhs'].copy()

# Handle missing values
X = X.fillna(X.median())

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n[SPLIT] Training: {len(X_train)} | Testing: {len(X_test)}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=15, min_samples_split=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

results = []
best_model = None
best_score = -1

print("\n" + "="*70)
print("TRAINING & EVALUATION")
print("="*70)

for name, model in models.items():
    print(f"\n[{name}]")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)
    mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    print(f"  Train R²: {train_r2:.4f} ({train_r2*100:.2f}%)")
    print(f"  Test R²:  {test_r2:.4f} ({test_r2*100:.2f}%) ⭐")
    print(f"  RMSE:     ₹{rmse:.2f} Lakhs")
    print(f"  MAE:      ₹{mae:.2f} Lakhs")
    print(f"  MAPE:     {mape:.2f}%")
    
    results.append({
        'Model': name,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'RMSE_Lakhs': rmse,
        'MAE_Lakhs': mae,
        'MAPE_%': mape,
        'Overfitting_Gap': train_r2 - test_r2
    })
    
    # Track best model
    if test_r2 > best_score:
        best_score = test_r2
        best_model = model
        best_model_name = name

print("\n" + "="*70)
print(f"BEST MODEL: {best_model_name}")
print(f"Accuracy: {best_score*100:.2f}%")
print("="*70)

# Save best model and artifacts
with open(f'best_model_realistic.pkl', 'wb') as f:
    pickle.dump(best_model, f)
    
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

# Save all models
for name, model in models.items():
    filename = f"model_{name.lower().replace(' ', '_')}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

# Save results
results_df = pd.DataFrame(results).sort_values('Test_R2', ascending=False)
results_df.to_csv('../data/processed/model_comparison_results.csv', index=False)

print(f"\n[SAVED] Best model: best_model_realistic.pkl")
print(f"[SAVED] Model comparison: model_comparison_results.csv")
print(f"[SAVED] All artifacts saved!")

# Feature importance for best model
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    importance_df.to_csv('../data/processed/feature_importance.csv', index=False)
    
    print(f"\n[FEATURE IMPORTANCE] Top features:")
    for idx, row in importance_df.iterrows():
        print(f"  {row['Feature']}: {row['Importance']*100:.2f}%")

print("\n" + "="*70)
print("MODEL TRAINING COMPLETE!")
print("="*70)
