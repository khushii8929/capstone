import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RETRAINING MODELS WITH CLEAN FEATURES (NO DATA LEAKAGE)")
print("="*80)

# Load Featured Data
df = pd.read_csv('../data/processed/featured_real_estate_data.csv')
print(f"\n[OK] Dataset loaded: {df.shape}")

# Select features for modeling (MAIN REAL FEATURES ONLY)
feature_columns = [
    'Area_SqFt',           # Property area
    'BHK',                 # Number of bedrooms
    'Locality_Encoded',    # Locality (encoded)
    'Furnishing_Encoded'   # Furnishing status
]

target_column = 'Price_Lakhs'

# Create feature matrix and target vector
X = df[feature_columns].copy()
y = df[target_column].copy()

print(f"\n[OK] Feature matrix shape: {X.shape}")
print(f"[OK] Using {len(feature_columns)} legitimate features (no data leakage!)")

# Remove any rows with missing values
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    print(f"[OK] After removing missing values: {X.shape[0]} samples")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n[INFO] Training set: {X_train.shape[0]} samples")
print(f"[INFO] Testing set: {X_test.shape[0]} samples")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature columns
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print("\n" + "="*80)
print("TRAINING MODELS...")
print("="*80)

models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10)
}

results = []

for name, model in models.items():
    print(f"\n[TRAIN] Training {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    print(f"   Training R²: {train_r2:.4f}")
    print(f"   Testing R²:  {test_r2:.4f} {'[GOOD]' if test_r2 > 0.7 else '[WARN]'}")
    print(f"   RMSE: Rs.{test_rmse:.2f}L")
    print(f"   MAE: ₹{test_mae:.2f}L")
    print(f"   MAPE: {test_mape:.2f}%")
    
    # Check overfitting
    overfit = train_r2 - test_r2
    if overfit > 0.1:
        print(f"   [WARN] Overfitting: {overfit:.4f} difference")
    else:
        print(f"   [OK] Good generalization")
    
    results.append({
        'Model': name,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'RMSE': test_rmse,
        'MAE': test_mae,
        'MAPE': test_mape
    })
    
    # Save model
    model_filename = f"best_model_{name.lower().replace(' ', '_')}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

# Results summary
results_df = pd.DataFrame(results).sort_values('Test_R2', ascending=False)
print("\n" + "="*80)
print("MODEL COMPARISON (TRUE PERFORMANCE - NO DATA LEAKAGE)")
print("="*80)
print(results_df.to_string(index=False))

# Save best model info
best_model_name = results_df.iloc[0]['Model']
best_r2 = results_df.iloc[0]['Test_R2']

model_info = {
    'best_model_name': best_model_name,
    'test_r2_score': best_r2,
    'all_results': results_df.to_dict('records')
}

with open('model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

# Save comparison results
results_df.to_csv('../data/processed/model_comparison_results.csv', index=False)

print(f"\n[BEST] BEST MODEL: {best_model_name}")
print(f"[INFO] Test R² Score: {best_r2:.4f} ({best_r2*100:.2f}%)")

print("\n" + "="*80)
print("[OK] RETRAINING COMPLETE!")
print("   This is the REAL performance without data leakage!")
print("   Previous ~95% R² was inflated due to Locality_Avg_Price feature")
print("="*80)
