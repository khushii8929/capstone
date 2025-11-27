"""
Hyperparameter Tuning for Random Forest
Using GridSearchCV to find optimal parameters
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("HYPERPARAMETER TUNING - RANDOM FOREST")
print("="*70)

# Load data
df = pd.read_csv('../data/processed/featured_real_estate_data.csv')
print(f"\n[INFO] Dataset: {df.shape}")

# Features
feature_columns = [
    'Area_SqFt',
    'BHK', 
    'Locality_Encoded',
    'Furnishing_Encoded',
    'Locality_Tier_Encoded'
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

print(f"\n[BASELINE] Current Random Forest Performance:")
print(f"  Accuracy: 73.27%")
print(f"  MAE: ₹21.94 Lakhs")

# Define parameter grid
print(f"\n[TUNING] Testing different hyperparameters...")
print("  This may take 2-3 minutes...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [15, 20, 25, None],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [2, 4, 6],
    'max_features': ['sqrt', 'log2', None]
}

# Initialize Random Forest
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# Grid Search with cross-validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# Fit
print("\n" + "="*70)
grid_search.fit(X_train_scaled, y_train)

# Best parameters
print("\n[BEST PARAMETERS FOUND]")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Train final model with best parameters
best_model = grid_search.best_estimator_

# Predictions
y_train_pred = best_model.predict(X_train_scaled)
y_test_pred = best_model.predict(X_test_scaled)

# Metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = mean_absolute_error(y_test, y_test_pred)
mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

print("\n" + "="*70)
print("TUNED MODEL PERFORMANCE")
print("="*70)
print(f"Train R²: {train_r2:.4f} ({train_r2*100:.2f}%)")
print(f"Test R²:  {test_r2:.4f} ({test_r2*100:.2f}%) ⭐")
print(f"RMSE:     ₹{rmse:.2f} Lakhs")
print(f"MAE:      ₹{mae:.2f} Lakhs")
print(f"MAPE:     {mape:.2f}%")

# Improvement
baseline_acc = 73.27
improvement = test_r2 * 100 - baseline_acc

print(f"\n[IMPROVEMENT]")
print(f"  Before: 73.27%")
print(f"  After:  {test_r2*100:.2f}%")
if improvement > 0:
    print(f"  Gain:   +{improvement:.2f}% 🎉")
else:
    print(f"  Change: {improvement:.2f}%")

# Save best model
with open('best_model_realistic.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print(f"\n[SAVED] Tuned model saved as best_model_realistic.pkl")

# Feature importance
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

importance_df.to_csv('../data/processed/feature_importance.csv', index=False)

print(f"\n[FEATURE IMPORTANCE]")
for idx, row in importance_df.iterrows():
    print(f"  {row['Feature']}: {row['Importance']*100:.2f}%")

# Update model comparison file
results = [{
    'Model': 'Random Forest (Tuned)',
    'Train_R2': train_r2,
    'Test_R2': test_r2,
    'RMSE_Lakhs': rmse,
    'MAE_Lakhs': mae,
    'MAPE_%': mape,
    'Overfitting_Gap': train_r2 - test_r2
}]

# Add other models (dummy values to keep structure)
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
    
    results.append({
        'Model': name,
        'Train_R2': r2_score(y_train, y_train_pred),
        'Test_R2': r2_score(y_test, y_pred),
        'RMSE_Lakhs': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE_Lakhs': mean_absolute_error(y_test, y_pred),
        'MAPE_%': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
        'Overfitting_Gap': r2_score(y_train, y_train_pred) - r2_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values('Test_R2', ascending=False)
results_df.to_csv('../data/processed/model_comparison_results.csv', index=False)

print("\n" + "="*70)
print("HYPERPARAMETER TUNING COMPLETE!")
print("="*70)
