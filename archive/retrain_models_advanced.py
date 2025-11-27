import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TRAINING MODELS WITH 18 ADVANCED FEATURES")
print("="*80)

# Load Featured Data
df = pd.read_csv('../data/processed/featured_real_estate_data.csv')
print(f"\n✓ Dataset loaded: {df.shape}")

# ============================================================================
# DEFINE 18 MODEL FEATURES (NO DATA LEAKAGE)
# ============================================================================
feature_columns = [
    # Base Features (5)
    'Area_SqFt',
    'BHK',
    'Bathrooms',
    'Furnishing_Encoded',
    'Locality_Encoded',
    
    # Interaction Features (5) - Capture complex relationships
    'Area_BHK_Interaction',      # PRIMARY FEATURE
    'Locality_Area',
    'Locality_BHK',
    'Furnishing_Area',
    'Bathroom_Area',
    
    # Polynomial Features (3) - Non-linear effects
    'Area_Squared',
    'BHK_Squared',
    'Bathroom_Squared',
    
    # Ratio Features (3) - Efficiency metrics
    'Area_per_BHK',
    'Bathroom_BHK_Ratio',
    'Area_per_Room',
    
    # Binary Flags (2) - Market segmentation
    'Is_Large_Property',
    'Is_Luxury_Config'
]

target_column = 'Price_Lakhs'

print(f"\n📊 Model Features: {len(feature_columns)} features")
for i, feat in enumerate(feature_columns, 1):
    print(f"   {i:2d}. {feat}")

# Create feature matrix and target vector
X = df[feature_columns].copy()
y = df[target_column].copy()

# Handle any NaN values
X = X.fillna(X.median())

print(f"\n✓ Feature matrix: {X.shape}")
print(f"✓ Target vector: {y.shape}")
print(f"✓ Missing values: {X.isna().sum().sum()}")

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Data Split:")
print(f"   Training: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Testing:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# FEATURE SCALING
# ============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Features scaled using StandardScaler")

# ============================================================================
# MODEL TRAINING
# ============================================================================
print(f"\n{'='*80}")
print("TRAINING 4 REGRESSION MODELS")
print("="*80)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=5,
        random_state=42
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
}

results = []

for name, model in models.items():
    print(f"\n{'─'*80}")
    print(f"🤖 Training: {name}")
    print(f"{'─'*80}")
    
    # Train model
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
    print(f"   Training R²:  {train_r2:.4f} ({train_r2*100:.2f}%)")
    print(f"   Testing R²:   {test_r2:.4f} ({test_r2*100:.2f}%)")
    print(f"   RMSE:         ₹{rmse:.2f} Lakhs")
    print(f"   MAE:          ₹{mae:.2f} Lakhs")
    print(f"   MAPE:         {mape:.2f}%")
    
    # Check for overfitting
    overfit_diff = train_r2 - test_r2
    if overfit_diff > 0.10:
        print(f"   ⚠️  Overfitting detected: {overfit_diff*100:.2f}% gap")
    else:
        print(f"   ✓ Good generalization: {overfit_diff*100:.2f}% gap")
    
    results.append({
        'Model': name,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'RMSE_Lakhs': rmse,
        'MAE_Lakhs': mae,
        'MAPE_%': mape,
        'Overfitting_Gap': overfit_diff
    })
    
    # Save model
    model_filename = f"best_model_{name.lower().replace(' ', '_')}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✓ Saved: {model_filename}")

# ============================================================================
# RESULTS COMPARISON
# ============================================================================
print(f"\n{'='*80}")
print("📊 MODEL COMPARISON RESULTS")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test_R2', ascending=False)

print(f"\n{results_df.to_string(index=False)}")

# Identify best model
best_model_name = results_df.iloc[0]['Model']
best_r2 = results_df.iloc[0]['Test_R2']
best_rmse = results_df.iloc[0]['RMSE_Lakhs']

print(f"\n{'='*80}")
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"{'='*80}")
print(f"   R² Score:  {best_r2:.4f} ({best_r2*100:.2f}%)")
print(f"   RMSE:      ₹{best_rmse:.2f} Lakhs")
print(f"   Features:  {len(feature_columns)} advanced features")

# ============================================================================
# FEATURE IMPORTANCE (for tree-based models)
# ============================================================================
if best_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
    print(f"\n{'='*80}")
    print(f"📊 FEATURE IMPORTANCE ({best_model_name})")
    print(f"{'='*80}\n")
    
    best_model = models[best_model_name]
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    # Save feature importance
    feature_importance.to_csv('../data/processed/feature_importance.csv', index=False)
    print(f"\n✓ Feature importance saved to: feature_importance.csv")

# ============================================================================
# SAVE ARTIFACTS
# ============================================================================
print(f"\n{'='*80}")
print("💾 SAVING MODEL ARTIFACTS")
print("="*80)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved: scaler.pkl")

# Save feature columns
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("✓ Saved: feature_columns.pkl")

# Save model info
model_info = {
    'best_model': best_model_name,
    'test_r2': best_r2,
    'rmse': best_rmse,
    'features': feature_columns,
    'n_features': len(feature_columns),
    'dataset_size': len(df),
    'train_size': len(X_train),
    'test_size': len(X_test)
}

with open('model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("✓ Saved: model_info.pkl")

# Save results comparison
results_df.to_csv('../data/processed/model_comparison_results.csv', index=False)
print("✓ Saved: model_comparison_results.csv")

# ============================================================================
# TEST PREDICTION
# ============================================================================
print(f"\n{'='*80}")
print("🔮 SAMPLE PREDICTION TEST")
print("="*80)

sample_property = X_test.iloc[0:1].copy()
print(f"\nSample Property Features:")
print(f"   Area: {sample_property['Area_SqFt'].values[0]:.0f} sq.ft")
print(f"   BHK: {sample_property['BHK'].values[0]:.0f}")
print(f"   Bathrooms: {sample_property['Bathrooms'].values[0]:.0f}")
print(f"   Area_BHK_Interaction: {sample_property['Area_BHK_Interaction'].values[0]:.0f}")

actual_price = y_test.iloc[0]
if best_model_name == 'Linear Regression':
    predicted_price = models[best_model_name].predict(scaler.transform(sample_property))[0]
else:
    predicted_price = models[best_model_name].predict(sample_property)[0]

error_pct = abs(predicted_price - actual_price) / actual_price * 100

print(f"\n   Actual Price:     ₹{actual_price:.2f} Lakhs")
print(f"   Predicted Price:  ₹{predicted_price:.2f} Lakhs")
print(f"   Error:            {error_pct:.2f}%")

if error_pct < 10:
    print(f"   ✓ Excellent prediction!")
elif error_pct < 20:
    print(f"   ✓ Good prediction")
else:
    print(f"   ⚠️  Prediction needs improvement")

print(f"\n{'='*80}")
print("✅ MODEL TRAINING COMPLETE!")
print("="*80)
print(f"\n📈 Performance Improvement:")
print(f"   • 18 advanced features engineered")
print(f"   • Best R²: {best_r2*100:.2f}%")
print(f"   • RMSE: ±₹{best_rmse:.2f} Lakhs")
print(f"   • No data leakage - production ready!")
print(f"\n{'='*80}")
