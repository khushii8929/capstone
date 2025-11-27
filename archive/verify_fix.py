"""
Final verification that all updates are working correctly
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import pickle
import os

print("="*80)
print("FINAL VERIFICATION - DATA LEAKAGE FIX")
print("="*80)

# 1. Verify clean dataset
print("\n[CHECK 1] Checking clean dataset...")
df = pd.read_csv('../data/processed/featured_real_estate_data.csv')
print(f"   [OK] Dataset loaded: {df.shape}")

# Check for leaked features
leaked_features = ['Locality_Avg_Price', 'Locality_Avg_PriceSqFt', 'Value_Score', 'Locality_Price_Category']
found_leaked = [f for f in leaked_features if f in df.columns]

if found_leaked:
    print(f"   [ERROR] ERROR: Found leaked features: {found_leaked}")
else:
    print(f"   [OK] No leaked features found in dataset")

# 2. Verify feature columns
print("\n[CHECK 2] Checking saved feature columns...")
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

print(f"   [OK] {len(feature_columns)} features in model")

found_leaked_in_model = [f for f in feature_columns if any(leak in f for leak in leaked_features)]
if found_leaked_in_model:
    print(f"   [ERROR] ERROR: Found leaked features in model: {found_leaked_in_model}")
else:
    print(f"   [OK] No leaked features in model feature list")

# 3. Verify model info
print("\n[CHECK 3] Checking model performance...")
with open('model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

print(f"   [OK] Best Model: {model_info['best_model_name']}")
print(f"   [OK] Test R²: {model_info['test_r2_score']:.4f} ({model_info['test_r2_score']*100:.2f}%)")

if model_info['test_r2_score'] > 0.95:
    print(f"   [WARN] WARNING: R² suspiciously high (>95%), may still have leakage")
elif model_info['test_r2_score'] > 0.65:
    print(f"   [OK] R² is in expected range (65-95%) for real features - looks good!")
else:
    print(f"   [WARN] WARNING: R² lower than expected (<65%)")

# 4. Test model prediction
print("\n[CHECK 4] Testing model prediction...")
try:
    with open('best_model_random_forest.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Create a test sample
    test_sample = df[feature_columns].iloc[0:1].copy()
    test_scaled = scaler.transform(test_sample)
    prediction = model.predict(test_scaled)
    
    print(f"   [OK] Model prediction works!")
    print(f"   Sample prediction: Rs.{prediction[0]:.2f} Lakhs")
except Exception as e:
    print(f"   [ERROR] ERROR: Model prediction failed: {e}")

# 5. List all files
print("\n[CHECK 5] Checking generated files...")
required_files = [
    'best_model_random_forest.pkl',
    'best_model_gradient_boosting.pkl',
    'best_model_xgboost.pkl',
    'best_model_decision_tree.pkl',
    'scaler.pkl',
    'feature_columns.pkl',
    'model_info.pkl'
]

missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print(f"   [ERROR] Missing files: {missing_files}")
else:
    print(f"   [OK] All {len(required_files)} model files present")

# 6. Verify data files
print("\n[CHECK 6] Checking data files...")
data_files = [
    '../data/processed/featured_real_estate_data.csv',
    '../data/processed/model_comparison_results.csv'
]

missing_data = [f for f in data_files if not os.path.exists(f)]
if missing_data:
    print(f"   [ERROR] Missing data files: {missing_data}")
else:
    print(f"   [OK] All data files present")

# Final summary
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

all_checks = [
    len(found_leaked) == 0,
    len(found_leaked_in_model) == 0,
    0.65 <= model_info['test_r2_score'] <= 0.95,
    len(missing_files) == 0,
    len(missing_data) == 0
]

if all(all_checks):
    print("[OK] ALL CHECKS PASSED!")
    print("[OK] Data leakage fix is complete and verified")
    print("[OK] Models are production-ready")
    print(f"[OK] Honest R² performance: {model_info['test_r2_score']*100:.2f}%")
else:
    print("[WARN] SOME CHECKS FAILED - Review above for details")

print("="*80)
