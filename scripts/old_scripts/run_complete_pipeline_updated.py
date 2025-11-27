"""
Updated Complete Pipeline - Uses Clean Features (No Data Leakage)
Runs all steps: Data Cleaning -> Feature Engineering -> ML Training -> EDA -> Verification
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import subprocess
import os
from datetime import datetime
import time

print("="*80)
print("AHMEDABAD REAL ESTATE ANALYTICS - COMPLETE PIPELINE (UPDATED)")
print("="*80)
print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Change to notebooks directory
os.chdir('../notebooks')

# ============================================================================
# STEP 1: DATA CLEANING
# ============================================================================
print("\n" + "="*80)
print("STEP 1: DATA CLEANING")
print("="*80)

start_time = time.time()
try:
    result = subprocess.run(
        ['jupyter', 'nbconvert', '--to', 'notebook', '--execute', 
         '01_data_cleaning.ipynb', '--output', '01_data_cleaning_executed.ipynb'],
        capture_output=True,
        text=True,
        timeout=120
    )
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"[OK] Data cleaning completed successfully in {elapsed:.1f} seconds")
    else:
        print(f"[ERROR] Data cleaning failed:")
        print(result.stderr)
except Exception as e:
    print(f"[ERROR] Data cleaning error: {e}")

# ============================================================================
# STEP 2: FEATURE ENGINEERING (NO LEAKAGE)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: FEATURE ENGINEERING (CLEAN FEATURES)")
print("="*80)

start_time = time.time()
try:
    result = subprocess.run(
        ['python', 'run_feature_engineering.py'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=120
    )
    elapsed = time.time() - start_time
    
    print(result.stdout)
    if result.returncode == 0:
        print(f"\n[OK] Feature engineering completed successfully in {elapsed:.1f} seconds")
    else:
        print(f"[ERROR] Feature engineering failed:")
        print(result.stderr)
except Exception as e:
    print(f"[ERROR] Feature engineering error: {e}")

# ============================================================================
# STEP 3: MACHINE LEARNING MODEL TRAINING (NO LEAKAGE)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: ML MODEL TRAINING (CLEAN FEATURES)")
print("="*80)

start_time = time.time()
try:
    result = subprocess.run(
        ['python', 'retrain_models_no_leakage.py'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=180
    )
    elapsed = time.time() - start_time
    
    print(result.stdout)
    if result.returncode == 0:
        print(f"\n[OK] Model training completed successfully in {elapsed:.1f} seconds")
    else:
        print(f"[ERROR] Model training failed:")
        print(result.stderr)
except Exception as e:
    print(f"[ERROR] Model training error: {e}")

# ============================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("="*80)

start_time = time.time()
try:
    result = subprocess.run(
        ['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
         '03_exploratory_data_analysis.ipynb', '--output', '03_exploratory_data_analysis_executed.ipynb'],
        capture_output=True,
        text=True,
        timeout=180
    )
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"[OK] EDA completed successfully in {elapsed:.1f} seconds")
    else:
        print(f"[WARN] EDA had issues (may still have generated visualizations):")
        print(result.stderr[:500])
except Exception as e:
    print(f"[WARN] EDA error: {e}")

# ============================================================================
# STEP 5: BUSINESS INSIGHTS & USE CASES
# ============================================================================
print("\n" + "="*80)
print("STEP 5: BUSINESS INSIGHTS & USE CASES")
print("="*80)

start_time = time.time()
try:
    result = subprocess.run(
        ['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
         '05_business_insights_usecases.ipynb', '--output', '05_business_insights_usecases_executed.ipynb'],
        capture_output=True,
        text=True,
        timeout=180
    )
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"[OK] Business insights completed successfully in {elapsed:.1f} seconds")
    else:
        print(f"[WARN] Business insights had issues:")
        print(result.stderr[:500])
except Exception as e:
    print(f"[WARN] Business insights error: {e}")

# ============================================================================
# STEP 6: FINAL VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("STEP 6: FINAL VERIFICATION")
print("="*80)

start_time = time.time()
try:
    result = subprocess.run(
        ['python', 'verify_fix.py'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=60
    )
    elapsed = time.time() - start_time
    
    print(result.stdout)
    if result.returncode == 0:
        print(f"\n[OK] Verification completed successfully in {elapsed:.1f} seconds")
    else:
        print(f"[ERROR] Verification failed:")
        print(result.stderr)
except Exception as e:
    print(f"[ERROR] Verification error: {e}")

# ============================================================================
# PIPELINE SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PIPELINE EXECUTION SUMMARY")
print("="*80)
print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n[INFO] Check the notebooks/ directory for:")
print("  - Model files (.pkl)")
print("  - Executed notebooks (*_executed.ipynb)")
print("  - Generated visualizations")
print("\n[INFO] Check the data/processed/ directory for:")
print("  - cleaned_real_estate_data.csv")
print("  - featured_real_estate_data.csv")
print("  - model_comparison_results.csv")
print("="*80)
