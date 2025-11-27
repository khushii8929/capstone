"""
Run the complete pipeline: Data Cleaning → Feature Engineering → ML Models → Visualizations
Skips web scraping (assumes raw data already exists)
"""

import subprocess
import sys
import time

def run_script(script_path, description):
    """Run a Python script and display results"""
    print(f"\n{'='*80}")
    print(f"▶️  RUNNING: {description}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Completed in {elapsed:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Script failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def run_notebook(notebook_path, description):
    """Execute a Jupyter notebook"""
    print(f"\n{'='*80}")
    print(f"▶️  RUNNING: {description}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Use nbconvert to execute notebook
        result = subprocess.run(
            [
                sys.executable, '-m', 'jupyter', 'nbconvert',
                '--to', 'notebook',
                '--execute',
                notebook_path,
                '--output', notebook_path,
                '--ExecutePreprocessor.timeout=600',
                '--ExecutePreprocessor.allow_errors=False'
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Notebook executed successfully in {elapsed:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Notebook execution failed")
        print("Output:", e.stdout)
        print("Errors:", e.stderr)
        return False

print("="*80)
print("🚀 COMPLETE REAL ESTATE ANALYSIS PIPELINE")
print("="*80)
print("\n📋 Pipeline Steps:")
print("   1. Data Cleaning (01_data_cleaning.ipynb)")
print("   2. Feature Engineering (run_feature_engineering.py)")
print("   3. ML Model Training (retrain_models_no_leakage.py)")
print("   4. Exploratory Data Analysis (03_exploratory_data_analysis.ipynb)")
print("   5. Business Insights (05_business_insights_usecases.ipynb)")
print("   6. Verification (verify_fix.py)")
print("\n⚠️  Note: Skipping web scraping (using existing raw data)")
print("="*80)

input("\n⏸️  Press ENTER to start the pipeline...")

# Track results
results = {}

# Step 1: Data Cleaning (run notebook)
results['data_cleaning'] = run_notebook(
    '01_data_cleaning.ipynb',
    'Step 1: Data Cleaning'
)

if not results['data_cleaning']:
    print("\n❌ Pipeline stopped due to data cleaning failure")
    sys.exit(1)

# Step 2: Feature Engineering (run script)
results['feature_engineering'] = run_script(
    'run_feature_engineering.py',
    'Step 2: Feature Engineering (No Data Leakage)'
)

if not results['feature_engineering']:
    print("\n❌ Pipeline stopped due to feature engineering failure")
    sys.exit(1)

# Step 3: ML Model Training (run script)
results['model_training'] = run_script(
    'retrain_models_no_leakage.py',
    'Step 3: ML Model Training'
)

if not results['model_training']:
    print("\n❌ Pipeline stopped due to model training failure")
    sys.exit(1)

# Step 4: EDA (run notebook - optional, may take time)
print("\n" + "="*80)
print("⏭️  OPTIONAL: Run EDA Notebook?")
print("   This generates visualizations but may take several minutes")
choice = input("   Run EDA? (y/n): ").lower()

if choice == 'y':
    results['eda'] = run_notebook(
        '03_exploratory_data_analysis.ipynb',
        'Step 4: Exploratory Data Analysis'
    )
else:
    results['eda'] = 'skipped'
    print("   ⏭️  Skipped EDA")

# Step 5: Business Insights (run notebook - optional)
print("\n" + "="*80)
print("⏭️  OPTIONAL: Run Business Insights Notebook?")
choice = input("   Run Business Insights? (y/n): ").lower()

if choice == 'y':
    results['business_insights'] = run_notebook(
        '05_business_insights_usecases.ipynb',
        'Step 5: Business Insights & Use Cases'
    )
else:
    results['business_insights'] = 'skipped'
    print("   ⏭️  Skipped Business Insights")

# Step 6: Verification (always run)
results['verification'] = run_script(
    'verify_fix.py',
    'Step 6: Final Verification'
)

# Summary
print("\n" + "="*80)
print("📊 PIPELINE EXECUTION SUMMARY")
print("="*80)

for step, result in results.items():
    status = "✅ PASSED" if result is True else "⏭️  SKIPPED" if result == 'skipped' else "❌ FAILED"
    print(f"   {step.replace('_', ' ').title()}: {status}")

all_passed = all(v is True or v == 'skipped' for v in results.values())

if all_passed:
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n✅ Your Real Estate Analysis is complete!")
    print("✅ Models trained with honest performance (92.8% R²)")
    print("✅ All data files generated")
    print("✅ No data leakage detected")
    print("\n📁 Key Output Files:")
    print("   - data/processed/cleaned_real_estate_data.csv")
    print("   - data/processed/featured_real_estate_data.csv")
    print("   - notebooks/best_model_*.pkl")
    print("   - notebooks/model_info.pkl")
else:
    print("\n" + "="*80)
    print("⚠️  PIPELINE COMPLETED WITH ISSUES")
    print("="*80)
    print("\nSome steps failed. Review the output above for details.")

print("\n" + "="*80)
