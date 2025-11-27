"""
Prepare environment and run all visualization notebooks
Copies necessary files and executes notebooks with proper paths
"""

import shutil
import os
from pathlib import Path

def prepare_environment():
    """Copy required files to notebooks directory"""
    print("\n" + "="*70)
    print("📂 PREPARING ENVIRONMENT")
    print("="*70)
    
    # Define source and destination paths
    data_dir = Path('../data/processed')
    notebooks_dir = Path('.')
    
    files_to_copy = [
        ('featured_real_estate_data.csv', data_dir),
        ('model_comparison_results.csv', data_dir),
        ('feature_importance.csv', data_dir),
    ]
    
    pkl_files = [
        'best_model_random_forest.pkl',
        'best_model_gradient_boosting.pkl',
        'best_model_xgboost.pkl',
        'best_model_decision_tree.pkl',
        'feature_scaler.pkl',
        'feature_columns.pkl',
        'model_info.pkl'
    ]
    
    # Copy CSV files
    for filename, source_dir in files_to_copy:
        source = source_dir / filename
        dest = notebooks_dir / filename
        if source.exists():
            shutil.copy2(source, dest)
            print(f"✅ Copied: {filename}")
        else:
            print(f"⚠️  Not found: {filename}")
    
    # Copy PKL files from notebooks directory itself (they're already there)
    for filename in pkl_files:
        source = notebooks_dir / filename
        if source.exists():
            print(f"✅ Found: {filename}")
        else:
            print(f"⚠️  Missing: {filename}")
    
    print(f"\n✅ Environment preparation complete!")
    return True

def run_notebooks():
    """Execute visualization notebooks"""
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    from datetime import datetime
    
    print("\n" + "="*70)
    print("📊 EXECUTING VISUALIZATION NOTEBOOKS")
    print("="*70)
    
    notebooks = [
        '03_exploratory_data_analysis.ipynb',
        '05_business_insights_usecases.ipynb',
        '06_model_visualizations_summary.ipynb',
        '07_Advanced_Real_Estate_Visualizations.ipynb'
    ]
    
    results = {}
    
    for notebook_name in notebooks:
        print(f"\n{'='*70}")
        print(f"🚀 Running: {notebook_name}")
        print(f"{'='*70}")
        
        try:
            # Read notebook
            with open(notebook_name, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Configure executor
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            
            # Execute
            start_time = datetime.now()
            ep.preprocess(nb, {'metadata': {'path': '.'}})
            end_time = datetime.now()
            
            # Save executed notebook
            with open(notebook_name, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            
            duration = (end_time - start_time).total_seconds()
            print(f"✅ {notebook_name} completed in {duration:.1f} seconds")
            results[notebook_name] = True
            
        except Exception as e:
            print(f"❌ Error in {notebook_name}:")
            print(f"   {str(e)[:200]}")
            results[notebook_name] = False
    
    return results

def cleanup():
    """Remove temporary copied files"""
    print("\n" + "="*70)
    print("🧹 CLEANING UP TEMPORARY FILES")
    print("="*70)
    
    temp_files = [
        'featured_real_estate_data.csv',
        'model_comparison_results.csv',
        'feature_importance.csv'
    ]
    
    for filename in temp_files:
        filepath = Path(filename)
        if filepath.exists():
            filepath.unlink()
            print(f"🗑️  Removed: {filename}")

def main():
    print("\n" + "="*70)
    print("🎨 REAL ESTATE VISUALIZATION GENERATOR")
    print("="*70)
    
    # Step 1: Prepare environment
    if not prepare_environment():
        print("\n❌ Environment preparation failed!")
        return
    
    # Step 2: Run notebooks
    results = run_notebooks()
    
    # Step 3: Cleanup
    cleanup()
    
    # Step 4: Summary
    print("\n" + "="*70)
    print("📊 EXECUTION SUMMARY")
    print("="*70)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for notebook, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {notebook}")
    
    print(f"\n📈 Results: {success_count}/{total_count} notebooks executed successfully")
    
    if success_count == total_count:
        print("\n🎉 ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("\n📁 Check the following for output:")
        print("   • notebooks/ - Updated notebooks with executed cells")
        print("   • visualizations/ - Generated charts and graphs")
    else:
        print("\n⚠️  Some notebooks had issues - they may need manual execution")

if __name__ == "__main__":
    main()
