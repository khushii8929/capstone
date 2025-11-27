"""
Script to execute all visualization notebooks
Runs: EDA, Business Insights, Model Visualizations, Advanced Visualizations
"""

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
from datetime import datetime

def run_notebook(notebook_path):
    """Execute a Jupyter notebook"""
    notebook_name = os.path.basename(notebook_path)
    print(f"\n{'='*70}")
    print(f"🚀 Running: {notebook_name}")
    print(f"{'='*70}")
    
    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Configure executor
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        
        # Execute notebook
        start_time = datetime.now()
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        end_time = datetime.now()
        
        # Save executed notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        duration = (end_time - start_time).total_seconds()
        print(f"✅ {notebook_name} completed in {duration:.1f} seconds")
        return True
        
    except Exception as e:
        print(f"❌ Error in {notebook_name}: {str(e)}")
        return False

def main():
    print("\n" + "="*70)
    print("📊 VISUALIZATION NOTEBOOKS EXECUTION")
    print("="*70)
    
    # Define notebooks to run
    notebooks = [
        '03_exploratory_data_analysis.ipynb',
        '05_business_insights_usecases.ipynb',
        '06_model_visualizations_summary.ipynb',
        '07_Advanced_Real_Estate_Visualizations.ipynb'
    ]
    
    results = {}
    
    # Run each notebook
    for notebook in notebooks:
        notebook_path = os.path.join(os.path.dirname(__file__), notebook)
        if os.path.exists(notebook_path):
            results[notebook] = run_notebook(notebook_path)
        else:
            print(f"⚠️  Notebook not found: {notebook}")
            results[notebook] = False
    
    # Summary
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
    else:
        print("\n⚠️  Some notebooks failed - check errors above")

if __name__ == "__main__":
    main()
