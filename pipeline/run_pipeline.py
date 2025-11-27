"""
Main Pipeline
End-to-end machine learning pipeline for real estate price prediction
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_processing.data_cleaning import load_raw_data, clean_data, save_cleaned_data
from data_processing.feature_engineering import engineer_features, remove_outliers, get_feature_columns
from modeling.train_model import prepare_data, train_all_models, evaluate_model, save_model
from visualization.generate_charts import setup_style, create_model_performance_charts, create_feature_importance_chart

def run_pipeline():
    """Execute complete ML pipeline"""
    
    print("="*70)
    print("AHMEDABAD REAL ESTATE PRICE PREDICTION - ML PIPELINE")
    print("="*70)
    
    # Paths
    raw_data_path = '../data/raw/ahmedabad_real_estate_data.csv'
    cleaned_data_path = '../data/processed/cleaned_real_estate_data.csv'
    featured_data_path = '../data/processed/featured_real_estate_data.csv'
    model_dir = '../models'
    viz_dir = '../visualizations/model_performance'
    
    # Step 1: Load Featured Data (already cleaned & featured)
    print("\n[STEP 1] Loading Featured Data...")
    df_featured = pd.read_csv(featured_data_path)
    print(f"✓ Loaded {len(df_featured)} properties with features")
    print(f"  Features: {list(get_feature_columns())}")
    # Step 2: Prepare Modeling Data
    print("\n[STEP 2] Preparing Modeling Data...")
    feature_columns = get_feature_columns()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df_featured, feature_columns)
    
    models = train_all_models(X_train, y_train)
    
    # Evaluate all models
    results = []
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        metrics['Model'] = name
        results.append(metrics)
        
        print(f"\n  {name}:")
        print(f"    Accuracy: {metrics['Test_R2']*100:.2f}%")
        print(f"    MAE: ₹{metrics['MAE_Lakhs']:.2f}L")
        
        if metrics['Test_R2'] > best_score:
            best_score = metrics['Test_R2']
            best_model = model
            best_model_name = name
    
    # Save best model
    print(f"\n✓ Best Model: {best_model_name} ({best_score*100:.2f}%)")
    save_model(best_model, scaler, feature_columns, model_dir)
    
    # Save results
    results_df = pd.DataFrame(results).sort_values('Test_R2', ascending=False)
    results_df.to_csv('../data/processed/model_comparison_results.csv', index=False)
    
    # Step 4: Generate Visualizations
    print("\n[STEP 4] Generating Visualizations...")
    setup_style()
    y_pred = best_model.predict(X_test)
    create_model_performance_charts(results_df, y_test, y_pred, viz_dir)
    create_feature_importance_chart(best_model, feature_columns, viz_dir)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print(f"Final Accuracy: {best_score*100:.2f}%")
    print("="*70)

if __name__ == "__main__":
    run_pipeline()
