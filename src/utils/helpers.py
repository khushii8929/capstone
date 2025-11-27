"""
Utility Functions
Helper functions used across the project
"""

import pandas as pd
import numpy as np

def display_dataframe_info(df, name="Dataset"):
    """Display useful information about a dataframe"""
    print(f"\n{name} Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing: {df.isnull().sum().sum()}")
    print(f"  Duplicates: {df.duplicated().sum()}")

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(title)
    print("="*70)

def format_price(price_lakhs):
    """Format price in lakhs to readable string"""
    if price_lakhs >= 100:
        return f"₹{price_lakhs/100:.2f} Cr"
    else:
        return f"₹{price_lakhs:.2f} L"

def calculate_metrics_summary(y_true, y_pred):
    """Calculate and return all metrics in a dictionary"""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    return {
        'R2_Score': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

def print_metrics(metrics, title="Model Performance"):
    """Print metrics in a formatted way"""
    print(f"\n{title}:")
    print(f"  Accuracy (R²): {metrics['R2_Score']*100:.2f}%")
    print(f"  MAE: ₹{metrics['MAE']:.2f} Lakhs")
    print(f"  RMSE: ₹{metrics['RMSE']:.2f} Lakhs")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
