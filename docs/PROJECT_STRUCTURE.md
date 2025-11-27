# Project Structure Documentation

## Directory Structure

```
Caapstone-Phase1/
│
├── README.md                         # Main project documentation
├── requirements.txt                  # Python dependencies
│
├── config/                           # Configuration files
│   └── config.py                     # Project settings & hyperparameters
│
├── src/                              # Source code (modularized)
│   ├── data_processing/              # Data pipeline
│   │   ├── data_cleaning.py          # Raw data cleaning functions
│   │   └── feature_engineering.py    # Feature creation & transformation
│   │
│   ├── modeling/                     # ML models
│   │   └── train_model.py            # Model training & evaluation
│   │
│   ├── visualization/                # Visualization generation
│   │   └── generate_charts.py        # Chart creation functions
│   │
│   └── utils/                        # Helper utilities
│       └── helpers.py                # Common utility functions
│
├── pipeline/                         # Execution pipeline
│   └── run_pipeline.py               # Main end-to-end pipeline script
│
├── data/                             # Data storage
│   ├── raw/                          # Original scraped data
│   │   └── ahmedabad_real_estate_data.csv
│   │
│   └── processed/                    # Processed data
│       ├── cleaned_real_estate_data.csv
│       ├── featured_real_estate_data.csv
│       ├── feature_importance.csv
│       └── model_comparison_results.csv
│
├── models/                           # Trained models
│   ├── best_model.pkl                # Final Random Forest model (78.07%)
│   ├── scaler.pkl                    # Feature scaler
│   └── feature_columns.pkl           # Feature list
│
├── visualizations/                   # Generated charts
│   ├── model_performance/            # Model evaluation charts
│   ├── master_dashboard/             # Dashboard visualizations
│   └── eda/                          # Exploratory analysis charts
│
├── notebooks/                        # Jupyter notebooks (exploratory)
│   └── *.ipynb                       # Analysis notebooks
│
├── scripts/                          # Legacy scripts (kept for reference)
│   └── generate_*.py                 # Dashboard generation scripts
│
├── docs/                             # Additional documentation
├── reports/                          # Analysis reports
└── archive/                          # Old/unused files
```

## Key Files

### Core Modules

**1. src/data_processing/data_cleaning.py**
- `load_raw_data()` - Load CSV data
- `clean_data()` - Remove duplicates, handle missing values
- `parse_price()` - Convert price strings to lakhs
- `parse_area()` - Extract square feet from area strings

**2. src/data_processing/feature_engineering.py**
- `engineer_features()` - Create all 6 features
- `add_locality_tier()` - Categorize localities (Premium/High-End/Mid-Range/Budget)
- `remove_outliers()` - Remove extreme values using IQR method
- `get_feature_columns()` - Return feature list

**3. src/modeling/train_model.py**
- `prepare_data()` - Train-test split & scaling
- `train_random_forest_optimized()` - Get optimized RF model
- `train_all_models()` - Train multiple models for comparison
- `evaluate_model()` - Calculate performance metrics
- `save_model()` / `load_model()` - Model persistence

**4. src/visualization/generate_charts.py**
- `create_model_performance_charts()` - Model comparison & actual vs predicted
- `create_feature_importance_chart()` - Feature importance visualization

**5. pipeline/run_pipeline.py**
- Complete end-to-end execution
- Orchestrates all modules
- Generates final results

### Configuration

**config/config.py**
- All hyperparameters
- File paths
- Model settings
- Feature definitions

### Utilities

**src/utils/helpers.py**
- Display functions
- Formatting utilities
- Metric calculators

## Usage

### Run Complete Pipeline
```bash
cd pipeline
python run_pipeline.py
```

### Import Modules
```python
from src.data_processing.feature_engineering import engineer_features
from src.modeling.train_model import train_random_forest_optimized
from src.visualization.generate_charts import create_model_performance_charts
```

## Model Details

**Algorithm**: Gradient Boosting Regressor  
**Accuracy**: 78.47%  
**Features**: 6 (BHK×Tier, Tier, BHK, Area, Locality, Furnishing)  
**Dataset**: 2,048 properties (outliers removed)  
**Optimization**: Hyperparameter tuning + feature interaction

*Alternative: Random Forest - 78.07% accuracy*

## Data Flow

1. Raw CSV → **data_cleaning.py** → Cleaned CSV
2. Cleaned CSV → **feature_engineering.py** → Featured CSV (6 features)
3. Featured CSV → **train_model.py** → Trained Model (.pkl)
4. Model + Data → **generate_charts.py** → Visualizations (.png)

## Best Practices Used

✅ **Modular code** - Separated concerns into distinct modules  
✅ **Configuration management** - Centralized settings in config.py  
✅ **Clean functions** - Single responsibility principle  
✅ **Reusability** - Functions can be imported and reused  
✅ **Documentation** - Docstrings for all functions  
✅ **Standard structure** - Industry-standard ML project layout  

## Quick Reference

**Train new model:**
```python
from pipeline.run_pipeline import run_pipeline
run_pipeline()
```

**Load saved model:**
```python
from src.modeling.train_model import load_model
model, scaler, features = load_model('models')
```

**Make predictions:**
```python
# Prepare new data
X_new = new_data[features]
X_scaled = scaler.transform(X_new)
predictions = model.predict(X_scaled)
```

---
**Clean, Modularized, Production-Ready Structure** ✅
