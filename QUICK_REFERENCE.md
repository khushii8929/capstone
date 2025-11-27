# Quick Reference Guide

## 🚀 Running the Project

### Option 1: Run Complete Pipeline
```bash
cd pipeline
python run_pipeline.py
```
**Output**: Trains all models, generates visualizations, saves results

### Option 2: Import and Use Modules
```python
# Load trained model
from src.modeling.train_model import load_model
model, scaler, features = load_model('models')

# Make predictions
predictions = model.predict(scaler.transform(new_data[features]))
```

---

## 📊 Key Files

| File | Purpose | Command |
|------|---------|---------|
| `pipeline/run_pipeline.py` | Run everything | `python run_pipeline.py` |
| `models/best_model.pkl` | Trained model (78.47%) | Load with pickle |
| `data/processed/featured_real_estate_data.csv` | Final dataset | 2,048 properties |
| `visualizations/model_performance/` | Charts | View PNG files |
| `config/config.py` | Settings | Edit hyperparameters |

---

## 🔧 Configuration

Edit `config/config.py` to change:
- File paths
- Model hyperparameters
- Feature columns
- Locality tier thresholds

---

## 📈 Model Performance

```
Best Model: Gradient Boosting
Accuracy: 78.47% (Highest)
MAE: ±₹12.56 Lakhs

Alternative: Random Forest
Accuracy: 78.07%
MAE: ±₹12.60 Lakhs

Dataset: 2,048 properties
Features: 6 (no data leakage)
```

---

## 🎯 Features Used

1. **BHK × Locality_Tier** (30.11%) - Interaction feature
2. **Locality_Tier** (26.29%) - Premium/High-End/Mid-Range/Budget
3. **BHK** (21.61%) - Number of bedrooms
4. **Area_SqFt** (14.85%) - Property size
5. **Locality** (4.23%) - Encoded location
6. **Furnishing** (2.91%) - Furnished/Semi/Unfurnished

---

## 📁 Directory Structure

```
src/
├── data_processing/     # Data cleaning & feature engineering
├── modeling/            # Model training & evaluation
├── visualization/       # Chart generation
└── utils/              # Helper functions

pipeline/
└── run_pipeline.py     # Main execution script

config/
└── config.py          # All settings

models/
├── best_model.pkl     # Trained model
├── scaler.pkl         # Feature scaler
└── feature_columns.pkl # Feature list

data/
└── processed/
    └── featured_real_estate_data.csv
```

---

## 💡 Common Tasks

### Train New Model
```bash
cd pipeline
python run_pipeline.py
```

### Load Existing Model
```python
import pickle
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Predict Price
```python
from src.modeling.train_model import load_model

model, scaler, features = load_model('models')

# Prepare new data
new_property = {
    'Area_SqFt': 1200,
    'BHK': 3,
    'Locality_Encoded': 15,
    'Furnishing_Encoded': 1,
    'Locality_Tier_Encoded': 2,
    'BHK_Tier_Interaction': 6  # BHK * Tier
}

X = [[new_property[f] for f in features]]
X_scaled = scaler.transform(X)
price = model.predict(X_scaled)[0]
print(f"Predicted Price: ₹{price:.2f} Lakhs")
```

### Generate Visualizations
```python
from src.visualization.generate_charts import create_model_performance_charts

create_model_performance_charts(
    y_test, y_pred, 
    model_names, accuracies,
    'visualizations/model_performance'
)
```

---

## 🐛 Troubleshooting

### Issue: Module not found
**Solution**: Run from project root or add to path:
```python
import sys
sys.path.append('..')
```

### Issue: File not found
**Solution**: Check paths in `config/config.py` are correct

### Issue: Low accuracy
**Solution**: Ensure using featured data with all 6 features

---

## 📚 Module Reference

### data_processing/data_cleaning.py
- `load_raw_data()` - Load CSV
- `clean_data()` - Remove invalid entries
- `parse_price()` - Convert price strings
- `parse_area()` - Extract square feet

### data_processing/feature_engineering.py
- `engineer_features()` - Create base features
- `add_locality_tier()` - Add tier classification
- `remove_outliers()` - IQR outlier removal
- `get_feature_columns()` - Return feature list

### modeling/train_model.py
- `train_random_forest_optimized()` - Get tuned RF model
- `train_all_models()` - Train multiple models
- `evaluate_model()` - Calculate metrics
- `save_model()` / `load_model()` - Persistence

### visualization/generate_charts.py
- `create_model_performance_charts()` - Comparison plots
- `create_feature_importance_chart()` - Feature importance

---

## 🎯 Best Practices

✅ **Always use featured data** - 6 features, no data leakage  
✅ **Load saved model** - Don't retrain unnecessarily  
✅ **Check feature order** - Must match training order  
✅ **Scale new data** - Use saved scaler  
✅ **Validate inputs** - Check for missing/invalid values  

---

## 📞 Quick Commands

```bash
# Run pipeline
python pipeline/run_pipeline.py

# Install dependencies
pip install -r requirements.txt

# Check data
python -c "import pandas as pd; print(pd.read_csv('data/processed/featured_real_estate_data.csv').shape)"

# List features
python -c "from config.config import FEATURE_COLUMNS; print(FEATURE_COLUMNS)"
```

---

**Project Status**: ✅ Production Ready  
**Accuracy**: 78.47%  
**Structure**: Fully Modularized  
