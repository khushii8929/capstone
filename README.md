# Ahmedabad Real Estate Price Prediction

Complete ML pipeline for predicting property prices in Ahmedabad using Gradient Boosting and ensemble methods.

## 🎯 Results

**Dataset**: 2,048 properties from cleaned data (68.5% retention rate)  
**Features**: 6 engineered features (no data leakage)

| Rank | Model | R² Score | MAE | RMSE |
|------|-------|----------|-----|------|
| 🥇 #1 | **Gradient Boosting** | **0.7847** | **12.56L** | **18.5L** |
| 🥈 #2 | Random Forest | 0.7807 | 12.60L | 18.58L |
| 🥉 #3 | Linear Regression | 0.4206 | 18.50L | 25.3L |
| #4 | Decision Tree | 0.4486 | 13.14L | 22.1L |

### Performance Interpretation
- **R² 0.7847**: Explains 78.47% of price variance
- **MAE 12.56L**: Average prediction error of ±₹12.56 Lakhs
- **Strong accuracy** for Ahmedabad's diverse real estate market


## 📁 Project Structure

```
Caapstone-Phase1/
├── data/
│   ├── raw/
│   │   └── ahmedabad_real_estate_data.csv      # 2,989 scraped properties
│   └── processed/
│       ├── cleaned_real_estate_data.csv        # 2,048 cleaned records
│       ├── featured_real_estate_data.csv       # With 6 ML features
│       ├── model_comparison_results.csv        # Performance metrics
│       └── feature_importance.csv              # Feature rankings
├── src/
│   ├── data_processing/
│   │   ├── data_cleaning.py         # Remove duplicates, handle nulls
│   │   └── feature_engineering.py   # Create 6 features
│   ├── modeling/
│   │   └── train_model.py           # Train 4 algorithms
│   └── visualization/
│       └── plot_results.py          # Generate charts
├── models/
│   ├── best_model_gradient_boosting.pkl    # Best model
│   ├── random_forest_model.pkl             # RF model
│   ├── scaler.pkl                          # StandardScaler
│   └── feature_names.pkl                   # Feature list
├── visualizations/
│   ├── eda/                         # 5 exploratory visualizations
│   ├── model_performance/           # 8 performance charts
│   └── master_dashboard/            # Executive summary dashboard
├── notebooks/
│   ├── 00_MASTER_PIPELINE.ipynb              # Complete workflow
│   ├── 01_data_cleaning.ipynb                # Data prep
│   ├── 02_feature_engineering.ipynb          # Feature creation
│   ├── 03_exploratory_data_analysis.ipynb    # EDA analysis
│   ├── 04_machine_learning_models.ipynb      # Model training
│   └── 05_business_insights_usecases.ipynb   # Business analysis
├── scripts/
│   ├── scraper.py                   # Web scraping (Housing.com, MagicBricks)
│   ├── run_complete_pipeline.py     # Execute all steps
│   └── generate_visualizations.py   # Create charts
├── run_pipeline.py                  # 🚀 Main entry point
└── README.md
```


## 🚀 Quick Start

### 1. Run Complete Pipeline (42 seconds)
```bash
python run_pipeline.py
```

**Output:**
```
[STEP 1/5] DATA CLEANING
  ✓ Loaded 2,048 cleaned properties

[STEP 2/5] EXPLORATORY DATA ANALYSIS
  ✓ Created 5 EDA visualizations

[STEP 3/5] FEATURE ENGINEERING
  ✓ Created 6 features

[STEP 4/5] MODEL TRAINING
  ✓ Gradient Boosting: 78.47% R² (BEST)
  ✓ Random Forest: 78.07% R²
  ✓ Linear Regression: 42.06% R²
  ✓ Decision Tree: 44.86% R²

[STEP 5/5] VISUALIZATIONS
  ✓ Generated 61+ PNG files

✅ PIPELINE COMPLETE! (42 seconds)
```

### 2. Make Predictions
```python
import pickle
import pandas as pd

# Load model and scaler
with open('models/best_model_gradient_boosting.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Property details
property_data = {
    'Area_SqFt': 1500,
    'BHK': 3,
    'Locality_Encoded': 25,           # Bopal
    'Furnishing_Encoded': 1,          # Semi-Furnished
    'Locality_Tier_Encoded': 2,       # High-End Tier
    'BHK_Tier_Interaction': 6         # 3 BHK × Tier 2
}

# Predict
features = pd.DataFrame([property_data])
features_scaled = scaler.transform(features)
predicted_price = model.predict(features_scaled)[0]

print(f"Predicted Price: ₹{predicted_price:.2f} Lakhs")
# Output: Predicted Price: ₹85.34 Lakhs
```

### 3. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```


## 🔧 Features (6 total)

### Core Features (2)
1. **Area_SqFt** - Property area in square feet (~15% importance)
2. **BHK** - Number of bedrooms: 1, 2, 3, 4+ (~22% importance)

### Encoded Features (2)
3. **Locality_Encoded** - 91 unique localities encoded numerically (~4% importance)
4. **Furnishing_Encoded** - Furnished / Semi-Furnished / Unfurnished (~3% importance)

### Engineered Features (2)
5. **Locality_Tier_Encoded** - Premium (Tier 1) / High-End (Tier 2) / Mid-Range (Tier 3) / Budget (Tier 4) (~26% importance)
6. **BHK_Tier_Interaction** - BHK × Locality Tier (**~30% importance - Most Important!**)

### Locality Tier Classification
```python
# Based on average price quartiles
Premium (Top 25%)     → Satellite, Science City, Prahlad Nagar
High-End (25-50%)     → Bodakdev, Thaltej, Bopal
Mid-Range (50-75%)    → Gota, Chandkheda, Nikol
Budget (Bottom 25%)   → Odhav, Vatva, CTM
```

**Key Innovation:** `BHK_Tier_Interaction` captures the market reality:
- 2BHK in Premium (Satellite) ≠ 4BHK in Budget (Odhav)
- Combines property size with location quality


## 📊 Data Processing

### Data Cleaning Pipeline
```
Raw Data (2,989 properties)
    ↓
Remove Duplicates (-523 rows)
    ↓
Handle Missing Values (-215 rows)
    ↓
Standardize Formats
    ↓
Remove Outliers (IQR method) (-203 rows)
    ↓
Clean Data (2,048 properties) → 68.5% retention
```

### Outlier Removal (IQR Method)
- **Price Range**: Kept 8L - 368L (removed extreme values)
- **Area Range**: Kept 200 - 8,500 sqft
- **Method**: Q1 - 1.5×IQR to Q3 + 1.5×IQR

### Feature Engineering Strategy
1. **Locality Encoding**: LabelEncoder for 91 localities
2. **Furnishing Encoding**: Furnished=2, Semi=1, Unfurnished=0
3. **Locality Tiers**: Created using price quartiles (no data leakage)
4. **Interaction Term**: Multiply BHK × Locality_Tier_Encoded

### Dataset Statistics
- **Total Properties**: 2,048
- **Unique Localities**: 91
- **Price Range**: ₹8L - ₹368L
- **Area Range**: 200 - 8,500 sqft
- **Average Price**: ~₹55 Lakhs
- **Most Common BHK**: 2 BHK (45%), 3 BHK (38%)


## 🎓 Model Details

### Gradient Boosting (Best Model)
- **n_estimators**: 400 trees
- **learning_rate**: 0.05
- **max_depth**: 7
- **R² Score**: 0.7847 (78.47%)
- **MAE**: 12.56 Lakhs
- **RMSE**: 18.5 Lakhs

### Training Configuration
- **Dataset Split**: 75% train (1,536) / 25% test (512)
- **Feature Scaling**: StandardScaler (mean=0, std=1)
- **Cross-Validation**: 5-fold CV
- **Evaluation Metrics**: R², MAE, RMSE

### Error Distribution
- **Within ±10L**: 52% of predictions
- **Within ±20L**: 78% of predictions
- **Underestimation**: 45% cases
- **Overestimation**: 55% cases

### Challenging Scenarios
- Luxury properties (>₹200L) → High variance
- 1BHK in premium areas → Limited training samples
- Under-construction properties → Market volatility


## 🔍 Top Localities (by property count)

1. **Bopal** (185 properties) - High-End Tier
2. **Shela** (142 properties) - Mid-Range Tier
3. **Gota** (128 properties) - Mid-Range Tier
4. **Chandkheda** (115 properties) - Budget Tier
5. **Satellite** (98 properties) - Premium Tier
6. **Science City** (87 properties) - Premium Tier
7. **Thaltej** (76 properties) - High-End Tier
8. **Bodakdev** (72 properties) - High-End Tier
9. **Prahlad Nagar** (65 properties) - Premium Tier
10. **Vaishno Devi** (54 properties) - Budget Tier

## 📦 Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

**Core Libraries:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization

**Data Sources:**
- Housing.com (web scraping)
- MagicBricks (web scraping)

## 🔄 Pipeline Workflow

### Step 1: Data Cleaning
```
Input:  2,989 raw properties
Tasks:  Remove duplicates, handle nulls, standardize formats
Output: 2,048 clean properties (68.5% retention)
```

### Step 2: Exploratory Data Analysis
```
Creates 5 visualizations:
  → Price distribution histogram
  → BHK distribution bar chart
  → Area vs Price scatter plot
  → Price by BHK boxplot
  → Feature correlation heatmap
```

### Step 3: Feature Engineering
```
Creates 6 features:
  → Area_SqFt, BHK (base features)
  → Locality_Encoded, Furnishing_Encoded
  → Locality_Tier_Encoded (Premium/High/Mid/Budget)
  → BHK_Tier_Interaction (interaction term)
Removes outliers using IQR method
```

### Step 4: Model Training
```
Trains 4 algorithms:
  → Gradient Boosting: 78.47% R² ✅ BEST
  → Random Forest: 78.07% R²
  → Linear Regression: 42.06% R²
  → Decision Tree: 44.86% R²
Saves: best_model.pkl, scaler.pkl, feature_names.pkl
```

### Step 5: Visualizations
```
Generates 61+ charts:
  → EDA visualizations (5)
  → Model performance (8)
  → Feature importance (1)
  → Executive dashboard (1)
  → Detailed insights (46+)
```
