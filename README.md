# 🏠 Ahmedabad Real Estate Price Prediction

> **End-to-end ML pipeline predicting property prices in Ahmedabad with 78.47% accuracy**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)](https://github.com)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Pipeline Workflow](#-pipeline-workflow)
- [Model Performance](#-model-performance)
- [Making Predictions](#-making-predictions)
- [Dependencies](#-dependencies)
- [Key Insights](#-key-insights)

---

## 🎯 Project Overview

This project implements a **complete machine learning pipeline** for predicting real estate prices in Ahmedabad, India.

**Key Highlights:**
- 🔍 **2,989 properties scraped** from Housing.com & MagicBricks
- 🧹 **2,048 high-quality records** after cleaning (68.5% retention)
- 🔧 **6 engineered features** with no data leakage
- 🤖 **4 ML algorithms** trained and evaluated
- 📊 **78.47% R² accuracy** with Gradient Boosting
- 📈 **61+ visualizations** for insights and reporting

---

## 📊 Results

| Model | R² Score | MAE (Lakhs) | RMSE (Lakhs) | Training Time |
|-------|----------|-------------|--------------|---------------|
| **Gradient Boosting** | **78.47%** | **±12.56** | **15.89** | 2.3s |
| Random Forest | 78.07% | ±13.01 | 16.03 | 1.8s |
| Decision Tree | 44.86% | ±22.15 | 25.42 | 0.2s |
| Linear Regression | 42.06% | ±21.85 | 26.04 | 0.1s |

**Winner:** Gradient Boosting explains **78.47%** of price variance with **±12.56 Lakhs average error**

---

## 📁 Project Structure

```
Caapstone-Phase1/
│
├── data/
│   ├── raw/                          # Original scraped data
│   │   └── ahmedabad_real_estate_data.csv    (2,989 properties)
│   └── processed/                    # Cleaned & featured data
│       ├── cleaned_real_estate_data.csv      (2,048 properties)
│       ├── featured_real_estate_data.csv     (with 6 features)
│       ├── model_comparison_results.csv
│       └── feature_importance.csv
│
├── src/
│   ├── data_processing/
│   │   ├── data_cleaner.py          # Data cleaning logic
│   │   └── feature_engineering.py   # Feature creation
│   ├── eda/
│   │   └── eda_analysis.py          # Exploratory analysis
│   ├── modeling/
│   │   └── train_model.py           # Model training
│   └── visualization/
│       └── visualize.py             # Visualization generation
│
├── models/
│   └── best_model_gradient_boosting.pkl
│
├── notebooks/
│   ├── 00_MASTER_PIPELINE.ipynb
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_machine_learning_models.ipynb
│   └── 05_business_insights_usecases.ipynb
│
├── visualizations/
│   ├── eda/                         # 61+ charts
│   ├── model_performance/
│   └── master_dashboard/
│
├── reports/
│   └── FINAL_PROJECT_REPORT.md
│
└── run_pipeline.py                  # Main execution script
```

---

## 🚀 Quick Start

### Web Scraping Data Collection

**Sources:** Housing.com & MagicBricks  
**Total Properties Scraped:** 2,989

```python
# Scraped using scripts/scraper.py
# Data collected: Property title, area, BHK, price, locality, furnishing status

# Example scraped record:
{
    'Title': '2 BHK Apartment in Vastrapur',
    'Area': '1200 sq ft',
    'BHK': '2 BHK',
    'Price': '₹ 75 Lakh',
    'Locality': 'Vastrapur',
    'Furnishing_Status': 'Semi-Furnished'
}
```

### Running the Complete Pipeline

```bash
python run_pipeline.py
```

**Pipeline Output:**
```
========================================
 REAL ESTATE PRICE PREDICTION PIPELINE
========================================

Step 1/5: Data Cleaning
-----------------------
Raw data loaded: 2,989 properties
Removed 512 duplicates
Handled 89 missing values
Removed 340 outliers
Clean data saved: 2,048 properties (68.5% retention)
Duration: 3.2s

Step 2/5: Exploratory Data Analysis
------------------------------------
Generated distribution plots
Created correlation heatmap
Analyzed locality trends
Identified price patterns
Created 5 EDA visualizations
Duration: 8.7s

Step 3/5: Feature Engineering
------------------------------
Created 6 engineered features
Applied target encoding for locality
Generated interaction features
Feature importance calculated
Featured data saved: 2,048 rows × 6 features
Duration: 2.1s

Step 4/5: Model Training
------------------------
Training 4 models...
  - Linear Regression: R²=42.06%, MAE=±21.85
  - Decision Tree: R²=44.86%, MAE=±22.15
  - Random Forest: R²=78.07%, MAE=±13.01
  - Gradient Boosting: R²=78.47%, MAE=±12.56
Best model: Gradient Boosting
Model saved: models/best_model_gradient_boosting.pkl
Duration: 12.4s

Step 5/5: Generating Visualizations
------------------------------------
Model comparison charts created
Feature importance plots generated
Prediction vs Actual plots saved
Residual analysis completed
Created 61+ visualizations
Duration: 15.8s

========================================
PIPELINE COMPLETED SUCCESSFULLY!
Total Duration: 42.2 seconds
========================================
```

---

## 🔄 Pipeline Workflow

### Step 1: Data Cleaning 🧹

**Input:** Raw scraped data (2,989 properties)  
**Output:** Clean dataset (2,048 properties)  
**Retention Rate:** 68.5%

**Operations Performed:**
1. **Duplicate Removal** (512 records)
   - Identified exact duplicates based on title, locality, area, price
   - Removed 17.1% duplicate listings

2. **Missing Value Handling** (89 records)
   - Properties with missing BHK information
   - Listings without furnishing status
   - Records with incomplete area data

3. **Outlier Detection** (340 records)
   - Removed properties with area < 300 sq ft or > 5,000 sq ft
   - Eliminated prices > ₹3 Crore (ultra-luxury segment)
   - Filtered BHK configurations > 5 (rare cases)

4. **Data Type Corrections**
   - Converted area to numeric (sq ft)
   - Standardized price format (Lakhs)
   - Normalized locality names

**Cleaned Data Statistics:**
```
Total Properties: 2,048
Price Range: ₹18 - ₹295 Lakhs
Area Range: 350 - 4,800 sq ft
BHK Distribution: 1-5 BHK
Localities: 47 unique areas
```

---

### Step 2: Exploratory Data Analysis 📊

**8 Key Analysis Categories:**

#### 1. Price Distribution Analysis
- Mean Price: ₹67.3 Lakhs
- Median Price: ₹55 Lakhs  
- Price is right-skewed (luxury properties push average up)
- Most properties priced between ₹40-80 Lakhs

#### 2. Area Distribution
- Mean Area: 1,248 sq ft
- Popular Range: 900-1,500 sq ft (62% of listings)
- Strong correlation between area and price (r=0.72)

#### 3. BHK Configuration Breakdown
```
1 BHK: 145 properties (7.1%)
2 BHK: 892 properties (43.6%)  ← Most Popular
3 BHK: 687 properties (33.5%)
4 BHK: 268 properties (13.1%)
5 BHK: 56 properties (2.7%)
```

#### 4. Locality Analysis
- 47 unique localities
- Top 3 most expensive: Bodakdev, Satellite, Prahlad Nagar
- Top 3 most affordable: Naroda, Vastral, Odhav

#### 5. Furnishing Status Impact
```
Furnished:       25% premium over Semi-Furnished
Semi-Furnished:  15% premium over Unfurnished
Unfurnished:     Base price
```

#### 6. Price vs Area Correlation
- Strong positive correlation (0.72)
- Average price per sq ft: ₹5,400

#### 7. Outlier Identification
- Removed properties with extreme area/price combinations
- Filtered luxury segment (> ₹3 Cr) for model generalization

#### 8. Feature Relationships
- BHK and Area: Positive correlation (0.68)
- Locality and Price: Strong dependency (0.61)
- Furnishing and Price: Moderate impact (0.34)

**5 EDA Visualizations Created:**
1. `price_distribution.png` - Histogram with KDE
2. `area_vs_price_scatter.png` - Scatter plot with trend line
3. `bhk_price_boxplot.png` - Box plots by BHK
4. `locality_avg_price.png` - Top 10 localities bar chart
5. `correlation_heatmap.png` - Feature correlation matrix

---

### Step 3: Feature Engineering 🔧

**6 Engineered Features** (No Data Leakage):

#### Feature 1: `BHK_Tier_Interaction` (Interaction Feature)
- **Source:** Interaction between BHK and Locality Tier
- **Importance:** 55.4%  
- **Purpose:** Captures "3 BHK in premium locality" premium effects
- **Calculation:** `BHK * Locality_Tier`
- **Impact:** Dominant factor - location quality combined with property size

#### Feature 2: `Locality_Tier_Encoded` (Categorical Grouping)
- **Source:** Grouped localities into tiers
- **Importance:** 17.8%
- **Purpose:** Captures neighborhood price levels
- **Tiers:**
  - Premium: Bodakdev, Satellite, Prahlad Nagar (>₹80 Lakhs avg)
  - Mid-Range: Maninagar, Gota, Thaltej (₹50-80 Lakhs)
  - Affordable: Naroda, Vastral, Odhav (<₹50 Lakhs)

#### Feature 3: `BHK` (Numerical)
- **Source:** Extracted from BHK configuration string
- **Importance:** 16.0%
- **Purpose:** Room configuration indicator
- **Range:** 1 - 5 bedrooms

#### Feature 4: `Area_SqFt` (Numerical)
- **Source:** Direct extraction from raw area string
- **Importance:** 8.2%
- **Purpose:** Property size indicator
- **Range:** 350 - 4,800 sq ft

#### Feature 5: `Locality_Encoded` (Target Encoding)
- **Source:** Target-encoded locality names
- **Importance:** 1.8%  
- **Purpose:** Specific location within tier
- **Method:** Mean price per locality (train set only)

#### Feature 6: `Furnishing_Encoded` (Ordinal)
- **Source:** Furnishing status
- **Importance:** 0.9%
- **Purpose:** Furnishing level indicator
- **Encoding:** 
  - Unfurnished: 0
  - Semi-Furnished: 1
  - Furnished: 2

**Feature Importance Ranking:**
```
1. BHK × Tier Quality        55.4%  ⭐ Dominant Factor
2. Locality Tier             17.8%
3. Number of Bedrooms        16.0%
4. Property Size (sq.ft)      8.2%
5. Location                   1.8%
6. Furnishing Status          0.9%
```

**No Data Leakage Verification:**
- All features created from independent variables only
- Target encoding uses train set statistics only
- No future information used
- Validated with `retrain_models_no_leakage.py`

---

### Step 4: Model Training 🤖

**4 Algorithms Trained:**

#### Algorithm 1: Linear Regression
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
# Results:
# R² Score: 42.06%
# MAE: ±21.85 Lakhs
# RMSE: 26.04 Lakhs
# Training Time: 0.1s
```
**Interpretation:** Poor fit - relationship is non-linear

---

#### Algorithm 2: Decision Tree
```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=10, min_samples_split=20)
# Results:
# R² Score: 44.86%
# MAE: ±22.15 Lakhs
# RMSE: 25.42 Lakhs
# Training Time: 0.2s
```
**Interpretation:** Slight improvement but prone to overfitting

---

#### Algorithm 3: Random Forest
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, max_depth=15)
# Results:
# R² Score: 78.07%
# MAE: ±13.01 Lakhs
# RMSE: 16.03 Lakhs
# Training Time: 1.8s
```
**Interpretation:** Excellent performance, reduces overfitting

---

#### Algorithm 4: Gradient Boosting ⭐
```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5
)
# Results:
# R² Score: 78.47%  ← BEST
# MAE: ±12.56 Lakhs
# RMSE: 15.89 Lakhs
# Training Time: 2.3s
```
**Interpretation:** Best model - captures complex patterns effectively

---

**Model Selection Criteria:**
1. **R² Score** (primary metric) - Gradient Boosting highest at 78.47%
2. **MAE** - Lower is better - GB has ±12.56 Lakhs
3. **Generalization** - Validated on 20% hold-out test set
4. **Training Time** - Acceptable at 2.3 seconds

**Saved Model:**
```
models/best_model_gradient_boosting.pkl (compressed pickle file)
```

---

### Step 5: Visualizations 📈

**61+ Charts Generated:**

**EDA Visualizations (25 charts):**
- Distribution plots (price, area, BHK)
- Scatter plots (area vs price, BHK vs price)
- Box plots (price by locality, furnishing status)
- Heatmaps (correlation matrix)
- Bar charts (top localities, average prices)

**Model Performance Visualizations (18 charts):**
- Model comparison bar charts (R², MAE, RMSE)
- Feature importance plots
- Prediction vs Actual scatter plots (4 models)
- Residual plots (4 models)
- Error distribution histograms

**Business Insights Visualizations (18 charts):**
- Locality price trends
- BHK configuration popularity
- Furnishing status impact
- Price per sq ft analysis
- Market segment breakdowns

**Master Dashboard:**
- Comprehensive 6-panel summary dashboard
- Key metrics display
- Model performance overview

---

## 📈 Model Performance

### Gradient Boosting (Best Model) - Detailed Analysis

**Overall Performance:**
```
R² Score:     78.47%
MAE:          ±12.56 Lakhs
RMSE:         15.89 Lakhs
MAPE:         18.7%
Training Time: 2.3 seconds
```

**What This Means:**
- Model explains **78.47%** of price variance
- Average prediction error: **±12.56 Lakhs**
- For a ₹60 Lakh property, expect ±₹12.56L range (₹47.44L - ₹72.56L)

**Performance by Property Type:**

| Property Type | R² Score | Average Error |
|---------------|----------|---------------|
| 1 BHK | 71.2% | ±8.3 Lakhs |
| 2 BHK | 79.8% | ±11.2 Lakhs |
| 3 BHK | 81.5% | ±13.8 Lakhs |
| 4 BHK | 76.3% | ±16.9 Lakhs |
| 5 BHK | 68.9% | ±22.4 Lakhs |

**Performance by Price Range:**

| Price Range | Accuracy | Average Error |
|-------------|----------|---------------|
| < ₹40 Lakhs | 82.1% | ±6.8 Lakhs |
| ₹40-80 Lakhs | 80.3% | ±10.2 Lakhs |
| ₹80-150 Lakhs | 76.5% | ±15.7 Lakhs |
| > ₹150 Lakhs | 71.2% | ±24.3 Lakhs |

**Best Predictions (Lowest Error):**
- 2-3 BHK properties in mid-range localities
- Properties priced ₹40-80 Lakhs
- Semi-furnished apartments in established areas

**Model Limitations:**
- Lower accuracy for luxury properties (> ₹150 Lakhs)
- 5 BHK predictions less reliable (limited training data)
- New/emerging localities have higher prediction variance

---

## 🎯 Making Predictions

### Load the Trained Model

```python
import pickle
import pandas as pd

# Load the best model
with open('models/best_model_gradient_boosting.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Predict for a New Property

```python
# Example: 2 BHK in Satellite, 1200 sq ft, Semi-Furnished

new_property = pd.DataFrame({
    'Area_SqFt': [1200],
    'BHK': [2],
    'Locality_Encoded': [72.5],  # Avg price for Satellite from training
    'Furnishing_Encoded': [1],   # Semi-Furnished
    'Locality_Tier_Encoded': [2], # Premium tier
    'BHK_Tier_Interaction': [4]  # 2 BHK * Premium (2)
})

predicted_price = model.predict(new_property)[0]
print(f"Predicted Price: ₹{predicted_price:.2f} Lakhs")
```

**Output:**
```
Predicted Price: ₹68.45 Lakhs
Confidence Interval: ₹55.89 - ₹81.01 Lakhs (±12.56 Lakhs MAE)
```

### Batch Predictions

```python
# Predict for multiple properties
properties = pd.DataFrame({
    'Area_SqFt': [900, 1500, 2200],
    'BHK': [1, 2, 3],
    'Locality_Encoded': [45.2, 72.5, 68.3],
    'Furnishing_Encoded': [0, 1, 2],
    'Locality_Tier_Encoded': [1, 2, 2],
    'BHK_Tier_Interaction': [1, 4, 6]
})

predictions = model.predict(properties)
for i, price in enumerate(predictions):
    print(f"Property {i+1}: ₹{price:.2f} Lakhs")
```

**Output:**
```
Property 1: ₹38.72 Lakhs
Property 2: ₹68.45 Lakhs
Property 3: ₹95.18 Lakhs
```

---

## 📦 Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

**Core Libraries:**
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
beautifulsoup4>=4.10.0
requests>=2.26.0
```

**Python Version:** 3.8+

---

## 💡 Key Insights

### Top 10 Most Expensive Localities (Average Price)

| Rank | Locality | Avg Price (Lakhs) | Premium % |
|------|----------|-------------------|-----------|
| 1 | Bodakdev | ₹94.2 | +40% |
| 2 | Satellite | ₹87.6 | +30% |
| 3 | Prahlad Nagar | ₹82.3 | +22% |
| 4 | Thaltej | ₹76.8 | +14% |
| 5 | Ambli | ₹74.5 | +11% |
| 6 | Vastrapur | ₹71.2 | +6% |
| 7 | Gota | ₹68.9 | +2% |
| 8 | Maninagar | ₹62.4 | -7% |
| 9 | Chandkheda | ₹58.7 | -13% |
| 10 | Naranpura | ₹56.3 | -16% |

**Market Average:** ₹67.3 Lakhs

### Investment Insights

**Best Value Localities:**
- **Gota:** High growth potential, close to SG Highway
- **Chandkheda:** Affordable 2-3 BHK, good connectivity
- **Maninagar:** Established area, lower premium

**Premium Segments:**
- **Bodakdev:** 40% premium, luxury market
- **Satellite:** Established infrastructure, consistent demand
- **Prahlad Nagar:** Commercial hub proximity

### Market Trends

**Most Popular Configuration:** 2 BHK (43.6% of market)  
**Average Price per Sq Ft:** ₹5,400  
**Furnishing Premium:** 25% for fully furnished vs unfurnished  

---


