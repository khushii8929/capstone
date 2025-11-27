# 🎯 Model Features Documentation

## Model Features (18 Total) - Advanced Feature Engineering

### **1. Area_SqFt**
- **Description**: Total built-up area of the property in square feet
- **Type**: Continuous numeric
- **Source**: Directly scraped from listings
- **Example**: 1200 sq.ft for a typical 2 BHK

### **2. BHK**
- **Description**: Number of bedrooms (Bedroom-Hall-Kitchen configuration)
- **Type**: Discrete numeric (1-5)
- **Source**: Extracted from property title using regex
- **Example**: 2.0 for "2 BHK", 3.0 for "3 BHK"

### **3. Bathrooms**
- **Description**: Number of bathrooms in the property
- **Type**: Discrete numeric (1-5)
- **Source**: Directly scraped from listings (filled intelligently for missing values)
- **Example**: 2.0 for typical 2 BHK, 3.0 for larger properties

### **4. Furnishing_Encoded**
- **Description**: Furnishing status encoded as numeric values
- **Type**: Categorical encoded (0-3)
- **Encoding**: Furnished=0, Semi-Furnished=1, Unfurnished=2, Unknown=3
- **Source**: Extracted from listing details

### **5. Locality_Encoded**
- **Description**: Neighborhood/area encoded using Label Encoding
- **Type**: Categorical encoded (0-1237)
- **Method**: Label encoding of all 1,238 unique localities in Ahmedabad
- **Example**: Bopal=120, Shela=890, Science City=650 (arbitrary numeric labels)

### **6. Area_BHK_Interaction**
- **Description**: Interaction feature = Area_SqFt × BHK
- **Type**: Continuous numeric (engineered)
- **Purpose**: **Most important feature (19.92% importance)** - Captures synergy between size and configuration
- **Example**: 1200 × 3 = 3600 for a 1200 sq.ft 3 BHK

### **7. Locality_Area**
- **Description**: Interaction feature = Locality_Encoded × Area_SqFt
- **Type**: Continuous numeric (engineered)
- **Purpose**: Captures locality-specific area premiums (premium areas with larger properties)
- **Example**: High values for large properties in premium localities (2.11% importance)

### **8. Locality_BHK**
- **Description**: Interaction feature = Locality_Encoded × BHK
- **Type**: Continuous numeric (engineered)
- **Purpose**: Captures locality-specific BHK configuration patterns
- **Example**: Certain localities may have more 3-4 BHK luxury apartments (1.85% importance)

### **9. Furnishing_Area**
- **Description**: Interaction feature = Furnishing_Encoded × Area_SqFt
- **Type**: Continuous numeric (engineered)
- **Purpose**: Captures how furnishing premium scales with property size
- **Example**: Furnished larger properties command higher premiums (1.90% importance)

### **10. Bathroom_Area**
- **Description**: Interaction feature = Bathrooms × Area_SqFt
- **Type**: Continuous numeric (engineered)
- **Purpose**: **Second most important feature (17.64% importance)** - Luxury bathroom amenities in larger properties
- **Example**: 3 bathrooms in 2000 sq.ft = 6000 luxury factor

### **11. Area_Squared**
- **Description**: Polynomial feature = Area_SqFt²
- **Type**: Continuous numeric (engineered)
- **Purpose**: Captures non-linear relationship between area and price (7.37% importance)
- **Example**: 1200² = 1,440,000 (larger properties have exponential price growth)

### **12. BHK_Squared**
- **Description**: Polynomial feature = BHK²
- **Type**: Continuous numeric (engineered)
- **Purpose**: Captures non-linear price increase for luxury configurations (8.72% importance)
- **Example**: 4² = 16 (4 BHK is disproportionately more expensive than 2 BHK)

### **13. Bathroom_Squared**
- **Description**: Polynomial feature = Bathrooms²
- **Type**: Continuous numeric (engineered)
- **Purpose**: Captures luxury bathroom premium effects (2.99% importance)
- **Example**: 3² = 9 (multiple bathrooms add exponential luxury value)

### **14. Area_per_BHK***
- **Description**: Average area per bedroom = Area_SqFt / BHK
- **Type**: Continuous numeric (engineered)
- **Purpose**: Measures spaciousness of each bedroom (2.26% importance)
- **Example**: 600 sq.ft/BHK indicates spacious rooms

### **15. Bathroom_BHK_Ratio**
- **Description**: Ratio feature = Bathrooms / BHK
- **Type**: Continuous numeric (engineered)
- **Purpose**: Luxury indicator - more bathrooms per bedroom (2.81% importance)
- **Example**: 1.0 = equal bathrooms and bedrooms (luxury), 0.67 = 2 bathrooms for 3 BHK

### **16. Area_per_Room**
- **Description**: Ratio feature = Area_SqFt / (BHK + Bathrooms)
- **Type**: Continuous numeric (engineered)
- **Purpose**: Overall spaciousness efficiency metric (2.37% importance)
- **Example**: 1200 sq.ft / (3+2) = 240 sq.ft per room

### **17. Is_Large_Property**
- **Description**: Binary flag indicating if property is in top 25% by area
- **Type**: Binary (0 or 1)
- **Threshold**: Area > 1530 sq.ft (75th percentile)
- **Purpose**: Identifies luxury/large properties (1.14% importance)

### **18. Is_Luxury_Config**
- **Description**: Binary flag for luxury configurations (≥4 BHK)
- **Type**: Binary (0 or 1)
- **Threshold**: BHK ≥ 4 (10.6% of all properties)
- **Purpose**: Identifies premium multi-bedroom properties (8.71% importance - 3rd most important!)

---

## Feature Importance Rankings (Random Forest - Actual Results):

1. **Area_BHK_Interaction**: 19.92% 🥇 (Most Important!)
2. **Bathroom_Area**: 17.64% 🥈 (Luxury amenity effect)
3. **BHK_Squared**: 8.72% 🥉
4. **Is_Luxury_Config**: 8.71%
5. **BHK**: 8.18%
6. **Area_Squared**: 7.37%
7. **Area_SqFt**: 4.91%
8. **Bathrooms**: 4.43%
9. **Bathroom_Squared**: 2.99%
10. **Bathroom_BHK_Ratio**: 2.81%
11. **Area_per_Room**: 2.37%
12. **Area_per_BHK**: 2.26%
13. **Locality_Area**: 2.11%
14. **Furnishing_Area**: 1.90%
15. **Locality_BHK**: 1.85%
16. **Locality_Encoded**: 1.79%
17. **Is_Large_Property**: 1.14%
18. **Furnishing_Encoded**: 0.89%

**Key Insights:**
- **Interaction features dominate**: Top 2 features are interactions (37.56% combined)
- **Size matters most**: Area-related features contribute ~50% total importance
- **Configuration is crucial**: BHK-related features contribute ~28% importance
- **Locality impact**: ~5.75% (Locality_Encoded + Locality_Area + Locality_BHK)

---

## Feature Engineering Pipeline

### Step 1: Base Features
- Load cleaned data with `Area_SqFt`, `BHK_Num`, `Furnishing`, `Locality`

### Step 2: Encoding
```python
# Label encode categorical features
from sklearn.preprocessing import LabelEncoder

le_furnishing = LabelEncoder()
df['Furnishing_Encoded'] = le_furnishing.fit_transform(df['Furnishing'])

le_locality = LabelEncoder()
df['Locality_Encoded'] = le_locality.fit_transform(df['Locality'])
```

### Step 3: Engineered Features
```python
# INTERACTION FEATURES (capture complex relationships)
df['Area_BHK_Interaction'] = df['Area_SqFt'] * df['BHK']  # MOST IMPORTANT!
df['Locality_Area'] = df['Locality_Encoded'] * df['Area_SqFt']
df['Locality_BHK'] = df['Locality_Encoded'] * df['BHK']
df['Furnishing_Area'] = df['Furnishing_Encoded'] * df['Area_SqFt']
df['Bathroom_Area'] = df['Bathrooms'] * df['Area_SqFt']  # 2nd MOST IMPORTANT!

# POLYNOMIAL FEATURES (non-linear effects)
df['Area_Squared'] = df['Area_SqFt'] ** 2
df['BHK_Squared'] = df['BHK'] ** 2
df['Bathroom_Squared'] = df['Bathrooms'] ** 2

# RATIO FEATURES (efficiency metrics)
df['Area_per_BHK'] = df['Area_SqFt'] / df['BHK']
df['Bathroom_BHK_Ratio'] = df['Bathrooms'] / df['BHK']
df['Total_Rooms'] = df['BHK'] + df['Bathrooms']
df['Area_per_Room'] = df['Area_SqFt'] / df['Total_Rooms']

# BINARY FLAGS (market segmentation)
df['Is_Large_Property'] = (df['Area_SqFt'] > df['Area_SqFt'].quantile(0.75)).astype(int)
df['Is_Luxury_Config'] = (df['BHK'] >= 4).astype(int)
```

---

## Model Performance

### Random Forest (Best Model) 🏆
- **R² Score**: 0.7124 (71.24%)
- **RMSE**: ±₹44.31 Lakhs
- **MAE**: ±₹25.38 Lakhs
- **MAPE**: 27.41%
- **Overfitting Gap**: 12.69% (acceptable)

### All Models Comparison:
| Model | R² Score | RMSE | MAE | Status |
|-------|----------|------|-----|--------|
| **Random Forest** | **71.24%** | **₹44.31L** | **₹25.38L** | 🏆 **Best** |
| Linear Regression | 69.14% | ₹45.90L | ₹27.50L | Good baseline |
| Gradient Boosting | 65.91% | ₹48.24L | ₹27.41L | Overfitting |
| Decision Tree | 61.62% | ₹51.18L | ₹30.32L | Overfitting |

### Training Parameters
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
```

---

## Data Leakage Prevention

### ❌ Removed Features (Data Leakage)
- `Locality_Avg_Price` - Average price per locality (derived from target)
- `Locality_Avg_PriceSqFt` - Average price/sqft per locality (derived from target)
- `Value_Score` - Difference from locality average (derived from target)
- `Locality_Price_Category` - Categorical price tier (derived from target)

### ✅ Clean Features (No Leakage)
All 16 features listed above are legitimate and would be available at prediction time for new properties.

---

## Usage Example

```python
import pandas as pd
import pickle

# Load trained model
with open('notebooks/best_model_random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare new property data (all 18 features required)
new_property = {
    'Area_SqFt': 1200,
    'BHK': 3,
    'Bathrooms': 2,
    'Furnishing_Encoded': 0,  # Furnished
    'Locality_Encoded': 120,  # Bopal
    'Area_BHK_Interaction': 3600,  # 1200 * 3
    'Locality_Area': 144000,  # 120 * 1200
    'Locality_BHK': 360,  # 120 * 3
    'Furnishing_Area': 0,  # 0 * 1200
    'Bathroom_Area': 2400,  # 2 * 1200
    'Area_Squared': 1440000,  # 1200^2
    'BHK_Squared': 9,  # 3^2
    'Bathroom_Squared': 4,  # 2^2
    'Area_per_BHK': 400,  # 1200 / 3
    'Bathroom_BHK_Ratio': 0.67,  # 2 / 3
    'Area_per_Room': 240,  # 1200 / (3+2)
    'Is_Large_Property': 0,
    'Is_Luxury_Config': 0
}

# Convert to DataFrame
df_new = pd.DataFrame([new_property])

# Predict price
predicted_price = model.predict(df_new)
print(f"Predicted Price: ₹{predicted_price[0]:.2f} Lakhs")
```

---

## Dataset Statistics

- **Total Properties**: 2,247
- **Localities**: 1,238 unique areas
- **Price Range**: ₹15L - ₹555L
- **Area Range**: 300 - 10,000 sq.ft
- **BHK Range**: 1-8 BHK configurations
- **Average Price**: ₹91.28 Lakhs
- **Median Price**: ₹69.80 Lakhs

---

## Feature Categories

### Interaction Features (5) - 43.51% Combined Importance 🔥
- `Area_BHK_Interaction` (19.92%)
- `Bathroom_Area` (17.64%)
- `Locality_Area` (2.11%)
- `Furnishing_Area` (1.90%)
- `Locality_BHK` (1.85%)

### Polynomial Features (3) - 19.08% Combined Importance
- `BHK_Squared` (8.72%)
- `Area_Squared` (7.37%)
- `Bathroom_Squared` (2.99%)

### Base Features (5) - 20.20% Combined Importance
- `BHK` (8.18%)
- `Area_SqFt` (4.91%)
- `Bathrooms` (4.43%)
- `Locality_Encoded` (1.79%)
- `Furnishing_Encoded` (0.89%)

### Ratio Features (3) - 7.44% Combined Importance
- `Bathroom_BHK_Ratio` (2.81%)
- `Area_per_Room` (2.37%)
- `Area_per_BHK` (2.26%)

### Binary Flags (2) - 9.85% Combined Importance
- `Is_Luxury_Config` (8.71%)
- `Is_Large_Property` (1.14%)

---

*Last Updated: November 27, 2025*
*Model Version: v2.0 (No Data Leakage)*
