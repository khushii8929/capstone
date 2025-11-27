# Project Configuration

# Paths
DATA_RAW = 'data/raw/ahmedabad_real_estate_data.csv'
DATA_CLEANED = 'data/processed/cleaned_real_estate_data.csv'
DATA_FEATURED = 'data/processed/featured_real_estate_data.csv'
MODEL_DIR = 'models'
VIZ_DIR = 'visualizations'

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.25
BEST_MODEL = 'gradient_boosting'  # Best performer: 78.47% vs RF: 78.07%

# Gradient Boosting Hyperparameters (Best Model - 78.47%)
GB_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'random_state': RANDOM_STATE
}

# Random Forest Hyperparameters (Alternative - 78.07%)
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 25,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Feature Columns
FEATURE_COLUMNS = [
    'Area_SqFt',
    'BHK',
    'Locality_Encoded',
    'Furnishing_Encoded',
    'Locality_Tier_Encoded',
    'BHK_Tier_Interaction'
]

TARGET_COLUMN = 'Price_Lakhs'

# Outlier Removal
OUTLIER_METHOD = 'IQR'
IQR_MULTIPLIER = 1.5

# Locality Tier Quartiles
LOCALITY_TIER_QUARTILES = [0.25, 0.50, 0.75]
LOCALITY_TIER_LABELS = ['Budget', 'Mid-Range', 'High-End', 'Premium']
