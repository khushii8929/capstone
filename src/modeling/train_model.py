"""
Model Training Module
Trains and evaluates machine learning models
"""

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def prepare_data(df, feature_columns, target='Price_Lakhs', test_size=0.25):
    """Prepare train-test split and scaling"""
    
    X = df[feature_columns].fillna(df[feature_columns].median())
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_random_forest_optimized():
    """Get optimized Random Forest model"""
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

def train_all_models(X_train, y_train):
    """Train multiple models with optimized hyperparameters"""
    
    models = {
        'Random Forest (Optimized)': train_random_forest_optimized(),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=7,
            min_samples_split=3,
            min_samples_leaf=2,
            subsample=0.9,
            random_state=42
        ),
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42
        )
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance"""
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'Train_R2': r2_score(y_train, y_train_pred),
        'Test_R2': r2_score(y_test, y_test_pred),
        'RMSE_Lakhs': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE_Lakhs': mean_absolute_error(y_test, y_test_pred),
        'MAPE_%': np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100,
        'Overfitting_Gap': r2_score(y_train, y_train_pred) - r2_score(y_test, y_test_pred)
    }
    
    return metrics

def save_model(model, scaler, feature_columns, model_dir='../models'):
    """Save trained model and artifacts"""
    
    with open(f'{model_dir}/best_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(f'{model_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(f'{model_dir}/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    print(f" Model saved to {model_dir}/")

def load_model(model_dir='../models'):
    """Load saved model and artifacts"""
    
    with open(f'{model_dir}/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open(f'{model_dir}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(f'{model_dir}/feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    
    return model, scaler, feature_columns
