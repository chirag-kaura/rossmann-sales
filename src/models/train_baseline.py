import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

def rmspe(y_true, y_pred):
    mask = y_true != 0
    return np.sqrt(np.mean(np.square((y_true[mask] - y_pred[mask]) / y_true[mask])))

def train_baseline():
    print("Loading feature dataset...")
    df = pd.read_csv('data/processed/feature_dataset.csv', parse_dates=['Date'])
    
    # Filter out closed stores for training
    df_train = df[(df['Open'] == 1) & (df['Sales'] > 0)].copy()
    
    # Select features
    features = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenMonths', 'Promo2OpenWeeks']
    
    # Simple encoding for categorical StoreType and Assortment if needed
    df_train['StoreType'] = df_train['StoreType'].astype('category').cat.codes
    df_train['Assortment'] = df_train['Assortment'].astype('category').cat.codes
    
    X = df_train[features]
    y = df_train['Sales']
    
    # Chronological or simple train-test split (train on older, validate on recent)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training baseline Random Forest model...")
    model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Validation
    preds = model.predict(X_val)
    score = rmspe(y_val, preds)
    print(f"Baseline Validation RMSPE: {score:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/baseline_model.pkl')
    print("Baseline model saved to models/baseline_model.pkl")

if __name__ == '__main__':
    train_baseline()
    