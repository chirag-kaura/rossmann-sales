import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.lightgbm
import joblib
import os

def rmspe(y_true, y_pred):
    mask = y_true != 0
    return np.sqrt(np.mean(np.square((y_true[mask] - y_pred[mask]) / y_true[mask])))

def train_with_mlflow():
    mlflow.set_experiment("rossmann-sales-forecast")
    
    with mlflow.start_run():
        print("Loading feature dataset...")
        df = pd.read_csv('data/processed/feature_dataset.csv', parse_dates=['Date'])
        
        df_train = df[(df['Open'] == 1) & (df['Sales'] > 0)].copy()
        
        features = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                    'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenMonths', 'Promo2OpenWeeks']
        
        df_train['StoreType'] = df_train['StoreType'].astype('category').cat.codes
        df_train['Assortment'] = df_train['Assortment'].astype('category').cat.codes
        
        X = df_train[features]
        y = df_train['Sales']
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'random_state': 42,
            'verbose': -1
        }
        
        # Log parameters to MLflow
        mlflow.log_params(params)
        
        print("Training LightGBM with MLflow tracking...")
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data]
        )
        
        preds = model.predict(X_val)
        score = rmspe(y_val, preds)
        
        # Log metric to MLflow
        mlflow.log_metric("rmspe", score)
        print(f"MLflow Logged RMSPE: {score:.4f}")
        
        # Log model artifact
        mlflow.lightgbm.log_model(model, "model")
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/best_model.pkl')
        print("Model saved locally and logged to MLflow.")

if __name__ == '__main__':
    train_with_mlflow()