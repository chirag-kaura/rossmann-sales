import pandas as pd
import numpy as np
import joblib
import os

def generate_predictions():
    print("Loading test dataset and best model...")
    test = pd.read_csv('data/raw/test.csv', parse_dates=['Date'])
    store = pd.read_csv('data/raw/store.csv')
    
    df_test = pd.merge(test, store, on='Store', how='left')
    
    # Handle missing values
    df_test['CompetitionDistance'].fillna(df_test['CompetitionDistance'].max() * 2, inplace=True)
    df_test['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    df_test['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    df_test['Promo2SinceWeek'].fillna(0, inplace=True)
    df_test['Promo2SinceYear'].fillna(0, inplace=True)
    df_test['PromoInterval'].fillna('', inplace=True)
    
    # Date processing
    df_test['Year'] = df_test['Date'].dt.year
    df_test['Month'] = df_test['Date'].dt.month
    df_test['Day'] = df_test['Date'].dt.day
    df_test['DayOfWeek'] = df_test['Date'].dt.dayofweek
    df_test['WeekOfYear'] = df_test['Date'].dt.isocalendar().week.astype(int)
    
    # Feature engineering for test
    df_test['CompetitionOpenSinceYear'] = df_test['CompetitionOpenSinceYear'].astype(int)
    df_test['CompetitionOpenSinceMonth'] = df_test['CompetitionOpenSinceMonth'].astype(int)
    df_test['CompetitionOpenMonths'] = 0
    mask = (df_test['CompetitionOpenSinceYear'] > 0) & (df_test['CompetitionOpenSinceMonth'] > 0)
    df_test.loc[mask, 'CompetitionOpenMonths'] = (
        (df_test['Date'].dt.year - df_test['CompetitionOpenSinceYear']) * 12 +
        (df_test['Date'].dt.month - df_test['CompetitionOpenSinceMonth'])
    )
    df_test['CompetitionOpenMonths'] = df_test['CompetitionOpenMonths'].apply(lambda x: max(0, x))
    
    df_test['Promo2SinceYear'] = df_test['Promo2SinceYear'].astype(int)
    df_test['Promo2SinceWeek'] = df_test['Promo2SinceWeek'].astype(int)
    df_test['Promo2OpenWeeks'] = 0
    mask_promo2 = (df_test['Promo2SinceYear'] > 0) & (df_test['Promo2SinceWeek'] > 0)
    df_test.loc[mask_promo2, 'Promo2OpenWeeks'] = (
        (df_test['Date'].dt.year - df_test['Promo2SinceYear']) * 52 +
        (df_test['Date'].dt.isocalendar().week.astype(int) - df_test['Promo2SinceWeek'])
    )
    df_test['Promo2OpenWeeks'] = df_test['Promo2OpenWeeks'].apply(lambda x: max(0, x))
    
    df_test['StateHoliday'] = df_test['StateHoliday'].astype(str).map({'0': 0, 'a': 1, 'b': 2, 'c': 3}).fillna(0)
    
    features = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenMonths', 'Promo2OpenWeeks']
    
    df_test['StoreType'] = df_test['StoreType'].astype('category').cat.codes
    df_test['Assortment'] = df_test['Assortment'].astype('category').cat.codes
    
    model = joblib.load('models/best_model.pkl')
    
    # Predict
    preds = model.predict(df_test[features])
    
    # Handle stores that are closed (Open == 0) -> Sales = 0
    preds[df_test['Open'] == 0] = 0
    
    submission = pd.DataFrame({
        'Id': df_test['Id'],
        'Sales': preds
    })
    
    os.makedirs('data/predictions', exist_ok=True)
    submission.to_csv('data/predictions/submission.csv', index=False)
    print("Submission file generated successfully at data/predictions/submission.csv!")

if __name__ == '__main__':
    generate_predictions()