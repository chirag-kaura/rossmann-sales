import pandas as pd
import numpy as np
import os

def build_features():
    print("Loading cleaned dataset...")
    df = pd.read_csv('data/processed/clean_merged_data.csv', parse_dates=['Date'])
    
    print("Engineering features...")
    # 1. Competition Open Duration in Months
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0).astype(int)
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0).astype(int)
    
    df['CompetitionOpenMonths'] = 0
    mask = (df['CompetitionOpenSinceYear'] > 0) & (df['CompetitionOpenSinceMonth'] > 0)
    df.loc[mask, 'CompetitionOpenMonths'] = (
        (df['Date'].dt.year - df['CompetitionOpenSinceYear']) * 12 +
        (df['Date'].dt.month - df['CompetitionOpenSinceMonth'])
    )
    df['CompetitionOpenMonths'] = df['CompetitionOpenMonths'].apply(lambda x: max(0, x))
    
    # 2. Promo2 Duration in Weeks
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0).astype(int)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0).astype(int)
    
    df['Promo2OpenWeeks'] = 0
    mask_promo2 = (df['Promo2SinceYear'] > 0) & (df['Promo2SinceWeek'] > 0)
    df.loc[mask_promo2, 'Promo2OpenWeeks'] = (
        (df['Date'].dt.year - df['Promo2SinceYear']) * 52 +
        (df['Date'].dt.isocalendar().week.astype(int) - df['Promo2SinceWeek'])
    )
    df['Promo2OpenWeeks'] = df['Promo2OpenWeeks'].apply(lambda x: max(0, x))
    
    # 3. Encoding categorical variables
    df['StateHoliday'] = df['StateHoliday'].astype(str).map({'0': 0, 'a': 1, 'b': 2, 'c': 3}).fillna(0)
    
    # Save feature dataset
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/feature_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Feature dataset saved to {output_path}. Shape: {df.shape}")

if __name__ == '__main__':
    build_features()