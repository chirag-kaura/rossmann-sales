
#================== Data Validation & Engineering ================
import pandas as pd
import os
'''
# Load data
train = pd.read_csv('data/raw/train.csv', low_memory=False)
store = pd.read_csv('data/raw/store.csv')

print(f"Train shape: {train.shape}")
print(f"Store shape: {store.shape}")

# Merge datasets on Store
df = pd.merge(train, store, on='Store', how='left')
print(f"Merged shape: {df.shape}")


#=====================Data Cleaning & Preprocessing=====================
missing = df.isnull().sum()
print("Missing values per column:\n", missing[missing > 0])

'''


'''
#====================================================
def process_data():
    print("Loading raw data...")
    train = pd.read_csv('data/raw/train.csv', low_memory=False)
    store = pd.read_csv('data/raw/store.csv')

    # Merge dataset
    df = pd.merge(train, store, on='Store', how='left')

    # Handling missing values
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].max() * 2, inplace=True)
    df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    df['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)
    df['PromoInterval'].fillna('', inplace=True)


    # Date processing
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)


    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/clean_merged_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}. Shape: {df.shape}")

    

if __name__ == '__main__':
    process_data()

'''

os.chdir("rossmann-sales") if os.path.exists("rossmann-sales") else None
print("Current dir:", os.getcwd())
print("Files in data/processed:", os.listdir("data/processed") if os.path.exists("data/processed") else "No data/processed folder")