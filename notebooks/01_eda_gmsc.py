import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

RANDOM_STATE = 42
TEST_SIZE = 0.2

def preprocess_gmsc(df, is_real=True):
    df = df.copy()
    
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    
    y = df['SeriousDlqin2yrs']
    X = df.drop(columns=['SeriousDlqin2yrs'])
    
    if is_real:
        X.loc[X['age'] < 18, 'age'] = np.nan
        X.loc[X['age'] > 100, 'age'] = np.nan
        X.loc[X['MonthlyIncome'] < 0, 'MonthlyIncome'] = np.nan
        X.loc[X['DebtRatio'] < 0, 'DebtRatio'] = np.nan
        X.loc[X['DebtRatio'] > 10, 'DebtRatio'] = 10
        X.loc[X['RevolvingUtilizationOfUnsecuredLines'] < 0, 'RevolvingUtilizationOfUnsecuredLines'] = np.nan
        X.loc[X['RevolvingUtilizationOfUnsecuredLines'] > 2, 'RevolvingUtilizationOfUnsecuredLines'] = 2
    
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    
    return X, y

if __name__ == "__main__":
    print("Loading GMSC data...")
    gmsc_real = pd.read_csv('Data/Processed/gmsc_cleaned.csv')
    
    print(f"Data shape: {gmsc_real.shape}")
    print(f"Target distribution:\n{gmsc_real['SeriousDlqin2yrs'].value_counts()}")
    
    X_real, y_real = preprocess_gmsc(gmsc_real, is_real=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real,
        test_size=TEST_SIZE,
        stratify=y_real,
        random_state=RANDOM_STATE
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Train default rate: {y_train.mean():.4f}")
    print(f"Test default rate: {y_test.mean():.4f}")
    
    os.makedirs('Data/Processed', exist_ok=True)
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv('Data/Processed/gmsc_train.csv', index=False)
    test_df.to_csv('Data/Processed/gmsc_test.csv', index=False)
    
    print("\nSaved processed data to Data/Processed/")