import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

RANDOM_STATE = 42
TEST_SIZE = 0.2

def preprocess_gmsc(df, is_real=True):
    df = df.copy()

    # drop the row ID column if it exists, we don't need it as a feature
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    y = df['SeriousDlqin2yrs']
    X = df.drop(columns=['SeriousDlqin2yrs'])

    if is_real:
        # remove clearly wrong age values
        X.loc[X['age'] < 18, 'age'] = np.nan
        X.loc[X['age'] > 100, 'age'] = np.nan

        # negative income doesn't make sense, treat as missing
        X.loc[X['MonthlyIncome'] < 0, 'MonthlyIncome'] = np.nan

        # negative debt ratio doesn't make sense, treat as missing
        # cap extreme values at 10 to reduce outlier influence
        X.loc[X['DebtRatio'] < 0, 'DebtRatio'] = np.nan
        X.loc[X['DebtRatio'] > 10, 'DebtRatio'] = 10

        # revolving utilisation should be between 0 and 2, cap anything outside
        X.loc[X['RevolvingUtilizationOfUnsecuredLines'] < 0, 'RevolvingUtilizationOfUnsecuredLines'] = np.nan
        X.loc[X['RevolvingUtilizationOfUnsecuredLines'] > 2, 'RevolvingUtilizationOfUnsecuredLines'] = 2

    # fill missing values with column medians
    X = X.fillna(X.median())

    # catch any infinities that slipped through and fill those too
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    return X, y


if __name__ == "__main__":

    # load the raw GMSC data
    print("Loading GMSC data...")
    gmsc_real = pd.read_csv('data/cs-training.csv')

    print(f"Data shape: {gmsc_real.shape}")
    print(f"Target distribution:\n{gmsc_real['SeriousDlqin2yrs'].value_counts()}")

    # clean and split features from target
    X_real, y_real = preprocess_gmsc(gmsc_real, is_real=True)

    # stratified split so both train and test have the same default rate
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

    # save cleaned train and test sets to data folder
    os.makedirs('data', exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # also save the full cleaned dataset for CTGAN to train on
    cleaned_df = pd.concat([X_real, y_real], axis=1)

    train_df.to_csv('data/gmsc_train.csv', index=False)
    test_df.to_csv('data/gmsc_test.csv', index=False)
    cleaned_df.to_csv('data/gmsc_cleaned.csv', index=False)

    print("\nSaved to data/gmsc_train.csv, data/gmsc_test.csv, data/gmsc_cleaned.csv")