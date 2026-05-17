import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

if __name__ == "__main__":

    print("Home Credit Data Preparation")
    print("="*60)

    # load the raw application table
    print("\n[1/4] Cleaning application table...")
    app_train = pd.read_csv('data/application_train.csv')

    # remove extreme outliers that would skew the models
    app_train = app_train[app_train['AMT_INCOME_TOTAL'] < 1e7].copy()
    app_train = app_train[app_train['AMT_CREDIT'] < 5e6].copy()
    app_train = app_train[app_train['AMT_GOODS_PRICE'] < 5e6].copy()

    # positive DAYS_EMPLOYED values are a known data error in this dataset
    app_train.loc[app_train['DAYS_EMPLOYED'] > 0, 'DAYS_EMPLOYED'] = np.nan

    # XNA is not a real gender, treat as missing
    app_train.loc[app_train['CODE_GENDER'] == 'XNA', 'CODE_GENDER'] = np.nan

    # drop columns with more than 80% missing values
    missing_pct = app_train.isnull().sum() / len(app_train)
    high_missing = missing_pct[missing_pct > 0.8].index.tolist()

    # drop columns where one value dominates more than 95% of rows
    low_variance_cols = []
    for col in app_train.columns:
        if app_train[col].dtype == 'object':
            value_counts = app_train[col].value_counts(normalize=True)
            if len(value_counts) > 0 and value_counts.iloc[0] > 0.95:
                low_variance_cols.append(col)

    cols_to_drop = list(set(high_missing + low_variance_cols))
    cols_to_drop = [c for c in cols_to_drop if c not in ['SK_ID_CURR', 'TARGET']]

    app_train_clean = app_train.drop(columns=cols_to_drop)

    print(f"  Original shape: {app_train.shape}")
    print(f"  Cleaned shape: {app_train_clean.shape}")
    print(f"  Dropped {len(cols_to_drop)} columns")

    # load the bureau table
    print("\n[2/4] Cleaning bureau table...")
    bureau = pd.read_csv('data/bureau.csv')

    # remove extreme credit amounts
    bureau = bureau[bureau['AMT_CREDIT_SUM'] < 5e6].copy()
    bureau = bureau[bureau['AMT_CREDIT_SUM_DEBT'] < 5e6].copy()

    # positive days values are data errors
    bureau.loc[bureau['DAYS_CREDIT'] > 0, 'DAYS_CREDIT'] = np.nan
    bureau.loc[bureau['DAYS_CREDIT_ENDDATE'] > 0, 'DAYS_CREDIT_ENDDATE'] = np.nan

    print(f"  Bureau shape: {bureau.shape}")

    # aggregate bureau data per applicant
    print("\n[3/4] Aggregating bureau data...")
    agg_specs = {
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'AMT_CREDIT_SUM': ['sum', 'mean', 'max'],
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean', 'max'],
        'AMT_CREDIT_SUM_LIMIT': ['sum', 'mean'],
        'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean', 'max'],
        'CREDIT_TYPE': ['nunique'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
        'AMT_ANNUITY': ['mean', 'max']
    }

    bureau_agg = bureau.groupby('SK_ID_CURR').agg(agg_specs)
    bureau_agg.columns = ['_'.join(col).upper() for col in bureau_agg.columns]
    bureau_agg = bureau_agg.reset_index()

    # add loan counts per applicant
    bureau_counts = bureau.groupby('SK_ID_CURR').size().reset_index(name='BUREAU_LOAN_COUNT')
    bureau_agg = bureau_agg.merge(bureau_counts, on='SK_ID_CURR', how='left')

    bureau_active = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').size().reset_index(name='BUREAU_ACTIVE_LOANS')
    bureau_closed = bureau[bureau['CREDIT_ACTIVE'] == 'Closed'].groupby('SK_ID_CURR').size().reset_index(name='BUREAU_CLOSED_LOANS')

    bureau_agg = bureau_agg.merge(bureau_active, on='SK_ID_CURR', how='left')
    bureau_agg = bureau_agg.merge(bureau_closed, on='SK_ID_CURR', how='left')

    bureau_agg['BUREAU_ACTIVE_LOANS'] = bureau_agg['BUREAU_ACTIVE_LOANS'].fillna(0)
    bureau_agg['BUREAU_CLOSED_LOANS'] = bureau_agg['BUREAU_CLOSED_LOANS'].fillna(0)

    print(f"  Aggregated features: {bureau_agg.shape[1] - 1}")

    # merge application and bureau tables
    print("\n[4/4] Merging application and bureau tables...")
    merged = app_train_clean.merge(bureau_agg, on='SK_ID_CURR', how='left')

    # applicants with no bureau history get 0 for bureau features
    bureau_cols = [c for c in bureau_agg.columns if c != 'SK_ID_CURR']
    for col in bureau_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    print(f"  Final merged shape: {merged.shape}")
    print(f"  Default rate: {merged['TARGET'].mean():.4f}")

    # save merged dataset
    merged.to_csv('data/home_credit_merged.csv', index=False)

    print("\n" + "="*60)
    print("Done.")
    print(f"  Rows: {merged.shape[0]:,}")
    print(f"  Features: {merged.shape[1] - 2}")
    print(f"  Saved to data/home_credit_merged.csv")
    print("="*60)