"""
Home Credit Data Preparation Pipeline
Combines Application, Bureau cleaning and merging operations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path('/path/to/data')  # UPDATE THIS
RAW_DIR = BASE_DIR / 'raw'
PROCESSED_DIR = BASE_DIR / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Home Credit Data Preparation Pipeline")
print("="*60)

# ===== PART 1: APPLICATION TABLE CLEANING =====
print("\n[1/4] Cleaning Application table...")

app_train = pd.read_csv(RAW_DIR / 'application_train.csv')

# Remove extreme outliers
app_train = app_train[app_train['AMT_INCOME_TOTAL'] < 1e7].copy()
app_train = app_train[app_train['AMT_CREDIT'] < 5e6].copy()
app_train = app_train[app_train['AMT_GOODS_PRICE'] < 5e6].copy()

# Fix data quality issues
app_train.loc[app_train['DAYS_EMPLOYED'] > 0, 'DAYS_EMPLOYED'] = np.nan
app_train.loc[app_train['CODE_GENDER'] == 'XNA', 'CODE_GENDER'] = np.nan

# Remove columns with >80% missing or low variance
missing_pct = app_train.isnull().sum() / len(app_train)
high_missing = missing_pct[missing_pct > 0.8].index.tolist()

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

app_train_clean.to_csv(PROCESSED_DIR / 'application_train_cleaned.csv', index=False)

# ===== PART 2: BUREAU TABLE CLEANING =====
print("\n[2/4] Cleaning Bureau table...")

bureau = pd.read_csv(RAW_DIR / 'bureau.csv')

# Remove outliers
bureau = bureau[bureau['AMT_CREDIT_SUM'] < 5e6].copy()
bureau = bureau[bureau['AMT_CREDIT_SUM_DEBT'] < 5e6].copy()

# Fix negative days
bureau.loc[bureau['DAYS_CREDIT'] > 0, 'DAYS_CREDIT'] = np.nan
bureau.loc[bureau['DAYS_CREDIT_ENDDATE'] > 0, 'DAYS_CREDIT_ENDDATE'] = np.nan

print(f"  Bureau shape: {bureau.shape}")

bureau.to_csv(PROCESSED_DIR / 'bureau_cleaned.csv', index=False)

# ===== PART 3: BUREAU AGGREGATION =====
print("\n[3/4] Aggregating Bureau data...")

# Aggregation specifications
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

# Aggregate
bureau_agg = bureau.groupby('SK_ID_CURR').agg(agg_specs)
bureau_agg.columns = ['_'.join(col).upper() for col in bureau_agg.columns]
bureau_agg = bureau_agg.reset_index()

# Add counts
bureau_counts = bureau.groupby('SK_ID_CURR').size().reset_index(name='BUREAU_LOAN_COUNT')
bureau_agg = bureau_agg.merge(bureau_counts, on='SK_ID_CURR', how='left')

# Active/Closed loan counts
bureau_active = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').size().reset_index(name='BUREAU_ACTIVE_LOANS')
bureau_closed = bureau[bureau['CREDIT_ACTIVE'] == 'Closed'].groupby('SK_ID_CURR').size().reset_index(name='BUREAU_CLOSED_LOANS')

bureau_agg = bureau_agg.merge(bureau_active, on='SK_ID_CURR', how='left')
bureau_agg = bureau_agg.merge(bureau_closed, on='SK_ID_CURR', how='left')

bureau_agg['BUREAU_ACTIVE_LOANS'] = bureau_agg['BUREAU_ACTIVE_LOANS'].fillna(0)
bureau_agg['BUREAU_CLOSED_LOANS'] = bureau_agg['BUREAU_CLOSED_LOANS'].fillna(0)

print(f"  Aggregated features: {bureau_agg.shape[1] - 1}")

bureau_agg.to_csv(PROCESSED_DIR / 'bureau_aggregated.csv', index=False)

# ===== PART 4: MERGE TABLES =====
print("\n[4/4] Merging Application + Bureau...")

# Merge
merged = app_train_clean.merge(bureau_agg, on='SK_ID_CURR', how='left')

# Fill missing bureau features with 0 (indicates no bureau history)
bureau_cols = [c for c in bureau_agg.columns if c != 'SK_ID_CURR']
for col in bureau_cols:
    if col in merged.columns:
        merged[col] = merged[col].fillna(0)

print(f"  Final merged shape: {merged.shape}")
print(f"  Default rate: {merged['TARGET'].mean():.4f}")

merged.to_csv(PROCESSED_DIR / 'home_credit_merged.csv', index=False)

print("\n" + "="*60)
print("Data preparation complete!")
print(f"Final dataset: {merged.shape}")
print(f"  Rows: {merged.shape[0]:,}")
print(f"  Features: {merged.shape[1] - 2} (excluding ID and TARGET)")
print(f"\nSaved to: {PROCESSED_DIR}")
print("="*60)