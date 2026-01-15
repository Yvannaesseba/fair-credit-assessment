import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score,
    f1_score, average_precision_score,
    classification_report, confusion_matrix
)
import time

SEED = 42

print("Loading data...")
gmsc_real = pd.read_csv('Data/Processed/gmsc_cleaned.csv')
gmsc_synth = pd.read_csv('Data/Synthetic/gmsc_synthetic_ctgan.csv')

def preprocess_data(df, is_real=True):
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

X_real, y_real = preprocess_data(gmsc_real, is_real=True)
X_synth, y_synth = preprocess_data(gmsc_synth, is_real=False)

common_cols = list(set(X_real.columns) & set(X_synth.columns))
X_real = X_real[common_cols]
X_synth = X_synth[common_cols]

from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2

X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
    X_real, y_real, test_size=TEST_SIZE, stratify=y_real, random_state=SEED
)

X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(
    X_synth, y_synth, test_size=TEST_SIZE, stratify=y_synth, random_state=SEED
)

print(f"\nReal train: {X_real_train.shape}, test: {X_real_test.shape}")
print(f"Synth train: {X_synth_train.shape}, test: {X_synth_test.shape}")

imbalance_ratio = (y_real_train == 0).sum() / (y_real_train == 1).sum()
print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")

print("\n" + "="*80)
print("XGBOOST MODEL")
print("="*80)

xgb_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': imbalance_ratio,
    'tree_method': 'hist',
    'random_state': SEED,
    'n_jobs': -1
}

print("\nTraining baseline (Real)...")
model_xgb_real = XGBClassifier(**xgb_params)
model_xgb_real.fit(X_real_train, y_real_train)

print("Training experimental (Synthetic)...")
model_xgb_synth = XGBClassifier(**xgb_params)
model_xgb_synth.fit(X_synth_train, y_synth_train)

y_pred_real = model_xgb_real.predict(X_real_test)
y_proba_real = model_xgb_real.predict_proba(X_real_test)[:, 1]

y_pred_synth = model_xgb_synth.predict(X_real_test)
y_proba_synth = model_xgb_synth.predict_proba(X_real_test)[:, 1]

xgb_results = {
    'Model': ['XGB_Real', 'XGB_Synth'],
    'AUC': [
        roc_auc_score(y_real_test, y_proba_real),
        roc_auc_score(y_real_test, y_proba_synth)
    ],
    'Recall': [
        recall_score(y_real_test, y_pred_real),
        recall_score(y_real_test, y_pred_synth)
    ],
    'Precision': [
        precision_score(y_real_test, y_pred_real),
        precision_score(y_real_test, y_pred_synth)
    ],
    'F1': [
        f1_score(y_real_test, y_pred_real),
        f1_score(y_real_test, y_pred_synth)
    ],
    'AvgPrec': [
        average_precision_score(y_real_test, y_proba_real),
        average_precision_score(y_real_test, y_proba_synth)
    ]
}

xgb_df = pd.DataFrame(xgb_results)
print("\nXGBoost Results:")
print(xgb_df.to_string(index=False))

print("\n" + "="*80)
print("LOGISTIC REGRESSION MODEL")
print("="*80)

lr_pipeline_real = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1
    ))
])

lr_pipeline_synth = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1
    ))
])

print("\nTraining baseline (Real)...")
lr_pipeline_real.fit(X_real_train, y_real_train)

print("Training experimental (Synthetic)...")
lr_pipeline_synth.fit(X_synth_train, y_synth_train)

y_pred_lr_real = lr_pipeline_real.predict(X_real_test)
y_proba_lr_real = lr_pipeline_real.predict_proba(X_real_test)[:, 1]

y_pred_lr_synth = lr_pipeline_synth.predict(X_real_test)
y_proba_lr_synth = lr_pipeline_synth.predict_proba(X_real_test)[:, 1]

lr_results = {
    'Model': ['LR_Real', 'LR_Synth'],
    'AUC': [
        roc_auc_score(y_real_test, y_proba_lr_real),
        roc_auc_score(y_real_test, y_proba_lr_synth)
    ],
    'Recall': [
        recall_score(y_real_test, y_pred_lr_real),
        recall_score(y_real_test, y_pred_lr_synth)
    ],
    'Precision': [
        precision_score(y_real_test, y_pred_lr_real),
        precision_score(y_real_test, y_pred_lr_synth)
    ],
    'F1': [
        f1_score(y_real_test, y_pred_lr_real),
        f1_score(y_real_test, y_pred_lr_synth)
    ],
    'AvgPrec': [
        average_precision_score(y_real_test, y_proba_lr_real),
        average_precision_score(y_real_test, y_proba_lr_synth)
    ]
}

lr_df = pd.DataFrame(lr_results)
print("\nLogistic Regression Results:")
print(lr_df.to_string(index=False))

print("\n" + "="*80)
print("EXPLAINABLE BOOSTING MACHINE (EBM)")
print("="*80)

print("\nTraining baseline (Real)...")
model_ebm_real = ExplainableBoostingClassifier(random_state=SEED, n_jobs=-1)
model_ebm_real.fit(X_real_train, y_real_train)

print("Training experimental (Synthetic)...")
model_ebm_synth = ExplainableBoostingClassifier(random_state=SEED, n_jobs=-1)
model_ebm_synth.fit(X_synth_train, y_synth_train)

y_pred_ebm_real = model_ebm_real.predict(X_real_test)
y_proba_ebm_real = model_ebm_real.predict_proba(X_real_test)[:, 1]

y_pred_ebm_synth = model_ebm_synth.predict(X_real_test)
y_proba_ebm_synth = model_ebm_synth.predict_proba(X_real_test)[:, 1]

ebm_results = {
    'Model': ['EBM_Real', 'EBM_Synth'],
    'AUC': [
        roc_auc_score(y_real_test, y_proba_ebm_real),
        roc_auc_score(y_real_test, y_proba_ebm_synth)
    ],
    'Recall': [
        recall_score(y_real_test, y_pred_ebm_real),
        recall_score(y_real_test, y_pred_ebm_synth)
    ],
    'Precision': [
        precision_score(y_real_test, y_pred_ebm_real),
        precision_score(y_real_test, y_pred_ebm_synth)
    ],
    'F1': [
        f1_score(y_real_test, y_pred_ebm_real),
        f1_score(y_real_test, y_pred_ebm_synth)
    ],
    'AvgPrec': [
        average_precision_score(y_real_test, y_proba_ebm_real),
        average_precision_score(y_real_test, y_proba_ebm_synth)
    ]
}

ebm_df = pd.DataFrame(ebm_results)
print("\nEBM Results:")
print(ebm_df.to_string(index=False))

all_results = pd.concat([xgb_df, lr_df, ebm_df], ignore_index=True)
all_results.to_csv('Results/gmsc_model_comparison.csv', index=False)
print("\nAll results saved to Results/gmsc_model_comparison.csv")