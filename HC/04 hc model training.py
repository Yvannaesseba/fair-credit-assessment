"""
Home Credit Model Training (TSTR Framework)
Trains models on real and synthetic data, evaluates on real test set
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost.callback import EarlyStopping
from interpret.glassbox import ExplainableBoostingClassifier

warnings.filterwarnings("ignore")

# Configuration
RANDOM_STATE = 42
BASE_DIR = Path('/path/to/dissertation')  # UPDATE THIS
FS_DIR = BASE_DIR / 'Feature_Selection_Results'
CTGAN_DIR = BASE_DIR / 'Results' / 'CTGAN'
RESULTS_DIR = BASE_DIR / 'Results' / 'Model_Training'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
train_real = pd.read_csv(FS_DIR / 'train_ctgan_ready.csv')
val_real = pd.read_csv(FS_DIR / 'val_ctgan_ready.csv')
test_real = pd.read_csv(FS_DIR / 'test_ctgan_ready.csv')
train_synth = pd.read_csv(CTGAN_DIR / 'synth_train_ctgan.csv')

# Identify column types
categorical_cols = [c for c in train_real.columns if train_real[c].dtype == 'object' and c != 'TARGET']

# Preprocessing function
def preprocess_for_training(train_df, val_df, test_df, categorical_cols):
    """Encode categoricals, handle missing values"""
    
    train_proc = train_df.copy()
    val_proc = val_df.copy()
    test_proc = test_df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        train_proc[col] = le.fit_transform(train_proc[col].astype(str))
        
        val_proc[col] = val_proc[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
        test_proc[col] = test_proc[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
        
        if 'Unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'Unknown')
        
        val_proc[col] = le.transform(val_proc[col].astype(str))
        test_proc[col] = le.transform(test_proc[col].astype(str))
        label_encoders[col] = le
    
    # Separate target
    X_train = train_proc.drop('TARGET', axis=1)
    y_train = train_proc['TARGET']
    X_val = val_proc.drop('TARGET', axis=1)
    y_val = val_proc['TARGET']
    X_test = test_proc.drop('TARGET', axis=1)
    y_test = test_proc['TARGET']
    
    return X_train, y_train, X_val, y_val, X_test, y_test, label_encoders

# Prepare datasets
X_train_real, y_train_real, X_val_real, y_val_real, X_test_real, y_test_real, le_real = \
    preprocess_for_training(train_real, val_real, test_real, categorical_cols)

X_train_synth, y_train_synth, _, _, _, _, le_synth = \
    preprocess_for_training(train_synth, val_real, test_real, categorical_cols)

# Evaluation function
def evaluate_model(model, X, y):
    """Evaluate model performance"""
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X)[:, 1]
    else:
        y_pred_proba = model.predict(X)
    
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    return {
        'AUC': roc_auc_score(y, y_pred_proba),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'F1': f1_score(y, y_pred, zero_division=0),
        'AvgPrec': average_precision_score(y, y_pred_proba)
    }

# XGBoost training
scale_pos_weight_real = (y_train_real == 0).sum() / (y_train_real == 1).sum()
scale_pos_weight_synth = (y_train_synth == 0).sum() / (y_train_synth == 1).sum()

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'tree_method': 'hist'
}

# Train on real data
xgb_real = xgb.XGBClassifier(**{**xgb_params, 'scale_pos_weight': scale_pos_weight_real})
xgb_real.fit(
    X_train_real, y_train_real,
    eval_set=[(X_val_real, y_val_real)],
    callbacks=[EarlyStopping(rounds=50, save_best=True)],
    verbose=False
)

# Train on synthetic data
xgb_synth = xgb.XGBClassifier(**{**xgb_params, 'scale_pos_weight': scale_pos_weight_synth})
xgb_synth.fit(
    X_train_synth, y_train_synth,
    eval_set=[(X_val_real, y_val_real)],
    callbacks=[EarlyStopping(rounds=50, save_best=True)],
    verbose=False
)

# Evaluate XGBoost models
xgb_real_results = evaluate_model(xgb_real, X_test_real, y_test_real)
xgb_synth_results = evaluate_model(xgb_synth, X_test_real, y_test_real)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

lr_real = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
lr_real.fit(X_train_real, y_train_real)

lr_synth = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
lr_synth.fit(X_train_synth, y_train_synth)

lr_real_results = evaluate_model(lr_real, X_test_real, y_test_real)
lr_synth_results = evaluate_model(lr_synth, X_test_real, y_test_real)

# EBM
ebm_real = ExplainableBoostingClassifier(random_state=RANDOM_STATE, max_bins=32)
ebm_real.fit(X_train_real, y_train_real)

ebm_synth = ExplainableBoostingClassifier(random_state=RANDOM_STATE, max_bins=32)
ebm_synth.fit(X_train_synth, y_train_synth)

ebm_real_results = evaluate_model(ebm_real, X_test_real, y_test_real)
ebm_synth_results = evaluate_model(ebm_synth, X_test_real, y_test_real)

# Compile results
results = []

for model_name, real_res, synth_res in [
    ('XGBoost', xgb_real_results, xgb_synth_results),
    ('LogisticRegression', lr_real_results, lr_synth_results),
    ('EBM', ebm_real_results, ebm_synth_results)
]:
    # Real data results
    results.append({
        'Dataset': 'Home Credit',
        'Model': model_name,
        'Data': 'Real',
        **real_res
    })
    
    # Synthetic data results
    results.append({
        'Dataset': 'Home Credit',
        'Model': model_name,
        'Data': 'Synthetic',
        **synth_res
    })

results_df = pd.DataFrame(results)

# Calculate retention rates
retention_results = []

for model_name, real_res, synth_res in [
    ('XGBoost', xgb_real_results, xgb_synth_results),
    ('LogisticRegression', lr_real_results, lr_synth_results),
    ('EBM', ebm_real_results, ebm_synth_results)
]:
    retention = {
        'Dataset': 'Home Credit',
        'Model': model_name,
        'AUC_Retention_%': (synth_res['AUC'] / real_res['AUC']) * 100 if real_res['AUC'] > 0 else 0,
        'Recall_Retention_%': (synth_res['Recall'] / real_res['Recall']) * 100 if real_res['Recall'] > 0 else 0,
        'Precision_Retention_%': (synth_res['Precision'] / real_res['Precision']) * 100 if real_res['Precision'] > 0 else 0,
        'F1_Retention_%': (synth_res['F1'] / real_res['F1']) * 100 if real_res['F1'] > 0 else 0,
        'AvgPrec_Retention_%': (synth_res['AvgPrec'] / real_res['AvgPrec']) * 100 if real_res['AvgPrec'] > 0 else 0
    }
    retention_results.append(retention)

retention_df = pd.DataFrame(retention_results)

# Save results
results_df.to_csv(RESULTS_DIR / 'model_performance.csv', index=False)
retention_df.to_csv(RESULTS_DIR / 'performance_retention.csv', index=False)

# Save models
xgb_real.save_model(str(RESULTS_DIR / 'xgb_real.json'))
xgb_synth.save_model(str(RESULTS_DIR / 'xgb_synth.json'))

import pickle
with open(RESULTS_DIR / 'lr_real.pkl', 'wb') as f:
    pickle.dump(lr_real, f)
with open(RESULTS_DIR / 'lr_synth.pkl', 'wb') as f:
    pickle.dump(lr_synth, f)
with open(RESULTS_DIR / 'ebm_real.pkl', 'wb') as f:
    pickle.dump(ebm_real, f)
with open(RESULTS_DIR / 'ebm_synth.pkl', 'wb') as f:
    pickle.dump(ebm_synth, f)

print("Model training complete")
print(f"\nXGBoost Performance:")
print(f"  Real AUC: {xgb_real_results['AUC']:.4f}")
print(f"  Synth AUC: {xgb_synth_results['AUC']:.4f}")
print(f"  Retention: {retention_results[0]['AUC_Retention_%']:.1f}%")
print(f"\nResults saved to {RESULTS_DIR}")