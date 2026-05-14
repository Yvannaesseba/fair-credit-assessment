"""
Home Credit Cross-Validation with TSTR Framework
Evaluates synthetic data utility through paired CV experiments
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
N_FOLDS = 5
BASE_DIR = Path('/path/to/dissertation')  # UPDATE THIS
FS_DIR = BASE_DIR / 'Feature_Selection_Results'
CTGAN_DIR = BASE_DIR / 'Results' / 'CTGAN'
RESULTS_DIR = BASE_DIR / 'Results' / 'CV_TSTR'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
train_real = pd.read_csv(FS_DIR / 'train_ctgan_ready.csv')
val_real = pd.read_csv(FS_DIR / 'val_ctgan_ready.csv')
test_real = pd.read_csv(FS_DIR / 'test_ctgan_ready.csv')
train_synth = pd.read_csv(CTGAN_DIR / 'synth_train_ctgan.csv')

# Combine train+val for CV
data_real = pd.concat([train_real, val_real], ignore_index=True)

# Preprocessing
categorical_cols = [c for c in data_real.columns if data_real[c].dtype == 'object' and c != 'TARGET']

def preprocess(df, categorical_cols, encoders=None):
    """Encode categoricals"""
    df_proc = df.copy()
    
    if encoders is None:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_proc[col] = le.fit_transform(df_proc[col].astype(str))
            encoders[col] = le
    else:
        for col in categorical_cols:
            le = encoders[col]
            df_proc[col] = df_proc[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
            if 'Unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'Unknown')
            df_proc[col] = le.transform(df_proc[col].astype(str))
    
    X = df_proc.drop('TARGET', axis=1)
    y = df_proc['TARGET']
    return X, y, encoders

# Stratified K-Fold CV
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

cv_results = []

print(f"Running {N_FOLDS}-Fold CV-TSTR...")

for fold, (train_idx, val_idx) in enumerate(skf.split(data_real, data_real['TARGET']), 1):
    print(f"\nFold {fold}/{N_FOLDS}")
    
    # Split real data
    fold_train_real = data_real.iloc[train_idx].copy()
    fold_val_real = data_real.iloc[val_idx].copy()
    
    # Preprocess
    X_train_real, y_train_real, encoders = preprocess(fold_train_real, categorical_cols)
    X_val_real, y_val_real, _ = preprocess(fold_val_real, categorical_cols, encoders)
    X_test_real, y_test_real, _ = preprocess(test_real, categorical_cols, encoders)
    
    # Generate synthetic data for this fold
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(fold_train_real)
    
    for col in categorical_cols + ['TARGET']:
        metadata.update_column(column_name=col, sdtype='categorical')
    
    ctgan_fold = CTGANSynthesizer(
        metadata=metadata,
        epochs=100,
        verbose=False,
        cuda=False
    )
    
    ctgan_fold.fit(fold_train_real)
    fold_train_synth = ctgan_fold.sample(num_rows=len(fold_train_real))
    
    # Preprocess synthetic
    X_train_synth, y_train_synth, _ = preprocess(fold_train_synth, categorical_cols, encoders)
    
    # Train models
    scale_pos_weight_real = (y_train_real == 0).sum() / (y_train_real == 1).sum()
    scale_pos_weight_synth = (y_train_synth == 0).sum() / (y_train_synth == 1).sum()
    
    # Real model
    xgb_real = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        scale_pos_weight=scale_pos_weight_real,
        random_state=RANDOM_STATE
    )
    xgb_real.fit(X_train_real, y_train_real, verbose=False)
    
    # Synthetic model
    xgb_synth = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        scale_pos_weight=scale_pos_weight_synth,
        random_state=RANDOM_STATE
    )
    xgb_synth.fit(X_train_synth, y_train_synth, verbose=False)
    
    # Evaluate on test set
    y_pred_real = xgb_real.predict_proba(X_test_real)[:, 1]
    y_pred_synth = xgb_synth.predict_proba(X_test_real)[:, 1]
    
    auc_real = roc_auc_score(y_test_real, y_pred_real)
    auc_synth = roc_auc_score(y_test_real, y_pred_synth)
    
    cv_results.append({
        'Fold': fold,
        'AUC_Real': auc_real,
        'AUC_Synth': auc_synth,
        'AUC_Diff': auc_synth - auc_real
    })
    
    print(f"  AUC Real: {auc_real:.4f}")
    print(f"  AUC Synth: {auc_synth:.4f}")
    print(f"  Diff: {auc_synth - auc_real:.4f}")

cv_df = pd.DataFrame(cv_results)

# Statistical tests
t_stat, p_value = stats.ttest_rel(cv_df['AUC_Real'], cv_df['AUC_Synth'])

# Effect size (Cohen's d)
diff = cv_df['AUC_Synth'] - cv_df['AUC_Real']
cohens_d = diff.mean() / diff.std()

# Bootstrap confidence intervals
n_bootstrap = 1000
bootstrap_diffs = []

for _ in range(n_bootstrap):
    sample = np.random.choice(diff, size=len(diff), replace=True)
    bootstrap_diffs.append(sample.mean())

ci_lower = np.percentile(bootstrap_diffs, 2.5)
ci_upper = np.percentile(bootstrap_diffs, 97.5)

# Summary
summary = {
    'Dataset': 'Home Credit',
    'N_Folds': N_FOLDS,
    'Mean_AUC_Real': cv_df['AUC_Real'].mean(),
    'Mean_AUC_Synth': cv_df['AUC_Synth'].mean(),
    'Mean_AUC_Diff': cv_df['AUC_Diff'].mean(),
    'SD_AUC_Diff': cv_df['AUC_Diff'].std(),
    'T_Statistic': t_stat,
    'P_Value': p_value,
    'Cohens_d': cohens_d,
    'CI_95_Lower': ci_lower,
    'CI_95_Upper': ci_upper,
    'Significant': p_value < 0.05
}

summary_df = pd.DataFrame([summary])

# Save results
cv_df.to_csv(RESULTS_DIR / 'cv_tstr_folds.csv', index=False)
summary_df.to_csv(RESULTS_DIR / 'cv_tstr_summary.csv', index=False)

print("\n" + "="*60)
print("CV-TSTR Results")
print("="*60)
print(f"Mean AUC (Real):  {summary['Mean_AUC_Real']:.4f}")
print(f"Mean AUC (Synth): {summary['Mean_AUC_Synth']:.4f}")
print(f"Mean Difference:  {summary['Mean_AUC_Diff']:.4f} ± {summary['SD_AUC_Diff']:.4f}")
print(f"95% CI: [{summary['CI_95_Lower']:.4f}, {summary['CI_95_Upper']:.4f}]")
print(f"Cohen's d: {summary['Cohens_d']:.4f}")
print(f"P-value: {summary['P_Value']:.4f}")
print(f"Significant: {'Yes' if summary['Significant'] else 'No'}")
print("="*60)
print(f"Results saved to {RESULTS_DIR}")