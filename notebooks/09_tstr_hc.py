import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 5

def safe_fit_ctgan(fold_train, categorical_cols, max_retries=3):
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata

    # cap categorical columns to top 20 categories to avoid NaN issues
    fold_clean = fold_train.copy()
    for col in categorical_cols:
        top_cats = fold_clean[col].value_counts().head(20).index
        fold_clean[col] = fold_clean[col].apply(lambda x: x if x in top_cats else 'Other')

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(fold_clean)
    for col in categorical_cols + ['TARGET']:
        if col in fold_clean.columns:
            metadata.update_column(column_name=col, sdtype='categorical')

    for attempt in range(max_retries):
        try:
            ctgan_fold = CTGANSynthesizer(
                metadata=metadata,
                epochs=100,
                verbose=False,
                batch_size=500,
                pac=10
            )
            ctgan_fold.fit(fold_clean)
            synth = ctgan_fold.sample(num_rows=len(fold_clean))
            return synth, fold_clean
        except ValueError as e:
            print(f"  CTGAN attempt {attempt+1} failed: {e}, retrying...")

    raise RuntimeError("CTGAN failed after max retries")


if __name__ == "__main__":

    print(f"Running {N_FOLDS}-fold CV-TSTR on Home Credit...")
    train_real = pd.read_csv('data/hc_train_ctgan_ready.csv')
    val_real = pd.read_csv('data/hc_val_ctgan_ready.csv')
    test_real = pd.read_csv('data/hc_test_ctgan_ready.csv')

    # combine train and val for cross validation
    data_real = pd.concat([train_real, val_real], ignore_index=True)

    categorical_cols = [c for c in data_real.columns if data_real[c].dtype == 'object' and c != 'TARGET']

    def preprocess(df, categorical_cols, encoders=None):
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

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(data_real, data_real['TARGET']), 1):
        print(f"\nFold {fold}/{N_FOLDS}")

        fold_train = data_real.iloc[train_idx].copy()
        fold_train['TARGET'] = fold_train['TARGET'].astype(str)

        X_train_real, y_train_real, encoders = preprocess(fold_train, categorical_cols)
        X_test_real, y_test_real, _ = preprocess(test_real, categorical_cols, encoders)

        # generate synthetic data for this fold
        fold_synth, fold_clean = safe_fit_ctgan(fold_train, categorical_cols)

        # make sure TARGET is binary
        fold_synth['TARGET'] = fold_synth['TARGET'].apply(
            lambda x: 1 if str(x) == '1' else 0
        ).astype(int)

        X_train_synth, y_train_synth, _ = preprocess(fold_synth, categorical_cols, encoders)

        # align columns between real and synthetic
        common_cols = [c for c in X_train_real.columns if c in X_train_synth.columns]
        X_train_real = X_train_real[common_cols]
        X_train_synth = X_train_synth[common_cols]
        X_test_real = X_test_real[common_cols]

        scale_pos_weight_real = (y_train_real == 0).sum() / (y_train_real == 1).sum()
        scale_pos_weight_synth = (y_train_synth == 0).sum() / (y_train_synth == 1).sum()

        xgb_real = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.05, n_estimators=300,
            scale_pos_weight=scale_pos_weight_real,
            random_state=RANDOM_STATE, tree_method='hist'
        )
        xgb_real.fit(X_train_real, y_train_real, verbose=False)

        xgb_synth = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.05, n_estimators=300,
            scale_pos_weight=scale_pos_weight_synth,
            random_state=RANDOM_STATE, tree_method='hist'
        )
        xgb_synth.fit(X_train_synth, y_train_synth, verbose=False)

        auc_real = roc_auc_score(y_test_real, xgb_real.predict_proba(X_test_real)[:, 1])
        auc_synth = roc_auc_score(y_test_real, xgb_synth.predict_proba(X_test_real)[:, 1])

        cv_results.append({
            'Fold': fold,
            'AUC_Real': auc_real,
            'AUC_Synth': auc_synth,
            'AUC_Diff': auc_synth - auc_real
        })

        print(f"  AUC Real: {auc_real:.4f}, AUC Synth: {auc_synth:.4f}, Diff: {auc_synth - auc_real:.4f}")

    cv_df = pd.DataFrame(cv_results)

    t_stat, p_value = stats.ttest_rel(cv_df['AUC_Real'], cv_df['AUC_Synth'])

    diff = cv_df['AUC_Synth'] - cv_df['AUC_Real']
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

    bootstrap_diffs = [np.random.choice(diff, size=len(diff), replace=True).mean() for _ in range(1000)]
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    summary = {
        'Dataset': 'HomeCredit',
        'N_Folds': N_FOLDS,
        'Mean_AUC_Real': cv_df['AUC_Real'].mean(),
        'Mean_AUC_Synth': cv_df['AUC_Synth'].mean(),
        'Mean_AUC_Diff': diff.mean(),
        'SD_AUC_Diff': diff.std(),
        'T_Statistic': t_stat,
        'P_Value': p_value,
        'Cohens_d': cohens_d,
        'CI_95_Lower': ci_lower,
        'CI_95_Upper': ci_upper,
        'Significant': p_value < 0.05
    }

    print("\n" + "="*60)
    print("CV-TSTR Summary")
    print("="*60)
    print(f"Mean AUC Real:  {summary['Mean_AUC_Real']:.4f}")
    print(f"Mean AUC Synth: {summary['Mean_AUC_Synth']:.4f}")
    print(f"Mean Diff:      {summary['Mean_AUC_Diff']:.4f} +/- {summary['SD_AUC_Diff']:.4f}")
    print(f"95% CI:         [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Cohen's d:      {cohens_d:.4f}")
    print(f"P-value:        {p_value:.4f}")
    print(f"Significant:    {'Yes' if summary['Significant'] else 'No'}")

    os.makedirs('outputs/metrics', exist_ok=True)
    cv_df.to_csv('outputs/metrics/hc_cv_tstr_folds.csv', index=False)
    pd.DataFrame([summary]).to_csv('outputs/metrics/hc_cv_tstr_summary.csv', index=False)

    print("\nSaved to outputs/metrics/")
    print("Done.")