import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score,
    f1_score, average_precision_score
)
from sklearn.model_selection import train_test_split
import os

SEED = 42

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


if __name__ == "__main__":

    # load real and synthetic GMSC data
    print("Loading data...")
    gmsc_real = pd.read_csv('data/gmsc_cleaned.csv')
    gmsc_synth = pd.read_csv('data/gmsc_synthetic.csv')

    X_real, y_real = preprocess_data(gmsc_real, is_real=True)
    X_synth, y_synth = preprocess_data(gmsc_synth, is_real=False)

    # keep only columns that exist in both datasets
    common_cols = list(set(X_real.columns) & set(X_synth.columns))
    X_real = X_real[common_cols]
    X_synth = X_synth[common_cols]

    # split real data into train and test
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.2, stratify=y_real, random_state=SEED
    )

    # split synthetic data into train and test
    X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(
        X_synth, y_synth, test_size=0.2, stratify=y_synth, random_state=SEED
    )

    print(f"Real train: {X_real_train.shape}, test: {X_real_test.shape}")
    print(f"Synth train: {X_synth_train.shape}, test: {X_synth_test.shape}")

    # class imbalance ratio used to weight XGBoost
    imbalance_ratio = (y_real_train == 0).sum() / (y_real_train == 1).sum()
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}")

    # results will collect all model metrics
    all_results = []

    # store probabilities for the validation script
    probabilities = {}

    # XGBoost
    print("\nTraining XGBoost...")
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

    model_xgb_real = XGBClassifier(**xgb_params)
    model_xgb_real.fit(X_real_train, y_real_train)

    model_xgb_synth = XGBClassifier(**xgb_params)
    model_xgb_synth.fit(X_synth_train, y_synth_train)

    y_proba_xgb_real = model_xgb_real.predict_proba(X_real_test)[:, 1]
    y_proba_xgb_synth = model_xgb_synth.predict_proba(X_real_test)[:, 1]
    y_pred_xgb_real = (y_proba_xgb_real >= 0.5).astype(int)
    y_pred_xgb_synth = (y_proba_xgb_synth >= 0.5).astype(int)

    probabilities['xgb_real'] = y_proba_xgb_real
    probabilities['xgb_synth'] = y_proba_xgb_synth

    all_results.append({'Model': 'XGBoost', 'Data': 'Real', 'AUC': roc_auc_score(y_real_test, y_proba_xgb_real), 'Recall': recall_score(y_real_test, y_pred_xgb_real), 'Precision': precision_score(y_real_test, y_pred_xgb_real), 'F1': f1_score(y_real_test, y_pred_xgb_real), 'AvgPrec': average_precision_score(y_real_test, y_proba_xgb_real)})
    all_results.append({'Model': 'XGBoost', 'Data': 'Synthetic', 'AUC': roc_auc_score(y_real_test, y_proba_xgb_synth), 'Recall': recall_score(y_real_test, y_pred_xgb_synth), 'Precision': precision_score(y_real_test, y_pred_xgb_synth), 'F1': f1_score(y_real_test, y_pred_xgb_synth), 'AvgPrec': average_precision_score(y_real_test, y_proba_xgb_synth)})

    # Logistic Regression
    print("Training Logistic Regression...")
    lr_real = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED, n_jobs=-1))])
    lr_synth = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED, n_jobs=-1))])

    lr_real.fit(X_real_train, y_real_train)
    lr_synth.fit(X_synth_train, y_synth_train)

    y_proba_lr_real = lr_real.predict_proba(X_real_test)[:, 1]
    y_proba_lr_synth = lr_synth.predict_proba(X_real_test)[:, 1]
    y_pred_lr_real = (y_proba_lr_real >= 0.5).astype(int)
    y_pred_lr_synth = (y_proba_lr_synth >= 0.5).astype(int)

    probabilities['lr_real'] = y_proba_lr_real
    probabilities['lr_synth'] = y_proba_lr_synth

    all_results.append({'Model': 'LogisticRegression', 'Data': 'Real', 'AUC': roc_auc_score(y_real_test, y_proba_lr_real), 'Recall': recall_score(y_real_test, y_pred_lr_real), 'Precision': precision_score(y_real_test, y_pred_lr_real), 'F1': f1_score(y_real_test, y_pred_lr_real), 'AvgPrec': average_precision_score(y_real_test, y_proba_lr_real)})
    all_results.append({'Model': 'LogisticRegression', 'Data': 'Synthetic', 'AUC': roc_auc_score(y_real_test, y_proba_lr_synth), 'Recall': recall_score(y_real_test, y_pred_lr_synth), 'Precision': precision_score(y_real_test, y_pred_lr_synth), 'F1': f1_score(y_real_test, y_pred_lr_synth), 'AvgPrec': average_precision_score(y_real_test, y_proba_lr_synth)})

    # EBM
    print("Training EBM...")
    ebm_real = ExplainableBoostingClassifier(random_state=SEED, n_jobs=-1)
    ebm_real.fit(X_real_train, y_real_train)

    ebm_synth = ExplainableBoostingClassifier(random_state=SEED, n_jobs=-1)
    ebm_synth.fit(X_synth_train, y_synth_train)

    y_proba_ebm_real = ebm_real.predict_proba(X_real_test)[:, 1]
    y_proba_ebm_synth = ebm_synth.predict_proba(X_real_test)[:, 1]
    y_pred_ebm_real = (y_proba_ebm_real >= 0.5).astype(int)
    y_pred_ebm_synth = (y_proba_ebm_synth >= 0.5).astype(int)

    probabilities['ebm_real'] = y_proba_ebm_real
    probabilities['ebm_synth'] = y_proba_ebm_synth

    all_results.append({'Model': 'EBM', 'Data': 'Real', 'AUC': roc_auc_score(y_real_test, y_proba_ebm_real), 'Recall': recall_score(y_real_test, y_pred_ebm_real), 'Precision': precision_score(y_real_test, y_pred_ebm_real), 'F1': f1_score(y_real_test, y_pred_ebm_real), 'AvgPrec': average_precision_score(y_real_test, y_proba_ebm_real)})
    all_results.append({'Model': 'EBM', 'Data': 'Synthetic', 'AUC': roc_auc_score(y_real_test, y_proba_ebm_synth), 'Recall': recall_score(y_real_test, y_pred_ebm_synth), 'Precision': precision_score(y_real_test, y_pred_ebm_synth), 'F1': f1_score(y_real_test, y_pred_ebm_synth), 'AvgPrec': average_precision_score(y_real_test, y_proba_ebm_synth)})

    # save results
    os.makedirs('outputs/metrics', exist_ok=True)
    os.makedirs('outputs/roc_curves', exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv('outputs/metrics/gmsc_model_performance.csv', index=False)
    print("\nModel performance:")
    print(results_df.to_string(index=False))

    # save probabilities for the validation and fairness scripts
    pd.DataFrame({'probability': probabilities['xgb_real']}).to_csv('outputs/metrics/gmsc_proba_xgb_real.csv', index=False)
    pd.DataFrame({'probability': probabilities['xgb_synth']}).to_csv('outputs/metrics/gmsc_proba_xgb_synth.csv', index=False)
    pd.DataFrame({'probability': probabilities['lr_real']}).to_csv('outputs/metrics/gmsc_proba_lr_real.csv', index=False)
    pd.DataFrame({'probability': probabilities['lr_synth']}).to_csv('outputs/metrics/gmsc_proba_lr_synth.csv', index=False)
    pd.DataFrame({'probability': probabilities['ebm_real']}).to_csv('outputs/metrics/gmsc_proba_ebm_real.csv', index=False)
    pd.DataFrame({'probability': probabilities['ebm_synth']}).to_csv('outputs/metrics/gmsc_proba_ebm_synth.csv', index=False)

    # save ROC curve data for the portfolio page
    from sklearn.metrics import roc_curve
    for model_name, proba_real, proba_synth in [
        ('xgboost', y_proba_xgb_real, y_proba_xgb_synth),
        ('lr', y_proba_lr_real, y_proba_lr_synth),
        ('ebm', y_proba_ebm_real, y_proba_ebm_synth)
    ]:
        for data_type, proba in [('real', proba_real), ('synth', proba_synth)]:
            fpr, tpr, _ = roc_curve(y_real_test, proba)
            auc = roc_auc_score(y_real_test, proba)
            roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
            roc_df['auc'] = auc
            roc_df.to_csv(f'outputs/roc_curves/gmsc_{model_name}_{data_type}.csv', index=False)

    print("\nSaved results to outputs/metrics/ and outputs/roc_curves/")
    print("Done.")