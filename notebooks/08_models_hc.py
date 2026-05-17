import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingClassifier
import warnings
import os

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

def preprocess(train_df, val_df, test_df):
    categorical_cols = [c for c in train_df.columns if train_df[c].dtype == 'object' and c != 'TARGET']

    train_proc = train_df.copy()
    val_proc = val_df.copy()
    test_proc = test_df.copy()

    # encode categorical columns using train set labels
    for col in categorical_cols:
        le = LabelEncoder()
        train_proc[col] = le.fit_transform(train_proc[col].astype(str))
        val_proc[col] = val_proc[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
        test_proc[col] = test_proc[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
        if 'Unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'Unknown')
        val_proc[col] = le.transform(val_proc[col].astype(str))
        test_proc[col] = le.transform(test_proc[col].astype(str))

    X_train = train_proc.drop('TARGET', axis=1)
    y_train = train_proc['TARGET']
    X_val = val_proc.drop('TARGET', axis=1)
    y_val = val_proc['TARGET']
    X_test = test_proc.drop('TARGET', axis=1)
    y_test = test_proc['TARGET']

    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        'AUC': roc_auc_score(y, y_proba),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'F1': f1_score(y, y_pred, zero_division=0),
        'AvgPrec': average_precision_score(y, y_proba)
    }, y_proba


if __name__ == "__main__":

    # load real and synthetic training data
    print("Loading Home Credit data...")
    train_real = pd.read_csv('data/hc_train_ctgan_ready.csv')
    val_real = pd.read_csv('data/hc_val_ctgan_ready.csv')
    test_real = pd.read_csv('data/hc_test_ctgan_ready.csv')
    train_synth = pd.read_csv('data/hc_synthetic.csv')

    # preprocess real data
    X_train_real, y_train_real, X_val_real, y_val_real, X_test_real, y_test_real = \
        preprocess(train_real, val_real, test_real)

    # preprocess synthetic data using same val and test splits
    X_train_synth, y_train_synth, _, _, _, _ = \
        preprocess(train_synth, val_real, test_real)

    print(f"Real train: {X_train_real.shape}, Test: {X_test_real.shape}")
    print(f"Synth train: {X_train_synth.shape}")

    all_results = []
    all_probas = {}

    # XGBoost
    print("\nTraining XGBoost...")
    scale_pos_weight_real = (y_train_real == 0).sum() / (y_train_real == 1).sum()
    scale_pos_weight_synth = (y_train_synth == 0).sum() / (y_train_synth == 1).sum()

    xgb_params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'tree_method': 'hist'
    }

    xgb_real = xgb.XGBClassifier(**{**xgb_params, 'scale_pos_weight': scale_pos_weight_real})
    xgb_real.fit(X_train_real, y_train_real, verbose=False)

    xgb_synth = xgb.XGBClassifier(**{**xgb_params, 'scale_pos_weight': scale_pos_weight_synth})
    xgb_synth.fit(X_train_synth, y_train_synth, verbose=False)

    xgb_real_res, xgb_real_proba = evaluate(xgb_real, X_test_real, y_test_real)
    xgb_synth_res, xgb_synth_proba = evaluate(xgb_synth, X_test_real, y_test_real)

    all_probas['xgb_real'] = xgb_real_proba
    all_probas['xgb_synth'] = xgb_synth_proba

    all_results.append({'Model': 'XGBoost', 'Data': 'Real', 'Dataset': 'HomeCredit', **xgb_real_res})
    all_results.append({'Model': 'XGBoost', 'Data': 'Synthetic', 'Dataset': 'HomeCredit', **xgb_synth_res})

    # Logistic Regression
    print("Training Logistic Regression...")
    lr_real = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE))])
    lr_synth = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE))])

    lr_real.fit(X_train_real, y_train_real)
    lr_synth.fit(X_train_synth, y_train_synth)

    lr_real_res, lr_real_proba = evaluate(lr_real, X_test_real, y_test_real)
    lr_synth_res, lr_synth_proba = evaluate(lr_synth, X_test_real, y_test_real)

    all_probas['lr_real'] = lr_real_proba
    all_probas['lr_synth'] = lr_synth_proba

    all_results.append({'Model': 'LogisticRegression', 'Data': 'Real', 'Dataset': 'HomeCredit', **lr_real_res})
    all_results.append({'Model': 'LogisticRegression', 'Data': 'Synthetic', 'Dataset': 'HomeCredit', **lr_synth_res})

    # EBM
    print("Training EBM...")
    ebm_real = ExplainableBoostingClassifier(random_state=RANDOM_STATE, max_bins=32)
    ebm_real.fit(X_train_real, y_train_real)

    ebm_synth = ExplainableBoostingClassifier(random_state=RANDOM_STATE, max_bins=32)
    ebm_synth.fit(X_train_synth, y_train_synth)

    ebm_real_res, ebm_real_proba = evaluate(ebm_real, X_test_real, y_test_real)
    ebm_synth_res, ebm_synth_proba = evaluate(ebm_synth, X_test_real, y_test_real)

    all_probas['ebm_real'] = ebm_real_proba
    all_probas['ebm_synth'] = ebm_synth_proba

    all_results.append({'Model': 'EBM', 'Data': 'Real', 'Dataset': 'HomeCredit', **ebm_real_res})
    all_results.append({'Model': 'EBM', 'Data': 'Synthetic', 'Dataset': 'HomeCredit', **ebm_synth_res})

    # save results
    os.makedirs('outputs/metrics', exist_ok=True)
    os.makedirs('outputs/roc_curves', exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv('outputs/metrics/hc_model_performance.csv', index=False)

    print("\nModel performance:")
    print(results_df[['Model', 'Data', 'AUC', 'Recall', 'Precision', 'F1']].to_string(index=False))

    # save probabilities for fairness and validation scripts
    for key, proba in all_probas.items():
        pd.DataFrame({'probability': proba}).to_csv(f'outputs/metrics/hc_proba_{key}.csv', index=False)

    # save test labels
    pd.DataFrame({'TARGET': y_test_real.values}).to_csv('outputs/metrics/hc_test_labels.csv', index=False)

    # save ROC curve data for the portfolio page
    model_map = [
        ('xgboost', all_probas['xgb_real'], all_probas['xgb_synth']),
        ('lr', all_probas['lr_real'], all_probas['lr_synth']),
        ('ebm', all_probas['ebm_real'], all_probas['ebm_synth'])
    ]

    for model_name, proba_real, proba_synth in model_map:
        for data_type, proba in [('real', proba_real), ('synth', proba_synth)]:
            fpr, tpr, _ = roc_curve(y_test_real, proba)
            auc = roc_auc_score(y_test_real, proba)
            roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
            roc_df['auc'] = auc
            roc_df.to_csv(f'outputs/roc_curves/hc_{model_name}_{data_type}.csv', index=False)

    print("\nSaved results to outputs/metrics/ and outputs/roc_curves/")
    print("Done.")