import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
import shap
import json
import os
import warnings

warnings.filterwarnings("ignore")

SEED = 42

def preprocess_gmsc(df):
    df = df.copy()
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    y = df['SeriousDlqin2yrs']
    X = df.drop(columns=['SeriousDlqin2yrs'])
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

def preprocess_hc(train_df, test_df):
    categorical_cols = [c for c in train_df.columns if train_df[c].dtype == 'object' and c != 'TARGET']
    train_proc = train_df.copy()
    test_proc = test_df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        train_proc[col] = le.fit_transform(train_proc[col].astype(str))
        test_proc[col] = test_proc[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
        if 'Unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'Unknown')
        test_proc[col] = le.transform(test_proc[col].astype(str))
    X_train = train_proc.drop('TARGET', axis=1)
    y_train = train_proc['TARGET']
    X_test = test_proc.drop('TARGET', axis=1)
    y_test = test_proc['TARGET']
    return X_train, y_train, X_test, y_test

def get_shap_importance(model, X, top_n=10):
    # sample 500 rows to keep SHAP fast
    sample = X.sample(n=min(500, len(X)), random_state=SEED)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    importance = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False).head(top_n)
    return importance.to_dict(orient='records')

def get_ebm_importance(model, top_n=10):
    # use term_names_ which matches the length of term_importances()
    imp_df = pd.DataFrame({
        'feature': model.term_names_,
        'mean_abs_shap': model.term_importances()
    }).sort_values('mean_abs_shap', ascending=False).head(top_n)
    return imp_df.to_dict(orient='records')

def save_shap(data, filename):
    os.makedirs('outputs/shap', exist_ok=True)
    with open(f'outputs/shap/{filename}', 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":

    print("Running SHAP explainability analysis...")

    # GMSC
    print("\nGMSC dataset...")
    gmsc_real = pd.read_csv('data/gmsc_cleaned.csv')
    gmsc_synth = pd.read_csv('data/gmsc_synthetic.csv')

    X_real, y_real = preprocess_gmsc(gmsc_real)
    X_synth, y_synth = preprocess_gmsc(gmsc_synth)

    common_cols = list(set(X_real.columns) & set(X_synth.columns))
    X_real = X_real[common_cols]
    X_synth = X_synth[common_cols]

    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(X_real, y_real, test_size=0.2, stratify=y_real, random_state=SEED)
    X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(X_synth, y_synth, test_size=0.2, stratify=y_synth, random_state=SEED)

    imbalance = (y_real_train == 0).sum() / (y_real_train == 1).sum()

    xgb_params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': imbalance, 'tree_method': 'hist', 'random_state': SEED}

    print("  Training XGBoost (real)...")
    xgb_real = XGBClassifier(**xgb_params)
    xgb_real.fit(X_real_train, y_real_train)

    print("  Training XGBoost (synthetic)...")
    xgb_synth = XGBClassifier(**xgb_params)
    xgb_synth.fit(X_synth_train, y_synth_train)

    print("  Computing SHAP values for XGBoost...")
    save_shap(get_shap_importance(xgb_real, X_real_test), 'gmsc_xgboost_real.json')
    save_shap(get_shap_importance(xgb_synth, X_real_test), 'gmsc_xgboost_synth.json')

    print("  Training EBM (real)...")
    ebm_real = ExplainableBoostingClassifier(random_state=SEED)
    ebm_real.fit(X_real_train, y_real_train)

    print("  Training EBM (synthetic)...")
    ebm_synth = ExplainableBoostingClassifier(random_state=SEED)
    ebm_synth.fit(X_synth_train, y_synth_train)

    print("  Computing EBM importances...")
    save_shap(get_ebm_importance(ebm_real), 'gmsc_ebm_real.json')
    save_shap(get_ebm_importance(ebm_synth), 'gmsc_ebm_synth.json')

    print("  GMSC done.")

    # Home Credit
    print("\nHome Credit dataset...")
    train_real = pd.read_csv('data/hc_train_ctgan_ready.csv')
    test_real = pd.read_csv('data/hc_test_ctgan_ready.csv')
    train_synth = pd.read_csv('data/hc_synthetic.csv')

    X_hc_train_real, y_hc_train_real, X_hc_test, y_hc_test = preprocess_hc(train_real, test_real)
    X_hc_train_synth, y_hc_train_synth, _, _ = preprocess_hc(train_synth, test_real)

    common_hc = [c for c in X_hc_train_real.columns if c in X_hc_train_synth.columns]
    X_hc_train_real = X_hc_train_real[common_hc]
    X_hc_train_synth = X_hc_train_synth[common_hc]
    X_hc_test = X_hc_test[common_hc]

    imbalance_hc = (y_hc_train_real == 0).sum() / (y_hc_train_real == 1).sum()

    print("  Training XGBoost (real)...")
    xgb_hc_real = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=imbalance_hc, tree_method='hist', random_state=SEED)
    xgb_hc_real.fit(X_hc_train_real, y_hc_train_real, verbose=False)

    print("  Training XGBoost (synthetic)...")
    xgb_hc_synth = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=imbalance_hc, tree_method='hist', random_state=SEED)
    xgb_hc_synth.fit(X_hc_train_synth, y_hc_train_synth, verbose=False)

    print("  Computing SHAP values for XGBoost...")
    save_shap(get_shap_importance(xgb_hc_real, X_hc_test), 'hc_xgboost_real.json')
    save_shap(get_shap_importance(xgb_hc_synth, X_hc_test), 'hc_xgboost_synth.json')

    print("  Training EBM (real)...")
    ebm_hc_real = ExplainableBoostingClassifier(random_state=SEED, max_bins=32)
    ebm_hc_real.fit(X_hc_train_real, y_hc_train_real)

    print("  Training EBM (synthetic)...")
    ebm_hc_synth = ExplainableBoostingClassifier(random_state=SEED, max_bins=32)
    ebm_hc_synth.fit(X_hc_train_synth, y_hc_train_synth)

    print("  Computing EBM importances...")
    save_shap(get_ebm_importance(ebm_hc_real), 'hc_ebm_real.json')
    save_shap(get_ebm_importance(ebm_hc_synth), 'hc_ebm_synth.json')

    print("  Home Credit done.")

    print("\nAll SHAP outputs saved to outputs/shap/")
    print("Done.")