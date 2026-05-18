import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import warnings
import os

warnings.filterwarnings("ignore")

THRESHOLD = 0.5

def make_age_groups(age_years):
    bins = [-np.inf, 35, 55, np.inf]
    labels = ["Young (<=35)", "Middle (36-55)", "Senior (56+)"]
    return pd.cut(age_years, bins=bins, labels=labels)

def compute_fairness(y_true, y_proba, age_groups, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    results = []

    for group in age_groups.unique():
        mask = age_groups == group
        if mask.sum() == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        yprob = y_proba[mask]

        # selection rate is the proportion predicted as default
        selection_rate = yp.mean()

        # true positive rate (equal opportunity)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        results.append({
            'Group': str(group),
            'Count': int(mask.sum()),
            'SelectionRate': selection_rate,
            'TPR': tpr,
            'FPR': fpr,
            'Precision': precision_score(yt, yp, zero_division=0),
            'Recall': recall_score(yt, yp, zero_division=0),
            'F1': f1_score(yt, yp, zero_division=0),
            'Accuracy': accuracy_score(yt, yp)
        })

    by_group = pd.DataFrame(results)

    # demographic parity difference — spread in selection rates across groups
    dp_diff = by_group['SelectionRate'].max() - by_group['SelectionRate'].min()

    # equal opportunity difference — spread in TPR across groups
    eo_diff = by_group['TPR'].max() - by_group['TPR'].min()

    summary = {
        'DP_Difference': float(dp_diff),
        'EO_Difference': float(eo_diff)
    }

    return by_group, summary


if __name__ == "__main__":

    os.makedirs('outputs/fairness', exist_ok=True)

    # run fairness analysis for both datasets
    datasets = [
        {
            'name': 'GMSC',
            'test_path': 'data/gmsc_test.csv',
            'age_col': 'age',
            'models': {
                'XGBoost': ('outputs/metrics/gmsc_proba_xgb_real.csv', 'outputs/metrics/gmsc_proba_xgb_synth.csv'),
                'LogisticRegression': ('outputs/metrics/gmsc_proba_lr_real.csv', 'outputs/metrics/gmsc_proba_lr_synth.csv'),
                'EBM': ('outputs/metrics/gmsc_proba_ebm_real.csv', 'outputs/metrics/gmsc_proba_ebm_synth.csv')
            }
        },
        {
            'name': 'HomeCredit',
            'test_path': 'data/hc_test_ctgan_ready.csv',
            'age_col': 'DAYS_BIRTH',
            'models': {
                'XGBoost': ('outputs/metrics/hc_proba_xgb_real.csv', 'outputs/metrics/hc_proba_xgb_synth.csv'),
                'LogisticRegression': ('outputs/metrics/hc_proba_lr_real.csv', 'outputs/metrics/hc_proba_lr_synth.csv'),
                'EBM': ('outputs/metrics/hc_proba_ebm_real.csv', 'outputs/metrics/hc_proba_ebm_synth.csv')
            }
        }
    ]

    all_summaries = []

    for ds in datasets:
        print(f"\nRunning fairness analysis for {ds['name']}...")
        print("="*60)

        test_df = pd.read_csv(ds['test_path'])

        # derive age in years
        if ds['age_col'] == 'DAYS_BIRTH':
            # DAYS_BIRTH is negative in Home Credit
            age_years = (-test_df['DAYS_BIRTH'] / 365.0)
        else:
            age_years = test_df[ds['age_col']]

        age_groups = make_age_groups(age_years).astype(str)

        # get true labels
        target_col = 'TARGET' if 'TARGET' in test_df.columns else 'SeriousDlqin2yrs'
        y_true = test_df[target_col].values

        for model_name, (real_path, synth_path) in ds['models'].items():
            print(f"\n  {model_name}")

            y_proba_real = pd.read_csv(real_path)['probability'].values
            y_proba_synth = pd.read_csv(synth_path)['probability'].values

            by_group_real, summary_real = compute_fairness(y_true, y_proba_real, age_groups)
            by_group_synth, summary_synth = compute_fairness(y_true, y_proba_synth, age_groups)

            by_group_real['TrainingData'] = 'Real'
            by_group_synth['TrainingData'] = 'Synthetic'
            by_group_real['Model'] = model_name
            by_group_synth['Model'] = model_name
            by_group_real['Dataset'] = ds['name']
            by_group_synth['Dataset'] = ds['name']

            combined = pd.concat([by_group_real, by_group_synth], ignore_index=True)
            combined.to_csv(f"outputs/fairness/{ds['name'].lower()}_{model_name.lower()}_fairness.csv", index=False)

            all_summaries.append({'Dataset': ds['name'], 'Model': model_name, 'TrainingData': 'Real', **summary_real})
            all_summaries.append({'Dataset': ds['name'], 'Model': model_name, 'TrainingData': 'Synthetic', **summary_synth})

            print(f"    Real      — DP diff: {summary_real['DP_Difference']:.4f}, EO diff: {summary_real['EO_Difference']:.4f}")
            print(f"    Synthetic — DP diff: {summary_synth['DP_Difference']:.4f}, EO diff: {summary_synth['EO_Difference']:.4f}")

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv('outputs/fairness/fairness_summary.csv', index=False)

    print("\nSaved all fairness results to outputs/fairness/")
    print("Done.")