import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import os

if __name__ == "__main__":

    # load the test labels and model probabilities saved by 03_models_gmsc.py
    print("Loading results...")
    y_test = pd.read_csv('data/gmsc_test.csv')['SeriousDlqin2yrs']

    models = ['xgb', 'lr', 'ebm']
    results = []

    os.makedirs('outputs/metrics', exist_ok=True)

    for model in models:
        print(f"\nRunning bootstrap validation for {model.upper()}...")

        y_proba_real = pd.read_csv(f'outputs/metrics/gmsc_proba_{model}_real.csv')['probability']
        y_proba_synth = pd.read_csv(f'outputs/metrics/gmsc_proba_{model}_synth.csv')['probability']

        # bootstrap 1000 times to get confidence intervals on AUC
        auc_real_scores = []
        auc_synth_scores = []
        auc_diffs = []
        n_samples = len(y_test)

        for i in range(1000):
            indices = resample(range(n_samples), n_samples=n_samples, random_state=i)
            y_boot = y_test.iloc[indices]
            p_real_boot = y_proba_real.iloc[indices]
            p_synth_boot = y_proba_synth.iloc[indices]

            auc_r = roc_auc_score(y_boot, p_real_boot)
            auc_s = roc_auc_score(y_boot, p_synth_boot)

            auc_real_scores.append(auc_r)
            auc_synth_scores.append(auc_s)
            auc_diffs.append(auc_s - auc_r)

        auc_real_scores = np.array(auc_real_scores)
        auc_synth_scores = np.array(auc_synth_scores)
        auc_diffs = np.array(auc_diffs)

        # paired t-test on the bootstrap differences
        t_stat, p_value = stats.ttest_1samp(auc_diffs, 0)

        # cohen's d effect size
        pooled_std = np.sqrt((auc_real_scores.std()**2 + auc_synth_scores.std()**2) / 2)
        cohens_d = auc_diffs.mean() / pooled_std if pooled_std > 0 else 0

        if abs(cohens_d) < 0.2:
            effect = "Negligible"
        elif abs(cohens_d) < 0.5:
            effect = "Small"
        elif abs(cohens_d) < 0.8:
            effect = "Medium"
        else:
            effect = "Large"

        results.append({
            'Model': model.upper(),
            'AUC_Real_Mean': auc_real_scores.mean(),
            'AUC_Real_CI_Lower': np.percentile(auc_real_scores, 2.5),
            'AUC_Real_CI_Upper': np.percentile(auc_real_scores, 97.5),
            'AUC_Synth_Mean': auc_synth_scores.mean(),
            'AUC_Synth_CI_Lower': np.percentile(auc_synth_scores, 2.5),
            'AUC_Synth_CI_Upper': np.percentile(auc_synth_scores, 97.5),
            'AUC_Diff_Mean': auc_diffs.mean(),
            'CI_Lower': np.percentile(auc_diffs, 2.5),
            'CI_Upper': np.percentile(auc_diffs, 97.5),
            'T_Statistic': t_stat,
            'P_Value': p_value,
            'Cohens_d': cohens_d,
            'Effect_Size': effect,
            'Significant': p_value < 0.05
        })

        print(f"  AUC Real:  {auc_real_scores.mean():.4f} [{np.percentile(auc_real_scores, 2.5):.4f}, {np.percentile(auc_real_scores, 97.5):.4f}]")
        print(f"  AUC Synth: {auc_synth_scores.mean():.4f} [{np.percentile(auc_synth_scores, 2.5):.4f}, {np.percentile(auc_synth_scores, 97.5):.4f}]")
        print(f"  Diff: {auc_diffs.mean():.4f}, p={p_value:.4f}, Cohen's d={cohens_d:.4f} ({effect})")

    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/metrics/gmsc_validation.csv', index=False)
    print("\nSaved to outputs/metrics/gmsc_validation.csv")