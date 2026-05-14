import pandas as pd
import numpy as np
from scipy import stats
from sklearn.utils import resample

print("Loading performance results...")

results = pd.read_csv('Results/gmsc_model_comparison.csv')

print("\n" + "="*80)
print("BOOTSTRAP CONFIDENCE INTERVALS")
print("="*80)

y_proba_real_xgb = pd.read_csv('Results/probabilities_real_xgb.csv')['probability']
y_proba_synth_xgb = pd.read_csv('Results/probabilities_synth_xgb.csv')['probability']
y_test = pd.read_csv('Data/Processed/gmsc_test.csv')['SeriousDlqin2yrs']

def bootstrap_auc(y_true, y_proba, n_iterations=1000, confidence_level=0.95):
    from sklearn.metrics import roc_auc_score
    
    auc_scores = []
    n_samples = len(y_true)
    
    for i in range(n_iterations):
        indices = resample(range(n_samples), n_samples=n_samples, random_state=i)
        y_true_boot = y_true.iloc[indices]
        y_proba_boot = y_proba.iloc[indices]
        
        auc = roc_auc_score(y_true_boot, y_proba_boot)
        auc_scores.append(auc)
    
    auc_scores = np.array(auc_scores)
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    
    alpha = 1 - confidence_level
    ci_lower = np.percentile(auc_scores, alpha/2 * 100)
    ci_upper = np.percentile(auc_scores, (1 - alpha/2) * 100)
    
    return mean_auc, std_auc, ci_lower, ci_upper

print("\nBootstrapping AUC (1000 iterations)...")

mean_real, std_real, ci_lower_real, ci_upper_real = bootstrap_auc(y_test, y_proba_real_xgb)
mean_synth, std_synth, ci_lower_synth, ci_upper_synth = bootstrap_auc(y_test, y_proba_synth_xgb)

bootstrap_results = pd.DataFrame({
    'Model': ['Real', 'Synthetic'],
    'Mean_AUC': [mean_real, mean_synth],
    'Std_AUC': [std_real, std_synth],
    'CI_Lower': [ci_lower_real, ci_lower_synth],
    'CI_Upper': [ci_upper_real, ci_upper_synth]
})

print("\nBootstrap AUC Results:")
print(bootstrap_results.to_string(index=False))

print("\n" + "="*80)
print("EFFECT SIZE ANALYSIS")
print("="*80)

auc_diff = mean_synth - mean_real
pooled_std = np.sqrt((std_real**2 + std_synth**2) / 2)
cohens_d = auc_diff / pooled_std

print(f"\nAUC Difference (Synth - Real): {auc_diff:.4f}")
print(f"Pooled Standard Deviation: {pooled_std:.4f}")
print(f"Cohen's d: {cohens_d:.4f}")

if abs(cohens_d) < 0.2:
    effect_interpretation = "Negligible"
elif abs(cohens_d) < 0.5:
    effect_interpretation = "Small"
elif abs(cohens_d) < 0.8:
    effect_interpretation = "Medium"
else:
    effect_interpretation = "Large"

print(f"Effect Size Interpretation: {effect_interpretation}")

print("\n" + "="*80)
print("PAIRED T-TEST")
print("="*80)

def bootstrap_paired_test(y_true, y_proba_real, y_proba_synth, n_iterations=1000):
    from sklearn.metrics import roc_auc_score
    
    auc_diffs = []
    n_samples = len(y_true)
    
    for i in range(n_iterations):
        indices = resample(range(n_samples), n_samples=n_samples, random_state=i)
        
        y_true_boot = y_true.iloc[indices]
        y_proba_real_boot = y_proba_real.iloc[indices]
        y_proba_synth_boot = y_proba_synth.iloc[indices]
        
        auc_real = roc_auc_score(y_true_boot, y_proba_real_boot)
        auc_synth = roc_auc_score(y_true_boot, y_proba_synth_boot)
        
        auc_diffs.append(auc_synth - auc_real)
    
    return np.array(auc_diffs)

print("\nPerforming bootstrap paired test...")

auc_differences = bootstrap_paired_test(y_test, y_proba_real_xgb, y_proba_synth_xgb)

t_stat, p_value = stats.ttest_1samp(auc_differences, 0)

print(f"\nPaired t-test results:")
print(f"  Mean difference: {np.mean(auc_differences):.4f}")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4e}")

if p_value < 0.001:
    significance = "***"
elif p_value < 0.01:
    significance = "**"
elif p_value < 0.05:
    significance = "*"
else:
    significance = "ns"

print(f"  Significance: {significance}")

statistical_summary = pd.DataFrame({
    'Test': ['Bootstrap_AUC_Diff', 'Cohens_d', 'Paired_t_test'],
    'Statistic': [auc_diff, cohens_d, t_stat],
    'P_Value': [np.nan, np.nan, p_value],
    'Interpretation': [f'{auc_diff:.4f}', effect_interpretation, significance]
})

import os
os.makedirs('Results/Statistical', exist_ok=True)

bootstrap_results.to_csv('Results/Statistical/bootstrap_ci.csv', index=False)
statistical_summary.to_csv('Results/Statistical/statistical_summary.csv', index=False)

effect_size_df = pd.DataFrame({
    'Metric': ['AUC_Diff', 'Pooled_Std', 'Cohens_d', 'Interpretation'],
    'Value': [auc_diff, pooled_std, cohens_d, effect_interpretation]
})

effect_size_df.to_csv('Results/Statistical/effect_size.csv', index=False)

print("\nStatistical results saved to Results/Statistical/")