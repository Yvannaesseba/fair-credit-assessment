"""
Fairness Analysis using Microsoft Fairlearn
Evaluates demographic parity and equal opportunity for both GMSC and Home Credit
Works with XGBoost models trained on real vs synthetic data
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix
)
import warnings

warnings.filterwarnings("ignore")

# Configuration
DATASET = "GMSC"  # Change to "HomeCredit" for Home Credit analysis
THRESHOLD = 0.5
BASE_DIR = Path('/path/to/dissertation')  # UPDATE THIS
OUTPUT_DIR = BASE_DIR / 'Chapter_5_Figures'
TABLES_DIR = BASE_DIR / 'Chapter_5_Tables'

for d in [OUTPUT_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Helper functions
def make_age_groups(age_years):
    """Create age groups: Young (<=35), Middle (36-55), Senior (56+)"""
    bins = [-np.inf, 35, 55, np.inf]
    labels = ["Young (<=35)", "Middle (36-55)", "Senior (56+)"]
    return pd.cut(age_years, bins=bins, labels=labels)

def derive_age_years_homecredit(df):
    """Extract age from Home Credit dataset"""
    if "AGE_YEARS" in df.columns:
        return pd.to_numeric(df["AGE_YEARS"], errors="coerce")
    if "DAYS_BIRTH" in df.columns:
        return (-pd.to_numeric(df["DAYS_BIRTH"], errors="coerce") / 365.0)
    if "age" in df.columns:
        return pd.to_numeric(df["age"], errors="coerce")
    raise ValueError("Could not derive age years for Home Credit")

def derive_age_years_gmsc(df):
    """Extract age from GMSC dataset"""
    if "age" in df.columns:
        return pd.to_numeric(df["age"], errors="coerce")
    raise ValueError("GMSC expected column 'age' not found")

def predict_probabilities(model, X):
    """Get prediction probabilities"""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raise ValueError("Model has no predict_proba method")

def compute_group_fairness(y_true, y_prob, sensitive_series, threshold=0.5):
    """Compute fairness metrics by demographic group"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    
    def true_positive_rate(y_t, y_p):
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def false_positive_rate(y_t, y_p):
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    metrics = {
        "SelectionRate": selection_rate,
        "TPR": true_positive_rate,
        "FPR": false_positive_rate,
        "Precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "Recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "F1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "Accuracy": accuracy_score,
    }
    
    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_series
    )
    
    by_group = mf.by_group.reset_index()
    by_group = by_group.rename(columns={by_group.columns[0]: "Group"})
    overall = mf.overall
    
    # Demographic parity (selection rate spread)
    sr = by_group[["Group", "SelectionRate"]].copy()
    dp_diff = sr["SelectionRate"].max() - sr["SelectionRate"].min()
    
    # Equal opportunity (TPR spread)
    tpr = by_group[["Group", "TPR"]].copy()
    eo_diff = tpr["TPR"].max() - tpr["TPR"].min()
    
    summary = {
        "DP_Difference_(SelectionRate_Spread)": float(dp_diff),
        "EO_Difference_(TPR_Spread)": float(eo_diff),
    }
    
    return by_group, overall, summary

def plot_metric_bars(by_group_real, by_group_synth, metric_col, title, save_png, save_pdf):
    """Create bar chart comparing real vs synthetic trained models"""
    groups = by_group_real["Group"].astype(str).tolist()
    
    real_vals = by_group_real.set_index("Group")[metric_col].reindex(groups).values
    synth_vals = by_group_synth.set_index("Group")[metric_col].reindex(groups).values
    
    x = np.arange(len(groups))
    width = 0.35
    
    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, real_vals, width, label="Real-trained")
    plt.bar(x + width/2, synth_vals, width, label="Synthetic-trained")
    plt.xticks(x, groups, rotation=0)
    plt.ylabel(metric_col)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25, axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf, bbox_inches="tight")
    plt.close()

# ===== MAIN ANALYSIS =====
# Required inputs (must be provided from model training script):
# - df_test_real: test dataframe with TARGET and age column(s)
# - X_test_real: preprocessed features for models
# - y_test_real: true labels
# - xgb_real: XGBoost model trained on real data
# - xgb_synth: XGBoost model trained on synthetic data

# These should be loaded/passed from your model training script
# Example:
# from model_training import df_test_real, X_test_real, y_test_real, xgb_real, xgb_synth

print(f"Running fairness analysis for {DATASET}")
print("="*80)

# Derive age groups
if DATASET == "GMSC":
    age_years = derive_age_years_gmsc(df_test_real)
else:
    age_years = derive_age_years_homecredit(df_test_real)

age_group = make_age_groups(age_years).astype(str)

# Get predictions on REAL test set
y_prob_real = predict_probabilities(xgb_real, X_test_real)
y_prob_synth = predict_probabilities(xgb_synth, X_test_real)

# Compute fairness metrics
by_group_real, overall_real, summary_real = compute_group_fairness(
    y_true=y_test_real, 
    y_prob=y_prob_real, 
    sensitive_series=age_group, 
    threshold=THRESHOLD
)

by_group_synth, overall_synth, summary_synth = compute_group_fairness(
    y_true=y_test_real, 
    y_prob=y_prob_synth, 
    sensitive_series=age_group, 
    threshold=THRESHOLD
)

# Prepare tables
by_group_real["TrainingData"] = "Real"
by_group_synth["TrainingData"] = "Synthetic"

fair_table = pd.concat([by_group_real, by_group_synth], ignore_index=True)
fair_table.insert(0, "Dataset", DATASET)

# Save detailed fairness table
if DATASET == "GMSC":
    table_name = "table_5_10_fairness_gmsc_by_age.csv"
else:
    table_name = "table_5_11_fairness_homecredit_by_age.csv"

fair_table.to_csv(TABLES_DIR / table_name, index=False)

# Save summary (DP/EO spreads)
summary_df = pd.DataFrame([
    {"Dataset": DATASET, "TrainingData": "Real", **summary_real},
    {"Dataset": DATASET, "TrainingData": "Synthetic", **summary_synth},
])

if DATASET == "GMSC":
    summary_name = "table_5_12_fairness_summary_spreads_gmsc.csv"
else:
    summary_name = "table_5_12_fairness_summary_spreads_homecredit.csv"

summary_df.to_csv(TABLES_DIR / summary_name, index=False)

print(f"\nSaved fairness tables:")
print(f"  {TABLES_DIR / table_name}")
print(f"  {TABLES_DIR / summary_name}")

# Create figures
if DATASET == "GMSC":
    fig_sel_png = OUTPUT_DIR / "figure_5_11_selection_rate_by_age_gmsc.png"
    fig_sel_pdf = OUTPUT_DIR / "figure_5_11_selection_rate_by_age_gmsc.pdf"
    fig_tpr_png = OUTPUT_DIR / "figure_5_11_tpr_by_age_gmsc.png"
    fig_tpr_pdf = OUTPUT_DIR / "figure_5_11_tpr_by_age_gmsc.pdf"
else:
    fig_sel_png = OUTPUT_DIR / "figure_5_12_selection_rate_by_age_homecredit.png"
    fig_sel_pdf = OUTPUT_DIR / "figure_5_12_selection_rate_by_age_homecredit.pdf"
    fig_tpr_png = OUTPUT_DIR / "figure_5_12_tpr_by_age_homecredit.png"
    fig_tpr_pdf = OUTPUT_DIR / "figure_5_12_tpr_by_age_homecredit.pdf"

# Selection rate (demographic parity proxy)
plot_metric_bars(
    by_group_real, by_group_synth,
    metric_col="SelectionRate",
    title=f"{DATASET}: Selection Rate by Age Group (Real vs Synthetic Trained)",
    save_png=fig_sel_png,
    save_pdf=fig_sel_pdf
)

# TPR (equal opportunity proxy)
plot_metric_bars(
    by_group_real, by_group_synth,
    metric_col="TPR",
    title=f"{DATASET}: True Positive Rate by Age Group (Real vs Synthetic Trained)",
    save_png=fig_tpr_png,
    save_pdf=fig_tpr_pdf
)

print(f"\nSaved fairness figures to {OUTPUT_DIR}")
print("\nFairness analysis complete!")
print("="*80)