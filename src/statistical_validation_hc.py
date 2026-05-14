"""
Home Credit Statistical Fidelity Analysis with Visualizations
Generates Tables 5.3-5.5 and Figures 5.3-5.5 for Home Credit
KS tests, chi-square tests, correlation analysis, categorical integrity
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Configuration
BASE_DIR = Path('/path/to/dissertation')  # UPDATE THIS
OUTPUT_DIR = BASE_DIR / 'Chapter_5_Figures'
TABLES_DIR = BASE_DIR / 'Chapter_5_Tables'

for d in [OUTPUT_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Publication-friendly styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

REAL_COLOR = "#2E86AB"
SYN_COLOR  = "#A23B72"

# Load data
print("Loading Home Credit data...")

REAL_PATH = BASE_DIR / 'Feature_Selection_Results' / 'train_ctgan_ready.csv'
SYN_PATH = BASE_DIR / 'Results' / 'CTGAN' / 'synth_train_ctgan.csv'

df_real = pd.read_csv(REAL_PATH)
df_synth = pd.read_csv(SYN_PATH)

print(f"  Real loaded: {df_real.shape}")
print(f"  Synthetic loaded: {df_synth.shape}")

# Schema alignment
real_cols = list(df_real.columns)

extra_in_syn = [c for c in df_synth.columns if c not in real_cols]
missing_in_syn = [c for c in real_cols if c not in df_synth.columns]

if extra_in_syn:
    print(f" Dropping {len(extra_in_syn)} extra cols in synthetic (not in real).")
    df_synth = df_synth.drop(columns=extra_in_syn)

if missing_in_syn:
    print(f" Synthetic is missing {len(missing_in_syn)} cols found in real.")
    print("   They will be added as NaN (your results will reflect this).")
    for c in missing_in_syn:
        df_synth[c] = np.nan

# Reorder synthetic to match real column order
df_synth = df_synth[real_cols]

# ----------------------------
# 4) Column typing
# ----------------------------
# Your explicit discrete/binary list (keep this explicit)
DISCRETE_COLS = ['FLAG_DOCUMENT_3', 'FLAG_PHONE', 'TARGET']

# Categorical: object dtype excluding target
categorical_cols = [c for c in df_real.columns if df_real[c].dtype == "object" and c != "TARGET"]

# Continuous numeric: numeric columns excluding DISCRETE and excluding categoricals
numeric_cols = df_real.select_dtypes(include=[np.number]).columns.tolist()
continuous_cols = [c for c in numeric_cols if c not in DISCRETE_COLS]

print(f"\nColumn summary:")
print(f"  • Categorical (object): {len(categorical_cols)}")
print(f"  • Discrete/binary:      {len(DISCRETE_COLS)} -> {DISCRETE_COLS}")
print(f"  • Continuous numeric:   {len(continuous_cols)}")

# ----------------------------
# Helper: safe KS (drops NaNs)
# ----------------------------
def ks_safe(a, b):
    a = pd.Series(a).dropna().values
    b = pd.Series(b).dropna().values
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    return stats.ks_2samp(a, b)

# Helper: chi-square for discrete/categorical with rare-category collapsing
def chi2_with_other(real_series, synth_series, min_count=50):
    real_counts  = real_series.value_counts(dropna=False)
    synth_counts = synth_series.value_counts(dropna=False)

    # Align categories
    all_cats = pd.Index(sorted(set(real_counts.index) | set(synth_counts.index)))

    real_aligned  = real_counts.reindex(all_cats, fill_value=0)
    synth_aligned = synth_counts.reindex(all_cats, fill_value=0)

    # Collapse rare categories into "Other" (stabilises chi-square)
    mask_keep = (real_aligned >= min_count) | (synth_aligned >= min_count)
    kept = all_cats[mask_keep]
    dropped = all_cats[~mask_keep]

    if len(dropped) > 0:
        real_other  = real_aligned.loc[dropped].sum()
        synth_other = synth_aligned.loc[dropped].sum()

        real_aligned  = real_aligned.loc[kept]
        synth_aligned = synth_aligned.loc[kept]

        real_aligned.loc["Other"]  = real_other
        synth_aligned.loc["Other"] = synth_other

    # Build contingency table
    cont = np.vstack([real_aligned.values, synth_aligned.values])

    # If degenerate, return NaNs
    if cont.sum() == 0 or cont.shape[1] < 2:
        return np.nan, np.nan, len(real_counts), len(synth_counts), len(real_aligned)

    chi2, p, dof, exp = stats.chi2_contingency(cont)
    return chi2, p, len(real_counts), len(synth_counts), len(real_aligned)

# =============================================================================
# SECTION 5.2.1: UNIVARIATE DISTRIBUTIONS (CURATED FIGURE)
# =============================================================================
print("\n" + "="*80)
print("SECTION 5.2.1: UNIVARIATE DISTRIBUTIONS (HOME CREDIT)")
print("="*80)

key_features = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'DAYS_EMPLOYED', 'TARGET']
feature_labels = {
    'EXT_SOURCE_3': 'External Source 3',
    'EXT_SOURCE_2': 'External Source 2',
    'DAYS_EMPLOYED': 'Days Employed',
    'TARGET': 'Default (Binary)'
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    ax = axes[i]

    if feat == "TARGET":
        real_prop  = df_real[feat].value_counts(normalize=True).sort_index()
        synth_prop = df_synth[feat].value_counts(normalize=True).sort_index()

        # Ensure both indices exist
        idx_union = sorted(set(real_prop.index) | set(synth_prop.index))
        real_prop  = real_prop.reindex(idx_union, fill_value=0)
        synth_prop = synth_prop.reindex(idx_union, fill_value=0)

        x = np.arange(len(idx_union))
        w = 0.35
        ax.bar(x - w/2, real_prop.values, w, label="Real", color=REAL_COLOR, alpha=0.7, edgecolor="black")
        ax.bar(x + w/2, synth_prop.values, w, label="Synthetic", color=SYN_COLOR, alpha=0.7, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(["No Default" if v == 0 else "Default" for v in idx_union])
        ax.set_ylabel("Proportion")

        # For binary, show absolute diff instead of KS
        abs_diff = float(np.abs(real_prop.values - synth_prop.values).max())
        annot = f"Max |Δp| = {abs_diff:.4f}"

    else:
        ks_stat, p_val = ks_safe(df_real[feat], df_synth[feat])

        ax.hist(pd.Series(df_real[feat]).dropna(), bins=50, density=True,
                alpha=0.45, label="Real", color=REAL_COLOR, edgecolor="black")
        ax.hist(pd.Series(df_synth[feat]).dropna(), bins=50, density=True,
                alpha=0.45, label="Synthetic", color=SYN_COLOR, edgecolor="black")
        ax.set_ylabel("Density")
        annot = f"KS = {ks_stat:.4f}"

    ax.text(0.95, 0.95, annot,
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            fontsize=9)

    ax.set_xlabel(feature_labels.get(feat, feat))
    ax.set_title(feature_labels.get(feat, feat), fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=9, frameon=True)

plt.tight_layout()
FIG_53_PNG = f"{OUTPUT_DIR}/figure_5_3_hc_distributions.png"
FIG_53_PDF = f"{OUTPUT_DIR}/figure_5_3_hc_distributions.pdf"
plt.savefig(FIG_53_PNG, dpi=300, bbox_inches="tight")
plt.savefig(FIG_53_PDF, bbox_inches="tight")
plt.show()
print(f" Saved: {FIG_53_PNG}")
print(f" Saved: {FIG_53_PDF}")

# =============================================================================
# TABLE 5.3: KS STATISTICS (CONTINUOUS ONLY) + p-values
# =============================================================================
ks_rows = []
for feat in continuous_cols:
    ks_stat, p_val = ks_safe(df_real[feat], df_synth[feat])

    if np.isnan(ks_stat):
        interp = "N/A"
    elif ks_stat < 0.1:
        interp = "Excellent"
    elif ks_stat < 0.2:
        interp = "Good"
    elif ks_stat < 0.3:
        interp = "Moderate"
    else:
        interp = "Poor"

    ks_rows.append({
        "Feature": feat,
        "KS Statistic": np.round(ks_stat, 4) if not np.isnan(ks_stat) else np.nan,
        "p-value": p_val,
        "Interpretation": interp,
        "Real_N_nonnull": int(pd.Series(df_real[feat]).notna().sum()),
        "Synth_N_nonnull": int(pd.Series(df_synth[feat]).notna().sum())
    })

table_5_3 = pd.DataFrame(ks_rows).sort_values("KS Statistic", ascending=False, na_position="last")
TAB_53 = f"{TABLES_DIR}/table_5_3_ks_statistics_hc.csv"
table_5_3.to_csv(TAB_53, index=False)
print(f"\n  Saved: {TAB_53}")
print("Top 10 (largest KS):")
print(table_5_3.head(10).to_string(index=False))

# =============================================================================
# SECTION 5.2.2: CORRELATION PRESERVATION (CONTINUOUS ONLY)
# =============================================================================
print("\n" + "="*80)
print("SECTION 5.2.2: CORRELATION PRESERVATION (HOME CREDIT)")
print("="*80)

# Compute Spearman correlations (pairwise complete observations handled by pandas)
corr_real  = df_real[continuous_cols].corr(method="spearman")
corr_synth = df_synth[continuous_cols].corr(method="spearman")
corr_diff  = corr_synth - corr_real

# Overall correlation preservation (upper triangle)
mask = np.triu(np.ones_like(corr_real, dtype=bool), k=1)
real_vals  = corr_real.where(mask).stack().values
synth_vals = corr_synth.where(mask).stack().values

overall_rho, overall_p = stats.spearmanr(real_vals, synth_vals)
mean_abs_diff = float(np.abs(corr_diff.values[mask]).mean())
max_abs_diff  = float(np.abs(corr_diff.values[mask]).max())
rmse          = float(np.sqrt(np.mean((corr_diff.values[mask])**2)))

# Plot heatmaps
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(corr_real,  cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, ax=axes[0], cbar_kws={"label": "Spearman ρ"})
axes[0].set_title("Real Data", fontsize=12, fontweight="bold")

sns.heatmap(corr_synth, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, ax=axes[1], cbar_kws={"label": "Spearman ρ"})
axes[1].set_title("Synthetic Data", fontsize=12, fontweight="bold")

sns.heatmap(corr_diff,  cmap="RdBu_r", center=0, vmin=-0.3, vmax=0.3,
            square=True, ax=axes[2], cbar_kws={"label": "Δρ (Synthetic − Real)"})
axes[2].set_title("Difference", fontsize=12, fontweight="bold")

fig.text(0.5, 0.02,
         f"Overall Correlation Preservation: ρ = {overall_rho:.4f} (p < 0.0001)",
         ha="center", fontsize=11, fontweight="bold",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

plt.tight_layout(rect=[0, 0.03, 1, 1])
FIG_54_PNG = f"{OUTPUT_DIR}/figure_5_4_hc_correlation.png"
FIG_54_PDF = f"{OUTPUT_DIR}/figure_5_4_hc_correlation.pdf"
plt.savefig(FIG_54_PNG, dpi=300, bbox_inches="tight")
plt.savefig(FIG_54_PDF, bbox_inches="tight")
plt.show()

print(f"  Saved: {FIG_54_PNG}")
print(f"  Saved: {FIG_54_PDF}")
print(f"  Overall ρ = {overall_rho:.4f}")

# Table 5.4
table_5_4 = pd.DataFrame({
    "Metric": ["Spearman ρ (matrices)", "p-value", "Mean |Δρ|", "Max |Δρ|", "RMSE"],
    "Value":  [f"{overall_rho:.4f}", "< 0.0001", f"{mean_abs_diff:.4f}", f"{max_abs_diff:.4f}", f"{rmse:.4f}"]
})
TAB_54 = f"{TABLES_DIR}/table_5_4_correlation_summary_hc.csv"
table_5_4.to_csv(TAB_54, index=False)
print(f" Saved: {TAB_54}")

# =============================================================================
# SECTION 5.2.3: CATEGORICAL DISTORTION (CHI-SQUARE + CATEGORY INTEGRITY)
# =============================================================================
print("\n" + "="*80)
print("SECTION 5.2.3: CATEGORICAL DISTORTION (HOME CREDIT)")
print("="*80)

cat_rows = []

# 5.2.3a) Object categoricals
for col in categorical_cols:
    chi2, p, real_k, synth_k, k_after = chi2_with_other(df_real[col], df_synth[col], min_count=50)

    if np.isnan(chi2):
        status = "N/A"
    elif chi2 < 100:
        status = "Success"
    elif chi2 < 1000:
        status = "Moderate"
    else:
        status = "Failure"

    cat_rows.append({
        "Feature": col,
        "Type": "categorical(object)",
        "Chi-square": np.round(chi2, 2) if not np.isnan(chi2) else np.nan,
        "p-value": p,
        "Real categories": real_k,
        "Synth categories": synth_k,
        "Categories used (after Other)": k_after,
        "Status": status
    })

# 5.2.3b) Discrete/binary columns (validate domain + chi-square)
for col in DISCRETE_COLS:
    # domain integrity
    real_unique = set(pd.Series(df_real[col]).dropna().unique().tolist())
    syn_unique  = set(pd.Series(df_synth[col]).dropna().unique().tolist())

    # Chi-square without "Other" (binary should be stable)
    chi2, p, real_k, synth_k, k_after = chi2_with_other(df_real[col], df_synth[col], min_count=1)

    if np.isnan(chi2):
        status = "N/A"
    elif chi2 < 50:
        status = "Success"
    elif chi2 < 200:
        status = "Moderate"
    else:
        status = "Failure"

    cat_rows.append({
        "Feature": col,
        "Type": "discrete/binary",
        "Chi-square": np.round(chi2, 2) if not np.isnan(chi2) else np.nan,
        "p-value": p,
        "Real categories": real_k,
        "Synth categories": synth_k,
        "Categories used (after Other)": k_after,
        "Status": status,
        "Real unique values": str(sorted(list(real_unique)))[:60],
        "Synth unique values": str(sorted(list(syn_unique)))[:60]
    })

table_5_5 = pd.DataFrame(cat_rows).sort_values("Chi-square", ascending=False, na_position="last")
TAB_55 = f"{TABLES_DIR}/table_5_5_categorical_corruption_hc.csv"
table_5_5.to_csv(TAB_55, index=False)
print(f"\n Saved: {TAB_55}")
print(table_5_5.head(15).to_string(index=False))

# =============================================================================
# FIGURE 5.5: Example categorical comparison plot
# (Pick a stable, meaningful categorical feature you know exists)
# =============================================================================
example_feature = "NAME_FAMILY_STATUS"
if example_feature in df_real.columns and example_feature in df_synth.columns:

    real_counts  = df_real[example_feature].value_counts()
    synth_counts = df_synth[example_feature].value_counts()

    # Keep top categories to avoid unreadable plots
    TOPK = 10
    real_top = real_counts.head(TOPK)
    synth_top = synth_counts.head(TOPK)

    # Union categories for fair plotting
    cats = list(pd.Index(real_top.index).union(pd.Index(synth_top.index)))

    real_plot  = real_counts.reindex(cats, fill_value=0)
    synth_plot = synth_counts.reindex(cats, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(len(cats)), real_plot.values, color=REAL_COLOR, alpha=0.7, edgecolor="black")
    axes[0].set_xticks(range(len(cats)))
    axes[0].set_xticklabels(cats, rotation=45, ha="right")
    axes[0].set_title(f"Real: {example_feature} (Top {TOPK})", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(range(len(cats)), synth_plot.values, color=SYN_COLOR, alpha=0.7, edgecolor="black")
    axes[1].set_xticks(range(len(cats)))
    axes[1].set_xticklabels(cats, rotation=45, ha="right")
    axes[1].set_title(f"Synthetic: {example_feature} (Top {TOPK})", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3, axis="y")

    # pull chi2 for this feature if present
    chi2_val = table_5_5.loc[table_5_5["Feature"] == example_feature, "Chi-square"]
    chi2_text = f"{chi2_val.values[0]:,.0f}" if len(chi2_val) else "N/A"

    fig.text(0.5, 0.02,
             f"Chi-square Statistic (with rare categories collapsed): {chi2_text}",
             ha="center", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    FIG_55_PNG = f"{OUTPUT_DIR}/figure_5_5_categorical_corruption.png"
    FIG_55_PDF = f"{OUTPUT_DIR}/figure_5_5_categorical_corruption.pdf"
    plt.savefig(FIG_55_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(FIG_55_PDF, bbox_inches="tight")
    plt.show()

    print(f" Saved: {FIG_55_PNG}")
    print(f" Saved: {FIG_55_PDF}")
else:
    print(f" Example feature '{example_feature}' not found in both datasets. Skipping Figure 5.5.")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SECTION 5.2 (HOME CREDIT) - COMPLETE")
print("="*80)

print("\n Tables Generated (Drive):")
print(f"  • {TAB_53}")
print(f"  • {TAB_54}")
print(f"  • {TAB_55}")

print("\n Figures Generated (Drive):")
print(f"  • {FIG_53_PNG} (+pdf)")
print(f"  • {FIG_54_PNG} (+pdf)")
print(f"  • {OUTPUT_DIR}/figure_5_5_categorical_corruption.png (+pdf)  [if feature exists]")

print("\n Key stats:")
print(f"  • Overall correlation preservation ρ = {overall_rho:.4f}")
print(f"  • Mean |Δρ| = {mean_abs_diff:.4f}, Max |Δρ| = {max_abs_diff:.4f}, RMSE = {rmse:.4f}")