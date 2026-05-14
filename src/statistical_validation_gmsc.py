"""
GMSC Statistical Fidelity Analysis with Visualizations
Generates Tables 5.1-5.2 and Figures 5.1-5.3 for GMSC
KS tests, correlation analysis, distribution comparisons
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

# Required inputs:
# df_real: Real GMSC data (with SeriousDlqin2yrs target)
# df_synth: Synthetic GMSC data

# Verify alignment
assert list(df_real.columns) == list(df_synth.columns), "Column mismatch!"

print("="*80)
print("GMSC STATISTICAL FIDELITY ANALYSIS")
print("="*80)
print(f"Real shape: {df_real.shape}")
print(f"Synthetic shape: {df_synth.shape}")

TARGET = "SeriousDlqin2yrs"
predictors = [c for c in df_real.columns if c != TARGET]

# ===== TABLE 5.1: KS STATISTICS =====

ks_rows = []
for col in predictors:
    x = df_real[col].dropna().values
    y = df_synth[col].dropna().values
    
    ks_stat, p_val = stats.ks_2samp(x, y)
    
    if ks_stat < 0.10:
        interp = "Excellent"
    elif ks_stat < 0.20:
        interp = "Good"
    elif ks_stat < 0.30:
        interp = "Moderate"
    else:
        interp = "Poor"
    
    ks_rows.append({
        "Feature": col,
        "KS Statistic": round(float(ks_stat), 4),
        "p-value": float(p_val),
        "Interpretation": interp
    })

table_5_1 = pd.DataFrame(ks_rows).sort_values("KS Statistic", ascending=False)

table_5_1_path = TABLES_DIR / "table_5_1_ks_statistics_gmsc.csv"
table_5_1.to_csv(table_5_1_path, index=False)

print("\n  Table 5.1: KS Statistics")
print(f"Saved to: {table_5_1_path}")
print("\nTop 5 features by KS statistic:")
print(table_5_1.head()[['Feature', 'KS Statistic', 'Interpretation']].to_string(index=False))

# ===== TARGET DISTRIBUTION CHECK =====

real_rate = df_real[TARGET].mean()
synth_rate = df_synth[TARGET].mean()
delta_pp = (synth_rate - real_rate) * 100

print(f"\nTarget distribution:")
print(f"  Real default rate: {real_rate:.4f}")
print(f"  Synthetic default rate: {synth_rate:.4f}")
print(f"  Difference: {delta_pp:+.2f} percentage points")

# ===== FIGURE 5.1: DISTRIBUTION COMPARISONS =====

# Select 4 key features for visualization
features_to_plot = ["age", "MonthlyIncome", "DebtRatio", TARGET]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, feature in enumerate(features_to_plot):
    ax = axes[idx]
    
    real_data = df_real[feature].dropna()
    synth_data = df_synth[feature].dropna()
    
    if feature == TARGET:
        # Bar plot for binary target
        labels = ['No Default', 'Default']
        real_counts = [len(real_data[real_data==0]), len(real_data[real_data==1])]
        synth_counts = [len(synth_data[synth_data==0]), len(synth_data[synth_data==1])]
        
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, real_counts, width, label='Real', alpha=0.8)
        ax.bar(x + width/2, synth_counts, width, label='Synthetic', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Count')
        ax.set_title(f'{feature}', fontweight='bold')
        ax.legend()
    else:
        # Histogram for continuous features
        bins = 30
        ax.hist(real_data, bins=bins, alpha=0.6, label='Real', density=True, edgecolor='black')
        ax.hist(synth_data, bins=bins, alpha=0.6, label='Synthetic', density=True, edgecolor='black')
        
        # Add KS statistic
        ks_stat, _ = stats.ks_2samp(real_data, synth_data)
        ax.text(0.65, 0.95, f'KS={ks_stat:.3f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'{feature}', fontweight='bold')
        ax.legend()
    
    ax.grid(alpha=0.3)

plt.suptitle('GMSC: Distribution Comparison (Real vs Synthetic)',
             fontsize=14, fontweight='bold')
plt.tight_layout()

fig_5_1_path = OUTPUT_DIR / "figure_5_1_gmsc_distributions.png"
plt.savefig(fig_5_1_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_5_1_path.with_suffix('.pdf'), bbox_inches='tight')
plt.close()

print(f"\n  Figure 5.1 saved: {fig_5_1_path}")

# ===== TABLE 5.2: CORRELATION ANALYSIS =====

# Compute correlation matrices
corr_real = df_real[predictors].corr()
corr_synth = df_synth[predictors].corr()

# Flatten upper triangle for comparison
mask = np.triu(np.ones_like(corr_real, dtype=bool), k=1)
corr_real_flat = corr_real.where(mask).stack().values
corr_synth_flat = corr_synth.where(mask).stack().values

# Correlation between correlation matrices
corr_of_corr = np.corrcoef(corr_real_flat, corr_synth_flat)[0, 1]

# Mean absolute difference
mad = np.mean(np.abs(corr_real_flat - corr_synth_flat))

# Frobenius norm
frob_norm = np.linalg.norm(corr_real.values - corr_synth.values, 'fro')

table_5_2 = pd.DataFrame({
    "Metric": [
        "Correlation of correlations",
        "Mean absolute difference",
        "Frobenius norm",
        "Number of pairwise correlations"
    ],
    "Value": [
        round(corr_of_corr, 4),
        round(mad, 4),
        round(frob_norm, 4),
        len(corr_real_flat)
    ]
})

table_5_2_path = TABLES_DIR / "table_5_2_correlation_summary_gmsc.csv"
table_5_2.to_csv(table_5_2_path, index=False)

print(f"\n  Table 5.2: Correlation Summary")
print(f"Saved to: {table_5_2_path}")
print("\nCorrelation preservation:")
print(table_5_2.to_string(index=False))

# ===== FIGURE 5.2: CORRELATION HEATMAPS =====

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Real correlations
sns.heatmap(corr_real, ax=axes[0], cmap='coolwarm', center=0,
            vmin=-1, vmax=1, square=True, cbar_kws={'shrink': 0.8})
axes[0].set_title('Real Data Correlations', fontsize=12, fontweight='bold')

# Synthetic correlations
sns.heatmap(corr_synth, ax=axes[1], cmap='coolwarm', center=0,
            vmin=-1, vmax=1, square=True, cbar_kws={'shrink': 0.8})
axes[1].set_title('Synthetic Data Correlations', fontsize=12, fontweight='bold')

# Difference
corr_diff = corr_synth - corr_real
sns.heatmap(corr_diff, ax=axes[2], cmap='RdBu_r', center=0,
            vmin=-0.5, vmax=0.5, square=True, cbar_kws={'shrink': 0.8})
axes[2].set_title('Difference (Synth - Real)', fontsize=12, fontweight='bold')

plt.suptitle('GMSC: Correlation Structure Comparison',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

fig_5_2_path = OUTPUT_DIR / "figure_5_2_gmsc_correlation_heatmaps.png"
plt.savefig(fig_5_2_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_5_2_path.with_suffix('.pdf'), bbox_inches='tight')
plt.close()

print(f"  Figure 5.2 saved: {fig_5_2_path}")

# ===== FIGURE 5.3: CORRELATION SCATTER =====

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(corr_real_flat, corr_synth_flat, alpha=0.5, s=20)
ax.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect match')

ax.set_xlabel('Real Data Correlations', fontsize=12)
ax.set_ylabel('Synthetic Data Correlations', fontsize=12)
ax.set_title(f'GMSC: Correlation Preservation (r={corr_of_corr:.3f})',
             fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

fig_5_3_path = OUTPUT_DIR / "figure_5_3_gmsc_correlation_scatter.png"
plt.savefig(fig_5_3_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_5_3_path.with_suffix('.pdf'), bbox_inches='tight')
plt.close()

print(f" Figure 5.3 saved: {fig_5_3_path}")

print("\n" + "="*80)
print("GMSC STATISTICAL VALIDATION COMPLETE")
print("="*80)
print("\nGenerated:")
print(f"  - Table 5.1: KS statistics ({len(ks_rows)} features)")
print(f"  - Table 5.2: Correlation summary")
print(f"  - Figure 5.1: Distribution comparisons (4 features)")
print(f"  - Figure 5.2: Correlation heatmaps")
print(f"  - Figure 5.3: Correlation scatter plot")