"""
Explainability Analysis using XGBoost Feature Importance
Compares feature importance rankings between real and synthetic trained models
Generates Tables 5.13 and 5.14 for both GMSC and Home Credit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATASET = "GMSC"  # Change to "HomeCredit" for Home Credit analysis
BASE_DIR = Path('/path/to/dissertation')  # UPDATE THIS
TABLES_DIR = BASE_DIR / 'Chapter_5_Tables'
FIGURES_DIR = BASE_DIR / 'Chapter_5_Figures'

for d in [TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("="*80)
print(f"EXPLAINABILITY ANALYSIS: {DATASET}")
print("="*80)

# Required inputs (from model training):
# - xgb_real: XGBoost model trained on real data
# - xgb_synth: XGBoost model trained on synthetic data
# - feature_names: list of feature names used in training

# Extract feature importance (XGBoost gain)
def get_feature_importance(model, feature_names):
    """Extract XGBoost gain importance"""
    importance_dict = model.get_booster().get_score(importance_type='gain')
    
    # Map f0, f1, f2... back to feature names
    importance_data = []
    for i, feature_name in enumerate(feature_names):
        xgb_key = f'f{i}'
        importance_score = importance_dict.get(xgb_key, 0.0)
        importance_data.append({
            'Feature': feature_name,
            'Importance': importance_score
        })
    
    df = pd.DataFrame(importance_data)
    df = df.sort_values('Importance', ascending=False).reset_index(drop=True)
    df['Rank'] = range(1, len(df) + 1)
    
    return df

# Get importance from both models
importance_real = get_feature_importance(xgb_real, feature_names)
importance_synth = get_feature_importance(xgb_synth, feature_names)

print(f"\nTop 10 Features - Real Model:")
print(importance_real.head(10)[['Rank', 'Feature', 'Importance']].to_string(index=False))

print(f"\nTop 10 Features - Synthetic Model:")
print(importance_synth.head(10)[['Rank', 'Feature', 'Importance']].to_string(index=False))

# ===== TABLE 5.13: SPEARMAN RANK CORRELATION =====

# Merge and calculate rank correlation
merged = pd.merge(
    importance_real[['Feature', 'Rank', 'Importance']],
    importance_synth[['Feature', 'Rank', 'Importance']],
    on='Feature',
    suffixes=('_Real', '_Synth')
)

rho, p_value = spearmanr(merged['Rank_Real'], merged['Rank_Synth'])

print(f"\n{'='*80}")
print("FEATURE IMPORTANCE ALIGNMENT")
print('='*80)
print(f"Spearman's ρ: {rho:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    if rho > 0.7:
        print("  Strong significant correlation - rankings well preserved")
    elif rho > 0.5:
        print("⚠ Moderate significant correlation - some ranking shifts")
    else:
        print("✗ Weak correlation - substantial ranking changes")
else:
    print("⚠ Not statistically significant (p >= 0.05)")

# Save Table 5.13
if DATASET == "GMSC":
    table_513 = pd.DataFrame([{
        'Dataset': 'GMSC',
        'Model': 'XGBoost',
        'Ranking_Method': 'Gain importance',
        'Spearman_rho': rho,
        'p_value': p_value
    }])
    table_513_path = TABLES_DIR / 'table_5_13_importance_rank_spearman_gmsc_xgb.csv'
else:
    table_513 = pd.DataFrame([{
        'Dataset': 'HomeCredit',
        'Model': 'XGBoost',
        'Spearman_rho_rank_alignment': rho,
        'p_value': p_value,
        'n_features': len(feature_names)
    }])
    table_513_path = TABLES_DIR / 'table_5_13_importance_rank_spearman_homecredit_xgb.csv'

table_513.to_csv(table_513_path, index=False)
print(f"\n  Saved: {table_513_path}")

# ===== TABLE 5.14: RANK SHIFT ANALYSIS =====

# Calculate rank shifts
rank_shift = merged.copy()
rank_shift['Rank_Shift'] = rank_shift['Rank_Synth'] - rank_shift['Rank_Real']

# Sort by absolute rank shift to show biggest changes
rank_shift['Abs_Shift'] = rank_shift['Rank_Shift'].abs()
rank_shift = rank_shift.sort_values('Abs_Shift', ascending=False)

# Show top 10 largest shifts
print(f"\n{'='*80}")
print("TOP 10 LARGEST RANK SHIFTS")
print('='*80)
top_shifts = rank_shift.head(10)
print(top_shifts[['Feature', 'Rank_Real', 'Rank_Synth', 'Rank_Shift']].to_string(index=False))

# Save Table 5.14
if DATASET == "GMSC":
    table_514 = rank_shift[[
        'Feature', 'Rank_Real', 'Rank_Synth', 'Rank_Shift', 
        'Importance_Real', 'Importance_Synth'
    ]].copy()
    table_514_path = TABLES_DIR / 'table_5_14_rank_shift_gmsc_xgb.csv'
else:
    table_514 = rank_shift[[
        'Feature', 'Importance_Real', 'Importance_Synth',
        'Rank_Real', 'Rank_Synth', 'Rank_Shift'
    ]].copy()
    table_514.columns = [
        'Feature', 'Gain_Real', 'Gain_Synth',
        'Rank_Real', 'Rank_Synth', 'Rank_Shift_(Synth-Real)'
    ]
    table_514_path = TABLES_DIR / 'table_5_14_rank_shift_homecredit_xgb.csv'

# Save first 25 features (or all if fewer)
table_514.head(25).to_csv(table_514_path, index=False)
print(f"  Saved: {table_514_path}")

# ===== FIGURE: FEATURE IMPORTANCE COMPARISON =====

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Top 15 features from each model
top_15_real = importance_real.head(15)
top_15_synth = importance_synth.head(15)

# Real model
y_pos_real = np.arange(len(top_15_real))
ax1.barh(y_pos_real, top_15_real['Importance'].values, color='steelblue')
ax1.set_yticks(y_pos_real)
ax1.set_yticklabels(top_15_real['Feature'].values)
ax1.set_xlabel('XGBoost Gain Importance', fontsize=12)
ax1.set_title('Top 15 Features - Real Model', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(alpha=0.3, axis='x')

# Synthetic model
y_pos_synth = np.arange(len(top_15_synth))
ax2.barh(y_pos_synth, top_15_synth['Importance'].values, color='coral')
ax2.set_yticks(y_pos_synth)
ax2.set_yticklabels(top_15_synth['Feature'].values)
ax2.set_xlabel('XGBoost Gain Importance', fontsize=12)
ax2.set_title('Top 15 Features - Synthetic Model', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(alpha=0.3, axis='x')

plt.suptitle(f'{DATASET}: Feature Importance Comparison (Real vs Synthetic Trained)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

if DATASET == "GMSC":
    fig_path = FIGURES_DIR / 'figure_5_13_feature_importance_gmsc_xgb.png'
else:
    fig_path = FIGURES_DIR / 'figure_5_14_feature_importance_homecredit_xgb.png'

plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_path.with_suffix('.pdf'), bbox_inches='tight')
plt.close()

print(f"  Saved: {fig_path}")

# ===== RANK SHIFT VISUALIZATION =====

# Show features with largest rank changes
top_shifts_viz = rank_shift.head(15)

fig, ax = plt.subplots(figsize=(12, 8))

features = top_shifts_viz['Feature'].values
shifts = top_shifts_viz['Rank_Shift'].values
colors = ['red' if s > 0 else 'green' for s in shifts]

y_pos = np.arange(len(features))
ax.barh(y_pos, shifts, color=colors, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.set_xlabel('Rank Shift (Synth - Real)', fontsize=12)
ax.set_title(f'{DATASET}: Feature Rank Changes\n(Red = dropped in importance, Green = gained importance)',
             fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.invert_yaxis()
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()

if DATASET == "GMSC":
    shift_fig_path = FIGURES_DIR / 'figure_5_13b_rank_shift_gmsc_xgb.png'
else:
    shift_fig_path = FIGURES_DIR / 'figure_5_14b_rank_shift_homecredit_xgb.png'

plt.savefig(shift_fig_path, dpi=300, bbox_inches='tight')
plt.savefig(shift_fig_path.with_suffix('.pdf'), bbox_inches='tight')
plt.close()

print(f"  Saved: {shift_fig_path}")

print("\n" + "="*80)
print("EXPLAINABILITY ANALYSIS COMPLETE")
print("="*80)
print(f"\nGenerated:")
print(f"  - Table 5.13: Spearman rank correlation")
print(f"  - Table 5.14: Feature rank shifts")
print(f"  - Figures: Feature importance comparison + rank shifts")