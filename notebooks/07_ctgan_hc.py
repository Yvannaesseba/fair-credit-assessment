"""
Home Credit Synthetic Data Generation with CTGAN
Generates synthetic training data while preserving statistical properties
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from pathlib import Path
import json
import gc
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance, spearmanr
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import torch

warnings.filterwarnings("ignore")

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# Paths
BASE_DIR = Path('/path/to/dissertation')  # UPDATE THIS
FS_DIR = BASE_DIR / 'Feature_Selection_Results'
MODEL_DIR = BASE_DIR / 'Models'
RESULTS_DIR = BASE_DIR / 'Results' / 'CTGAN'
FIGURES_DIR = RESULTS_DIR / 'Figures'

for d in [MODEL_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load CTGAN-ready data from feature selection
train_df = pd.read_csv(FS_DIR / "train_ctgan_ready.csv")
val_df = pd.read_csv(FS_DIR / "val_ctgan_ready.csv")
test_df = pd.read_csv(FS_DIR / "test_ctgan_ready.csv")

# Identify column types
categorical_cols = [c for c in train_df.columns if train_df[c].dtype == "object" and c != "TARGET"]
numeric_cols = [c for c in train_df.columns if c not in categorical_cols and c != "TARGET"]

discrete_cols = []
continuous_cols = []

for c in numeric_cols:
    nunique = train_df[c].nunique(dropna=False)
    if c.startswith(("FLAG_", "CNT_")) or nunique <= 20:
        discrete_cols.append(c)
    else:
        continuous_cols.append(c)

# Build metadata for CTGAN
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_df)

for col in categorical_cols:
    metadata.update_column(column_name=col, sdtype='categorical')

for col in discrete_cols:
    metadata.update_column(column_name=col, sdtype='numerical', computer_representation='Int64')

for col in continuous_cols:
    metadata.update_column(column_name=col, sdtype='numerical', computer_representation='Float')

metadata.update_column(column_name='TARGET', sdtype='categorical')
metadata.set_primary_key(None)

# Smoke test (fast pre-flight check)
ctgan_test = CTGANSynthesizer(
    metadata=metadata,
    epochs=2,
    verbose=False,
    cuda=torch.cuda.is_available()
)

ctgan_test.fit(train_df.head(1000))
smoke_sample = ctgan_test.sample(10)
assert smoke_sample.shape[1] == train_df.shape[1], "Column count mismatch"
assert set(smoke_sample.columns) == set(train_df.columns), "Column names mismatch"

del ctgan_test, smoke_sample
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Full CTGAN training
EPOCHS = 300  # Adjust based on dataset complexity

ctgan = CTGANSynthesizer(
    metadata=metadata,
    epochs=EPOCHS,
    verbose=True,
    cuda=torch.cuda.is_available(),
    batch_size=500,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    generator_lr=2e-4,
    discriminator_lr=2e-4,
    discriminator_steps=1,
    pac=10
)

training_start = datetime.now()
ctgan.fit(train_df)
training_duration = (datetime.now() - training_start).total_seconds()

# Generate synthetic data (same size as training set)
synth_train_df = ctgan.sample(num_rows=len(train_df))

# Post-processing: enforce constraints
for col in discrete_cols:
    if col in synth_train_df.columns:
        synth_train_df[col] = synth_train_df[col].round().astype('Int64')

if 'TARGET' in synth_train_df.columns:
    synth_train_df['TARGET'] = synth_train_df['TARGET'].apply(
        lambda x: 1 if x >= 0.5 else 0
    ).astype(int)

# Statistical validation
validation_results = []

# KS test for continuous features
for col in continuous_cols:
    if col in train_df.columns and col in synth_train_df.columns:
        real_col = train_df[col].replace(-999, np.nan).dropna()
        synth_col = synth_train_df[col].replace(-999, np.nan).dropna()
        
        if len(real_col) > 0 and len(synth_col) > 0:
            ks_stat, p_val = ks_2samp(real_col, synth_col)
            validation_results.append({
                'Feature': col,
                'Type': 'Continuous',
                'KS_Statistic': ks_stat,
                'p_value': p_val
            })

# Chi-square test for categorical features
for col in categorical_cols + ['TARGET']:
    if col in train_df.columns and col in synth_train_df.columns:
        real_counts = train_df[col].value_counts()
        synth_counts = synth_train_df[col].value_counts()
        
        all_categories = set(real_counts.index) | set(synth_counts.index)
        
        if len(all_categories) > 1:
            real_freq = [real_counts.get(cat, 0) for cat in all_categories]
            synth_freq = [synth_counts.get(cat, 0) for cat in all_categories]
            
            contingency = np.array([real_freq, synth_freq])
            if contingency.sum() > 0:
                chi2, p_val, _, _ = chi2_contingency(contingency)
                validation_results.append({
                    'Feature': col,
                    'Type': 'Categorical',
                    'Chi2_Statistic': chi2,
                    'p_value': p_val
                })

validation_df = pd.DataFrame(validation_results)

# Save outputs
synth_train_df.to_csv(RESULTS_DIR / 'synth_train_ctgan.csv', index=False)
validation_df.to_csv(RESULTS_DIR / 'statistical_validation.csv', index=False)

ctgan.save(str(MODEL_DIR / 'ctgan_model.pkl'))

# Save metadata report
report = {
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'training_duration_seconds': training_duration,
    'epochs': EPOCHS,
    'real_train_shape': train_df.shape,
    'synth_train_shape': synth_train_df.shape,
    'categorical_features': len(categorical_cols),
    'discrete_features': len(discrete_cols),
    'continuous_features': len(continuous_cols),
    'cuda_available': torch.cuda.is_available(),
    'real_default_rate': float(train_df['TARGET'].mean()),
    'synth_default_rate': float(synth_train_df['TARGET'].mean())
}

with open(RESULTS_DIR / 'generation_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"Synthetic data generation complete")
print(f"Training duration: {training_duration/60:.1f} minutes")
print(f"Synthetic samples: {len(synth_train_df):,}")
print(f"Results saved to {RESULTS_DIR}")