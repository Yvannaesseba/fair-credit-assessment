import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import os

TARGET_COL = "SeriousDlqin2yrs"
ID_COL = "ID"
RANDOM_STATE = 42

gmsc = pd.read_csv('Data/Processed/gmsc_cleaned.csv')

print(f"Data shape: {gmsc.shape}")
print(f"Missing values: {gmsc.isnull().sum().sum()}")

target_counts = gmsc[TARGET_COL].value_counts().sort_index()
target_props = gmsc[TARGET_COL].value_counts(normalize=True).sort_index()

print("\nTarget distribution:")
for val in target_counts.index:
    print(f"  {val}: {target_counts[val]} ({target_props[val]:.3%})")

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=gmsc)

if ID_COL in metadata.to_dict()["columns"]:
    metadata.update_column(column_name=ID_COL, sdtype="id")

metadata.validate_data(data=gmsc)
print("\nMetadata validated")

os.makedirs('Data/Metadata', exist_ok=True)
metadata.save_to_json('Data/Metadata/gmsc_metadata.json')

print("\nTraining CTGAN on full dataset...")
synthesizer = CTGANSynthesizer(
    metadata=metadata,
    epochs=300,
    batch_size=500,
    verbose=True
)

synthesizer.fit(gmsc)
print("CTGAN training complete")

print("\nGenerating synthetic data with conditional sampling...")

real_target_dist = gmsc[TARGET_COL].value_counts(normalize=True).sort_index()
FULL_SYNTH_ROWS = gmsc.shape[0]
real_default_rate = real_target_dist.get(1, 0.0)

n_defaults = int(FULL_SYNTH_ROWS * real_default_rate)
n_non_defaults = FULL_SYNTH_ROWS - n_defaults

print(f"\nTarget counts:")
print(f"  Class 0: {n_non_defaults:,} ({(n_non_defaults/FULL_SYNTH_ROWS)*100:.2f}%)")
print(f"  Class 1: {n_defaults:,} ({(n_defaults/FULL_SYNTH_ROWS)*100:.2f}%)")

conditions_default = pd.DataFrame({TARGET_COL: [1] * n_defaults})
conditions_non_default = pd.DataFrame({TARGET_COL: [0] * n_non_defaults})

print("\nGenerating defaults...")
synth_defaults = synthesizer.sample_remaining_columns(known_columns=conditions_default)

print("Generating non-defaults...")
synth_non_defaults = synthesizer.sample_remaining_columns(known_columns=conditions_non_default)

gmsc_synth_full = pd.concat([synth_defaults, synth_non_defaults], ignore_index=True)
gmsc_synth_full = gmsc_synth_full.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"\nSynthetic data shape: {gmsc_synth_full.shape}")

synth_target_dist = gmsc_synth_full[TARGET_COL].value_counts(normalize=True).sort_index()

print("\nTarget distribution comparison:")
print("Real:")
print(real_target_dist)
print("\nSynthetic:")
print(synth_target_dist)

os.makedirs('Data/Synthetic', exist_ok=True)
gmsc_synth_full.to_csv('Data/Synthetic/gmsc_synthetic_ctgan.csv', index=False)

print("\nSynthetic data saved to Data/Synthetic/gmsc_synthetic_ctgan.csv")