import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import os

TARGET_COL = "SeriousDlqin2yrs"
RANDOM_STATE = 42

if __name__ == "__main__":

    # load the cleaned GMSC data produced by 01_eda_gmsc.py
    print("Loading cleaned GMSC data...")
    gmsc = pd.read_csv('data/gmsc_cleaned.csv')

    print(f"Data shape: {gmsc.shape}")
    print(f"Missing values: {gmsc.isnull().sum().sum()}")

    target_counts = gmsc[TARGET_COL].value_counts().sort_index()
    target_props = gmsc[TARGET_COL].value_counts(normalize=True).sort_index()

    print("\nTarget distribution:")
    for val in target_counts.index:
        print(f"  {val}: {target_counts[val]} ({target_props[val]:.3%})")

    # build metadata so CTGAN understands the column types
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=gmsc)
    metadata.validate_data(data=gmsc)
    print("\nMetadata validated")

    # save metadata in case we need it later
    os.makedirs('data', exist_ok=True)
    metadata.save_to_json('data/gmsc_metadata.json')

    # train CTGAN on the full cleaned dataset
    print("\nTraining CTGAN on full dataset...")
    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        epochs=300,
        batch_size=500,
        verbose=True
    )

    synthesizer.fit(gmsc)
    print("CTGAN training complete")

    # generate synthetic data matching the real class distribution
    print("\nGenerating synthetic data...")

    real_default_rate = target_props.get(1, 0.0)
    n_total = gmsc.shape[0]
    n_defaults = int(n_total * real_default_rate)
    n_non_defaults = n_total - n_defaults

    print(f"  Class 0: {n_non_defaults:,}")
    print(f"  Class 1: {n_defaults:,}")

    conditions_default = pd.DataFrame({TARGET_COL: [1] * n_defaults})
    conditions_non_default = pd.DataFrame({TARGET_COL: [0] * n_non_defaults})

    print("Generating defaults...")
    synth_defaults = synthesizer.sample_remaining_columns(known_columns=conditions_default)

    print("Generating non-defaults...")
    synth_non_defaults = synthesizer.sample_remaining_columns(known_columns=conditions_non_default)

    # combine and shuffle
    gmsc_synth = pd.concat([synth_defaults, synth_non_defaults], ignore_index=True)
    gmsc_synth = gmsc_synth.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"\nSynthetic data shape: {gmsc_synth.shape}")

    # quick check that the class distribution looks right
    synth_dist = gmsc_synth[TARGET_COL].value_counts(normalize=True).sort_index()
    print("\nTarget distribution comparison:")
    print(f"  Real:      {target_props.to_dict()}")
    print(f"  Synthetic: {synth_dist.to_dict()}")

    # save synthetic data
    gmsc_synth.to_csv('data/gmsc_synthetic.csv', index=False)
    print("\nSaved to data/gmsc_synthetic.csv")