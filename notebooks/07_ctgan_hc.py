import pandas as pd
import numpy as np
import warnings
import json
import gc
from pathlib import Path
from datetime import datetime
from scipy.stats import ks_2samp, chi2_contingency
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

if __name__ == "__main__":

    # load the CTGAN-ready training data from feature selection
    print("Loading Home Credit CTGAN-ready data...")
    train_df = pd.read_csv('data/hc_train_ctgan_ready.csv')

    print(f"Training data shape: {train_df.shape}")
    print(f"Default rate: {train_df['TARGET'].mean():.4f}")

    # identify column types
    categorical_cols = [c for c in train_df.columns if train_df[c].dtype == 'object' and c != 'TARGET']
    numeric_cols = [c for c in train_df.columns if c not in categorical_cols and c != 'TARGET']

    # separate discrete and continuous numerical columns
    discrete_cols = []
    continuous_cols = []
    for c in numeric_cols:
        nunique = train_df[c].nunique(dropna=False)
        if c.startswith(("FLAG_", "CNT_")) or nunique <= 20:
            discrete_cols.append(c)
        else:
            continuous_cols.append(c)

    print(f"\nColumn types:")
    print(f"  Categorical: {len(categorical_cols)}")
    print(f"  Discrete numerical: {len(discrete_cols)}")
    print(f"  Continuous numerical: {len(continuous_cols)}")

    # build metadata for CTGAN
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

    # train CTGAN
    print("\nTraining CTGAN on Home Credit data (this will take a while)...")
    EPOCHS = 300

    ctgan = CTGANSynthesizer(
        metadata=metadata,
        epochs=EPOCHS,
        verbose=True,
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
    print(f"\nCTGAN training complete in {training_duration/60:.1f} minutes")

    # generate synthetic training data same size as real training set
    print("\nGenerating synthetic data...")
    synth_train_df = ctgan.sample(num_rows=len(train_df))

    # round discrete columns back to integers
    for col in discrete_cols:
        if col in synth_train_df.columns:
            synth_train_df[col] = synth_train_df[col].round().astype('Int64')

    # make sure TARGET is binary
    if 'TARGET' in synth_train_df.columns:
        synth_train_df['TARGET'] = synth_train_df['TARGET'].apply(
            lambda x: 1 if str(x) == '1' else 0
        ).astype(int)

    print(f"Synthetic data shape: {synth_train_df.shape}")
    print(f"Synthetic default rate: {synth_train_df['TARGET'].mean():.4f}")
    print(f"Real default rate:      {train_df['TARGET'].mean():.4f}")

    # run KS tests on continuous features to check distributional similarity
    print("\nRunning statistical validation...")
    validation_results = []

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

    for col in categorical_cols + ['TARGET']:
        if col in train_df.columns and col in synth_train_df.columns:
            real_counts = train_df[col].value_counts()
            synth_counts = synth_train_df[col].value_counts()
            all_cats = set(real_counts.index) | set(synth_counts.index)
            if len(all_cats) > 1:
                real_freq = [real_counts.get(cat, 0) for cat in all_cats]
                synth_freq = [synth_counts.get(cat, 0) for cat in all_cats]
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

    # save outputs
    import os
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)

    synth_train_df.to_csv('data/hc_synthetic.csv', index=False)
    validation_df.to_csv('outputs/metrics/hc_ctgan_validation.csv', index=False)

    # save a small report
    report = {
        'training_duration_minutes': round(training_duration / 60, 1),
        'epochs': EPOCHS,
        'real_train_rows': len(train_df),
        'synth_train_rows': len(synth_train_df),
        'real_default_rate': float(train_df['TARGET'].mean()),
        'synth_default_rate': float(synth_train_df['TARGET'].mean()),
        'features': train_df.shape[1]
    }

    with open('outputs/metrics/hc_ctgan_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved synthetic data to data/hc_synthetic.csv")
    print(f"Saved validation to outputs/metrics/hc_ctgan_validation.csv")
    print("Done.")