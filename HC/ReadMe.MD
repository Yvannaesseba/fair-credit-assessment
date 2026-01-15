# Home Credit Synthetic Data Evaluation Pipeline

Empirical evaluation of CTGAN-generated synthetic data for credit scoring using the Train-Synthetic-Test-Real (TSTR) framework.

## Pipeline Overview

This pipeline implements the complete experimental workflow described in the dissertation:

1. **Data Preparation**: Clean and merge Application + Bureau tables
2. **Feature Selection**: Identify top features using XGBoost (85% importance)
3. **Synthetic Data Generation**: Train CTGAN and generate synthetic training data
4. **Model Training**: Train models on real vs synthetic data (TSTR)
5. **Statistical Validation**: Cross-validation and significance testing

## Dataset

**Home Credit Default Risk**
- Source: Kaggle competition dataset
- Samples: 307,511 loan applications
- Features: 107 after feature selection (from 122 original)
- Default rate: ~8%
- Protected attribute: Age (derived from DAYS_BIRTH)

## File Structure

```
00_hc_data_preparation.py      # Data cleaning and merging
01_hc_feature_selection.py     # XGBoost feature selection
02_hc_synthetic_generation.py  # CTGAN training and generation
03_hc_model_training.py         # TSTR model training
04_hc_cv_tstr.py               # Cross-validation experiments
05_hc_fairness_analysis.py     # Demographic parity, equal opportunity (upload later)
06_hc_explainability_shap.py   # SHAP analysis (upload later)
```

 