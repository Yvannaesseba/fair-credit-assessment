"""
Home Credit Feature Selection with XGBoost
Selects top features for CTGAN while preparing CTGAN-ready datasets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
import xgboost as xgb
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
DATA_PATH = "/path/to/home_credit_merged.csv"  # UPDATE THIS
OUTPUT_DIR = "/path/to/output"  # UPDATE THIS

# Load data
df = pd.read_csv(DATA_PATH)

# Remove ID column
if 'SK_ID_CURR' in df.columns:
    df = df.drop('SK_ID_CURR', axis=1)

# Separate target
y = df['TARGET'].copy()
X = df.drop('TARGET', axis=1)

# Identify feature types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
ext_source_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

# Train/Val/Test split (60/20/20) BEFORE preprocessing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
)

# Preprocessing for XGBoost (temporary - only for feature selection)
X_train_xgb = X_train.copy()
X_val_xgb = X_val.copy()
X_test_xgb = X_test.copy()

# Handle missing values for XGBoost
for col in numerical_cols:
    if X_train_xgb[col].isnull().any():
        fill_value = X_train_xgb[col].median()
        X_train_xgb[col].fillna(fill_value, inplace=True)
        X_val_xgb[col].fillna(fill_value, inplace=True)
        X_test_xgb[col].fillna(fill_value, inplace=True)

for col in categorical_cols:
    if X_train_xgb[col].isnull().any():
        fill_value = X_train_xgb[col].mode()[0] if len(X_train_xgb[col].mode()) > 0 else 'Unknown'
        X_train_xgb[col].fillna(fill_value, inplace=True)
        X_val_xgb[col].fillna(fill_value, inplace=True)
        X_test_xgb[col].fillna(fill_value, inplace=True)

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train_xgb[col] = le.fit_transform(X_train_xgb[col].astype(str))
    
    X_val_xgb[col] = X_val_xgb[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
    X_test_xgb[col] = X_test_xgb[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
    
    if 'Unknown' not in le.classes_:
        le.classes_ = np.append(le.classes_, 'Unknown')
    
    X_val_xgb[col] = le.transform(X_val_xgb[col].astype(str))
    X_test_xgb[col] = le.transform(X_test_xgb[col].astype(str))
    label_encoders[col] = le

# Outlier treatment (skip EXT_SOURCE features)
for col in numerical_cols:
    if col in ext_source_features:
        continue
    
    Q1 = X_train_xgb[col].quantile(0.25)
    Q3 = X_train_xgb[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = ((X_train_xgb[col] < lower_bound) | (X_train_xgb[col] > upper_bound)).sum()
    if outliers > 0:
        X_train_xgb[col] = X_train_xgb[col].clip(lower_bound, upper_bound)
        X_val_xgb[col] = X_val_xgb[col].clip(lower_bound, upper_bound)
        X_test_xgb[col] = X_test_xgb[col].clip(lower_bound, upper_bound)

# Train XGBoost
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale_pos_weight,
    'random_state': RANDOM_STATE,
    'tree_method': 'hist'
}

xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train_xgb, y_train, verbose=False)

# Evaluate model
def evaluate_model(model, X, y):
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    return {
        'AUC': roc_auc_score(y, y_pred_proba),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1': f1_score(y, y_pred, zero_division=0),
        'Avg_Precision': average_precision_score(y, y_pred_proba)
    }

train_results = evaluate_model(xgb_model, X_train_xgb, y_train)
val_results = evaluate_model(xgb_model, X_val_xgb, y_val)
test_results = evaluate_model(xgb_model, X_test_xgb, y_test)

# Extract feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train_xgb.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

feature_importance['Cumulative_Importance'] = feature_importance['Importance'].cumsum()
feature_importance['Cumulative_Pct'] = (
    feature_importance['Cumulative_Importance'] / feature_importance['Importance'].sum()
) * 100

# Select top features (85% cumulative importance)
recommended_n = (feature_importance['Cumulative_Pct'] <= 85).sum()
top_features = feature_importance.head(recommended_n)['Feature'].tolist()

top_categorical = [f for f in top_features if f in categorical_cols]
top_numerical = [f for f in top_features if f in numerical_cols]

# Prepare CTGAN-ready datasets
X_train_ctgan = X_train[top_features].copy()
X_val_ctgan = X_val[top_features].copy()
X_test_ctgan = X_test[top_features].copy()

X_train_ctgan['TARGET'] = y_train.values
X_val_ctgan['TARGET'] = y_val.values
X_test_ctgan['TARGET'] = y_test.values

# Handle missing values for CTGAN: -999 for numerical, 'Missing' for categorical
for col in top_numerical:
    if X_train_ctgan[col].isnull().any():
        X_train_ctgan[col].fillna(-999, inplace=True)
        X_val_ctgan[col].fillna(-999, inplace=True)
        X_test_ctgan[col].fillna(-999, inplace=True)

for col in top_categorical:
    X_train_ctgan[col] = X_train_ctgan[col].astype(str)
    X_val_ctgan[col] = X_val_ctgan[col].astype(str)
    X_test_ctgan[col] = X_test_ctgan[col].astype(str)
    
    X_train_ctgan[col] = X_train_ctgan[col].replace('nan', 'Missing')
    X_val_ctgan[col] = X_val_ctgan[col].replace('nan', 'Missing')
    X_test_ctgan[col] = X_test_ctgan[col].replace('nan', 'Missing')

# Save results
os.makedirs(OUTPUT_DIR, exist_ok=True)

feature_importance.to_csv(f"{OUTPUT_DIR}/feature_importance_rankings.csv", index=False)

with open(f"{OUTPUT_DIR}/top_{recommended_n}_features.txt", 'w') as f:
    f.write(f"Top {recommended_n} Features (85% Cumulative Importance)\n")
    f.write("=" * 60 + "\n\n")
    f.write("CATEGORICAL FEATURES:\n")
    for feat in top_categorical:
        f.write(f"  - {feat}\n")
    f.write(f"\nNUMERICAL FEATURES:\n")
    for feat in top_numerical:
        f.write(f"  - {feat}\n")

X_train_ctgan.to_csv(f"{OUTPUT_DIR}/train_ctgan_ready.csv", index=False)
X_val_ctgan.to_csv(f"{OUTPUT_DIR}/val_ctgan_ready.csv", index=False)
X_test_ctgan.to_csv(f"{OUTPUT_DIR}/test_ctgan_ready.csv", index=False)

performance_df = pd.DataFrame([
    {'Set': 'Training', **train_results},
    {'Set': 'Validation', **val_results},
    {'Set': 'Test', **test_results}
])
performance_df.to_csv(f"{OUTPUT_DIR}/model_performance.csv", index=False)

xgb_model.save_model(f"{OUTPUT_DIR}/xgboost_feature_selection.json")

print(f"Feature selection complete: {recommended_n} features selected")
print(f"Test AUC: {test_results['AUC']:.4f}")
print(f"Results saved to {OUTPUT_DIR}")