import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
import os

warnings.filterwarnings('ignore')

RANDOM_STATE = 42

if __name__ == "__main__":

    # load the merged home credit dataset
    print("Loading Home Credit merged data...")
    df = pd.read_csv('data/home_credit_merged.csv')

    # drop the ID column, it's not a feature
    if 'SK_ID_CURR' in df.columns:
        df = df.drop('SK_ID_CURR', axis=1)

    y = df['TARGET'].copy()
    X = df.drop('TARGET', axis=1)

    print(f"Data shape: {X.shape}")
    print(f"Default rate: {y.mean():.4f}")

    # identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # EXT_SOURCE features are external credit scores, keep them as-is
    ext_source_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

    # 60/20/20 split before any preprocessing
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # preprocess for XGBoost feature selection only
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    X_test_xgb = X_test.copy()

    # fill missing numerical values with train medians
    for col in numerical_cols:
        fill_value = X_train_xgb[col].median()
        X_train_xgb[col] = X_train_xgb[col].fillna(fill_value)
        X_val_xgb[col] = X_val_xgb[col].fillna(fill_value)
        X_test_xgb[col] = X_test_xgb[col].fillna(fill_value)

    # fill missing categorical values with train mode
    for col in categorical_cols:
        fill_value = X_train_xgb[col].mode()[0] if len(X_train_xgb[col].mode()) > 0 else 'Unknown'
        X_train_xgb[col] = X_train_xgb[col].fillna(fill_value)
        X_val_xgb[col] = X_val_xgb[col].fillna(fill_value)
        X_test_xgb[col] = X_test_xgb[col].fillna(fill_value)

    # encode categorical columns
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

    # cap outliers using IQR, skip EXT_SOURCE features
    for col in numerical_cols:
        if col in ext_source_features:
            continue
        Q1 = X_train_xgb[col].quantile(0.25)
        Q3 = X_train_xgb[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        X_train_xgb[col] = X_train_xgb[col].clip(lower, upper)
        X_val_xgb[col] = X_val_xgb[col].clip(lower, upper)
        X_test_xgb[col] = X_test_xgb[col].clip(lower, upper)

    # train XGBoost to rank feature importance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        tree_method='hist'
    )

    print("\nTraining XGBoost for feature selection...")
    xgb_model.fit(X_train_xgb, y_train, verbose=False)

    val_auc = roc_auc_score(y_val, xgb_model.predict_proba(X_val_xgb)[:, 1])
    print(f"Validation AUC: {val_auc:.4f}")

    # rank features by importance and select top ones covering 85% cumulative importance
    feature_importance = pd.DataFrame({
        'Feature': X_train_xgb.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    feature_importance['Cumulative_Pct'] = (
        feature_importance['Importance'].cumsum() / feature_importance['Importance'].sum()
    ) * 100

    recommended_n = (feature_importance['Cumulative_Pct'] <= 85).sum()
    top_features = feature_importance.head(recommended_n)['Feature'].tolist()

    print(f"\nSelected {recommended_n} features covering 85% cumulative importance")

    # prepare CTGAN-ready datasets using raw (unencoded) values
    top_categorical = [f for f in top_features if f in categorical_cols]
    top_numerical = [f for f in top_features if f in numerical_cols]

    X_train_ctgan = X_train[top_features].copy()
    X_val_ctgan = X_val[top_features].copy()
    X_test_ctgan = X_test[top_features].copy()

    X_train_ctgan['TARGET'] = y_train.values
    X_val_ctgan['TARGET'] = y_val.values
    X_test_ctgan['TARGET'] = y_test.values

    # use -999 for missing numericals and 'Missing' for categoricals
    for col in top_numerical:
        X_train_ctgan[col] = X_train_ctgan[col].fillna(-999)
        X_val_ctgan[col] = X_val_ctgan[col].fillna(-999)
        X_test_ctgan[col] = X_test_ctgan[col].fillna(-999)

    for col in top_categorical:
        for df_ in [X_train_ctgan, X_val_ctgan, X_test_ctgan]:
            df_[col] = df_[col].astype(str).replace('nan', 'Missing')

    # save outputs
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)

    X_train_ctgan.to_csv('data/hc_train_ctgan_ready.csv', index=False)
    X_val_ctgan.to_csv('data/hc_val_ctgan_ready.csv', index=False)
    X_test_ctgan.to_csv('data/hc_test_ctgan_ready.csv', index=False)

    feature_importance.to_csv('outputs/metrics/hc_feature_importance.csv', index=False)

    print(f"\nSaved CTGAN-ready datasets to data/")
    print(f"Saved feature importance to outputs/metrics/hc_feature_importance.csv")
    print("Done.")