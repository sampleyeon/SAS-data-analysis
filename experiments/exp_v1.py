import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold


def prepare_features(df):

    drop_cols = [
        'customer_id',
        'target_churn',
        'target_ltv',
        'target_ltv_log'
    ]

    X = df.drop(columns=drop_cols)
    y_churn = df['target_churn']
    y_ltv   = df['target_ltv_log']

    return X, y_churn, y_ltv


def run_cv_v1(X, y_churn, y_ltv):

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores  = []
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_churn)):

        X_tr, X_vl   = X.iloc[train_idx], X.iloc[val_idx]
        yc_tr, yc_vl = y_churn.iloc[train_idx], y_churn.iloc[val_idx]
        yl_tr, yl_vl = y_ltv.iloc[train_idx], y_ltv.iloc[val_idx]

        # Baseline 모델 
        cm = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            scale_pos_weight=9,
            random_state=42,
            verbose=-1
        )

        cm.fit(
            X_tr, yc_tr,
            eval_set=[(X_vl, yc_vl)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        auc = roc_auc_score(yc_vl, cm.predict_proba(X_vl)[:, 1])
        auc_scores.append(auc)

        # 🔵 LTV baseline
        lm = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )

        lm.fit(
            X_tr, yl_tr,
            eval_set=[(X_vl, yl_vl)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        ltv_pred   = np.expm1(lm.predict(X_vl))
        ltv_actual = np.expm1(yl_vl)

        rmse = np.sqrt(mean_squared_error(ltv_actual, ltv_pred))
        rmse_scores.append(rmse)

        print(f"[V1] Fold {fold+1} | AUC: {auc:.4f} | RMSE: {rmse:,.0f}")

    print("\n===== V1 결과 =====")
    print(f"AUC  : {np.mean(auc_scores):.4f}")
    print(f"RMSE : {np.mean(rmse_scores):,.0f}")

    return auc_scores, rmse_scores


def run_exp_v1(df):

    X, y_churn, y_ltv = prepare_features(df)

    print(f"X shape: {X.shape}")
    print(f"Churn 분포: {y_churn.value_counts().to_dict()}")

    run_cv_v1(X, y_churn, y_ltv)
