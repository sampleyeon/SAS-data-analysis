import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split


# =========================
# 1. Feature 준비
# =========================
def prepare_features(df):

    drop_cols = [
        'customer_id',
        'target_churn',
        'target_ltv',
        'target_ltv_log'
    ]

    X = df.drop(columns=drop_cols)
    y_churn = df['target_churn']
    y_ltv   = df['target_ltv_log']   # log 기준 유지

    return X, y_churn, y_ltv


# =========================
# 2. Cross Validation
# =========================
def run_cv(X, y_churn, y_ltv, df, n_splits=5):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    auc_scores  = []
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_churn)):

        X_tr, X_vl   = X.iloc[train_idx], X.iloc[val_idx]
        yc_tr, yc_vl = y_churn.iloc[train_idx], y_churn.iloc[val_idx]
        yl_tr, yl_vl = y_ltv.iloc[train_idx], y_ltv.iloc[val_idx]

        # ── Churn ──
        cm = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.01,
            num_leaves=31,
            min_child_samples=20,
            scale_pos_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

        cm.fit(
            X_tr, yc_tr,
            eval_set=[(X_vl, yc_vl)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        auc = roc_auc_score(yc_vl, cm.predict_proba(X_vl)[:, 1])
        auc_scores.append(auc)

        # ── LTV ──
        lm = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.01,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

        lm.fit(
            X_tr, yl_tr,
            eval_set=[(X_vl, yl_vl)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        ltv_pred   = np.expm1(lm.predict(X_vl))     # log → 원복
        ltv_actual = np.expm1(yl_vl)

        rmse = np.sqrt(mean_squared_error(ltv_actual, ltv_pred))
        rmse_scores.append(rmse)

        print(f"Fold {fold+1} | AUC: {auc:.4f} | RMSE: {rmse:,.0f}")

    print("\n===== CV 결과 =====")
    print(f"AUC  : {np.mean(auc_scores):.4f}")
    print(f"RMSE : {np.mean(rmse_scores):,.0f}")

    return auc_scores, rmse_scores


# =========================
# 3. 최종 모델 학습
# =========================
def train_final_model(X, y_churn, y_ltv):

    # 전체 데이터로 학습 (validation 없음)
    churn_model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=31,
        min_child_samples=20,
        scale_pos_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    churn_model.fit(X, y_churn)

    ltv_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    ltv_model.fit(X, y_ltv)

    return churn_model, ltv_model


# =========================
# 4. 전체 실행 함수
# =========================
def train(df):

    # 1. feature 준비
    X, y_churn, y_ltv = prepare_features(df)

    print(f"X shape: {X.shape}")

    # 2. CV로 성능 확인 (선택)
    run_cv(X, y_churn, y_ltv, df)

    # 3. 최종 모델 학습
    churn_model, ltv_model = train_final_model(X, y_churn, y_ltv)

    return churn_model, ltv_model
