# ─────────────────────────────
# 1. 모델 생성 함수
# ─────────────────────────────
def get_models():

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

    ltv_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    return churn_model, ltv_model

# ─────────────────────────────
# 2. Hold-out 평가
# ─────────────────────────────
def train_and_evaluate(X, y_churn, y_ltv):

    from sklearn.model_selection import train_test_split

    X_train, X_val, yc_tr, yc_val, yl_tr, yl_val = train_test_split(
        X, y_churn, y_ltv,
        test_size=0.2,
        random_state=42,
        stratify=y_churn
    )

    churn_model, ltv_model = get_models()

    # Churn
    churn_model.fit(
        X_train, yc_tr,
        eval_set=[(X_val, yc_val)],
        callbacks=[lgb.early_stopping(100)]
    )

    # LTV
    ltv_model.fit(
        X_train, yl_tr,
        eval_set=[(X_val, yl_val)],
        callbacks=[lgb.early_stopping(100)]
    )

    return churn_model, ltv_model

# ─────────────────────────────
# 3. Cross Validation
# ─────────────────────────────
def run_cv(X, y_churn, y_ltv, df, n_splits=5, random_state=42):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    auc_scores  = []
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_churn)):

        X_tr, X_vl   = X.iloc[train_idx], X.iloc[val_idx]
        yc_tr, yc_vl = y_churn.iloc[train_idx], y_churn.iloc[val_idx]
        yl_tr, yl_vl = y_ltv.iloc[train_idx], y_ltv.iloc[val_idx]

        # 모델 가져오기
        cm, lm = get_models()

        # ── Churn ──
        cm.fit(
            X_tr, yc_tr,
            eval_set=[(X_vl, yc_vl)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(-1)
            ]
        )

        auc = roc_auc_score(yc_vl, cm.predict_proba(X_vl)[:, 1])
        auc_scores.append(auc)

        # ── LTV ──
        lm.fit(
            X_tr, yl_tr,
            eval_set=[(X_vl, yl_vl)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(-1)
            ]
        )

        ltv_pred   = lm.predict(X_vl) ** 2
        ltv_actual = df['target_ltv'].iloc[val_idx]

        rmse = np.sqrt(mean_squared_error(ltv_actual, ltv_pred))
        rmse_scores.append(rmse)

        print(f"Fold {fold+1}  AUC: {auc:.4f}  RMSE: {rmse:,.0f}")

    print(f"\n{'='*45}")
    print(f"CV AUC  평균: {np.mean(auc_scores):.4f}  std: {np.std(auc_scores):.4f}")
    print(f"CV RMSE 평균: {np.mean(rmse_scores):,.0f}  std: {np.std(rmse_scores):,.0f}")

    return auc_scores, rmse_scores
