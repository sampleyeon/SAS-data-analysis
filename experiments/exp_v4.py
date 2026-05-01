from sklearn.metrics import roc_auc_score, mean_squared_error

auc_scores_v4  = []
rmse_scores_v4 = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_final, y_churn)):

    X_tr, X_vl   = X_final.iloc[train_idx], X_final.iloc[val_idx]
    yc_tr, yc_vl = y_churn.iloc[train_idx],  y_churn.iloc[val_idx]
    yl_tr, yl_vl = y_ltv.iloc[train_idx],    y_ltv.iloc[val_idx]

    # ── Churn v4 ──
    # num_leaves 확대 → 더 복잡한 패턴 학습
    # scale_pos_weight 낮춤 → 정밀도/재현율 균형
    cm = lgb.LGBMClassifier(
        n_estimators     = 3000,
        learning_rate    = 0.005,   # 더 천천히 학습
        num_leaves       = 63,      # 31 → 63 확대
        min_child_samples= 15,      # 20 → 15 완화
        scale_pos_weight = 4,       # 5 → 4 완화
        subsample        = 0.8,
        colsample_bytree = 0.7,     # 0.8 → 0.7 축소
        reg_alpha        = 0.1,     # L1 규제 추가
        reg_lambda       = 1.0,     # L2 규제 추가
        random_state     = 42,
        verbose          = -1
    )
    cm.fit(X_tr, yc_tr,
           eval_set=[(X_vl, yc_vl)],
           callbacks=[lgb.early_stopping(150, verbose=False),
                      lgb.log_evaluation(-1)])

    auc = roc_auc_score(yc_vl, cm.predict_proba(X_vl)[:, 1])
    auc_scores_v4.append(auc)

    # ── LTV v4 ──
    # num_leaves 확대 + 규제 추가
    lm = lgb.LGBMRegressor(
        n_estimators     = 3000,
        learning_rate    = 0.005,
        num_leaves       = 63,
        min_child_samples= 15,
        subsample        = 0.8,
        colsample_bytree = 0.7,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        random_state     = 42,
        verbose          = -1
    )
    lm.fit(X_tr, yl_tr,
           eval_set=[(X_vl, yl_vl)],
           callbacks=[lgb.early_stopping(150, verbose=False),
                      lgb.log_evaluation(-1)])

    lltv_pred = np.maximum(lm.predict(X_vl), 0) ** 2
    ltv_actual = df['target_ltv'].iloc[val_idx]
    rmse       = np.sqrt(mean_squared_error(ltv_actual, ltv_pred))
    rmse_scores_v4.append(rmse)

    print(f"Fold {fold+1}  AUC: {auc:.4f}  RMSE: {rmse:,.0f}")

print(f"\n{'='*45}")
print(f"CV AUC  평균: {np.mean(auc_scores_v4):.4f}  std: {np.std(auc_scores_v4):.4f}  (기존: 0.7887)")
print(f"CV RMSE 평균: {np.mean(rmse_scores_v4):,.0f}  std: {np.std(rmse_scores_v4):,.0f}  (기존: 1,376,715)")
