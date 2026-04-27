# 모델 학습 준비
drop_cols = ['customer_id', 'target_churn', 'target_ltv', 'target_ltv_log']

X = df.drop(columns=drop_cols)
y_churn = df['target_churn']
y_ltv = df['target_ltv_log']

print(f"X shape      : {X.shape}")
print(f"y_churn 분포 : {y_churn.value_counts().to_dict()}")
print(f"y_ltv   범위 : {y_ltv.min():.2f} ~ {y_ltv.max():.2f}")

# 데이터셋 분할
from sklearn.model_selection import train_test_split

X_train, X_val, y_churn_train, y_churn_val, y_ltv_train, y_ltv_val = train_test_split(
    X, y_churn, y_ltv,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y_churn
)

print(f"Train : {X_train.shape}")
print(f"Val   : {X_val.shape}")

# 모델 학습
# ── Churn 모델 ──
churn_model = lgb.LGBMClassifier(
    n_estimators     = 1000,
    learning_rate    = 0.05,
    num_leaves       = 31,
    scale_pos_weight = 9,
    random_state     = 42,
    verbose          = -1
)

churn_model.fit(
    X_train, y_churn_train,
    eval_set  = [(X_val, y_churn_val)],
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# ── LTV 모델 ──
ltv_model = lgb.LGBMRegressor(
    n_estimators  = 1000,
    learning_rate = 0.05,
    num_leaves    = 31,
    random_state  = 42,
    verbose       = -1
)

ltv_model.fit(
    X_train, y_ltv_train,
    eval_set  = [(X_val, y_ltv_val)],
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# 성능 확인 (첫시도)
# ── Churn AUC ──
churn_pred_proba = churn_model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_churn_val, churn_pred_proba)
print(f"Churn AUC : {auc:.4f}")

# ── LTV RMSE (log 역변환 후) ──
ltv_pred_log = ltv_model.predict(X_val)
ltv_pred     = np.expm1(ltv_pred_log)
ltv_actual   = np.expm1(y_ltv_val)

rmse = np.sqrt(mean_squared_error(ltv_actual, ltv_pred))
print(f"LTV RMSE  : {rmse:,.0f}")
