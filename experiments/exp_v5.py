# 1. 전처리 및 고도화된 Feature Engineering
# 1. 가속도 및 트렌드 피처 생성 (Train/Test 공통 적용)
def engineering_features(df_tx, reference_date):
    # RFM 기본 집계
    agg = df_tx.groupby('customer_id').agg(
        recency=('trans_date', lambda x: (reference_date - x.max()).days),
        frequency=('trans_id', 'count'),
        total_amount=('trans_amount', 'sum'),
        mean_amount=('trans_amount', 'mean'),
        mean_purchase_gap=('trans_date', lambda x: x.sort_values().diff().dt.days.mean())
    ).reset_index()

    # 최근 1개월 vs 이전 5개월 소비 트렌드
    last_1m = df_tx[df_tx['trans_date'] >= (reference_date - pd.Timedelta(days=30))]
    prev_5m = df_tx[df_tx['trans_date'] < (reference_date - pd.Timedelta(days=30))]
    
    amt_1m = last_1m.groupby('customer_id')['trans_amount'].sum().rename('amt_last_1m')
    amt_5m = prev_5m.groupby('customer_id')['trans_amount'].sum().rename('amt_prev_5m')
    
    trend = pd.concat([amt_1m, amt_5m], axis=1).fillna(0)
    trend['amt_trend_ratio'] = (trend['amt_last_1m'] - trend['amt_prev_5m']/5) / (trend['amt_prev_5m']/5 + 1)
    
    # 가속도 피처 (Velocity)
    last_3m = df_tx[df_tx['trans_date'] >= (reference_date - pd.Timedelta(days=90))]
    amt_3m = last_3m.groupby('customer_id').agg(amt_3m=('trans_amount', 'sum'), freq_3m=('trans_id', 'count'))
    
    # 최종 병합
    features = agg.merge(trend.reset_index(), on='customer_id', how='left')\
                  .merge(amt_3m.reset_index(), on='customer_id', how='left').fillna(0)
                  
    features['velocity_amt'] = features['amt_last_1m'] / (features['amt_3m'] / 3 + 1)
    features['gap_delay_ratio'] = features['recency'] / (features['mean_purchase_gap'] + 1)
    
    return features

# 2. 최적화된 하이퍼파라미터 및 XGBoost 에러 수정
# LTV 타겟 변환 (왜도 개선)
df['target_ltv_sqrt'] = np.sqrt(df['target_ltv'])

# Churn 파라미터 (LGBM & XGB 앙상블용)
CHURN_PARAMS = {
    'n_estimators': 5000, 'learning_rate': 0.001, 'num_leaves': 63,
    'scale_pos_weight': 4, 'random_state': 100, 'verbose': -1
}

CHURN_XGB_PARAMS = {
    'n_estimators': 3000, 'learning_rate': 0.005, 'max_depth': 6,
    'scale_pos_weight': 4, 'tree_method': 'hist', 'random_state': 100,
    'early_stopping_rounds': 150 # 에러 방지 핵심
}

# 3. Stratified K-Fold 기반 앙상블 예측

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
churn_preds = np.zeros(len(test_df))
ltv_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_final, y_churn)):
    X_tr, X_vl = X_final.iloc[train_idx], X_final.iloc[val_idx]
    yc_tr, yc_vl = y_churn.iloc[train_idx], y_churn.iloc[val_idx]
    yl_tr, yl_vl = y_ltv.iloc[train_idx], y_ltv.iloc[val_idx]

    # [1] Churn 앙상블 (LGBM + XGB)
    m_lgb = lgb.LGBMClassifier(**CHURN_PARAMS).fit(X_tr, yc_tr, eval_set=[(X_vl, yc_vl)], callbacks=[lgb.early_stopping(150)])
    m_xgb = xgb.XGBClassifier(**CHURN_XGB_PARAMS).fit(X_tr, yc_tr, eval_set=[(X_vl, yc_vl)], verbose=False)
    
    fold_prob = (m_lgb.predict_proba(X_test_final)[:, 1] * 0.5) + (m_xgb.predict_proba(X_test_final)[:, 1] * 0.5)
    churn_preds += fold_prob / 5

    # [2] LTV 학습 및 예측 (sqrt 역변환 적용)
    m_ltv = lgb.LGBMRegressor(**LTV_PARAMS).fit(X_tr, yl_tr, eval_set=[(X_vl, yl_vl)], callbacks=[lgb.early_stopping(150)])
    ltv_preds += (np.maximum(m_ltv.predict(X_test_final), 0) ** 2) / 5


