import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 경로 및 파라미터 설정
# ==========================================
BASE = "my_drive"

# [수정] Seed 값 변경
SEED = 997

CHURN_PARAMS = {
    'n_estimators': 10000,
    'learning_rate': 0.005,
    'num_leaves': 31,
    'min_child_samples': 30,
    'scale_pos_weight': 3.378,
    'subsample': 0.857,
    'colsample_bytree': 0.570,
    'reg_alpha': 1.785,
    'reg_lambda': 0.318,
    'random_state': SEED,
    'verbose': -1
}

LTV_PARAMS = {
    'n_estimators': 3000,
    'learning_rate': 0.003,
    'num_leaves': 63,
    'min_child_samples': 15,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': SEED,

# ==========================================
# 2. 피처 엔지니어링
# ==========================================
def get_features(cust_path, tran_path, fin_path, target_df=None):
    cust = pd.read_csv(cust_path)
    tran = pd.read_csv(tran_path)
    fin = pd.read_csv(fin_path)

    ref_date = pd.Timestamp('2024-01-01')
    tran['trans_date'] = pd.to_datetime(tran['trans_date'])

    agg = tran.groupby('customer_id').agg(
        recency=('trans_date', lambda x: (ref_date - x.max()).days),
        frequency=('trans_id', 'count'),
        amt_sum=('trans_amount', 'sum'),
        amt_mean=('trans_amount', 'mean'),
        amt_std=('trans_amount', 'std'),
        amt_max=('trans_amount', 'max'),
        trans_active_days=('trans_date', lambda x: x.dt.date.nunique())
    ).reset_index()

    agg['amt_cv'] = agg['amt_std'] / (agg['amt_mean'] + 1)
    agg['amount_per_trans'] = agg['amt_sum'] / (agg['frequency'] + 1)
    agg['amount_per_active_day'] = agg['amt_sum'] / (agg['trans_active_days'] + 1)
    agg['trans_per_active_day'] = agg['frequency'] / (agg['trans_active_days'] + 1)

    # 카테고리 다양성
    cat_nunique = tran.groupby('customer_id')['item_category'].nunique().rename('cat_nunique')

    # 최근 1개월 vs 이전
    t_l1m = tran[tran['trans_date'] >= pd.Timestamp('2023-12-01')]
    t_p5m = tran[tran['trans_date'] < pd.Timestamp('2023-12-01')]

    l1m_sum = t_l1m.groupby('customer_id')['trans_amount'].sum().rename('amt_l1m')
    p5m_sum = t_p5m.groupby('customer_id')['trans_amount'].sum().rename('amt_p5m')

    trend = pd.merge(l1m_sum, p5m_sum, on='customer_id', how='outer').fillna(0)
    trend['amt_trend_ratio'] = (
        trend['amt_l1m'] - trend['amt_p5m'] / 5
    ) / (trend['amt_p5m'] / 5 + 1)

    gap = tran.groupby('customer_id')['trans_date'].apply(
        lambda x: x.sort_values().diff().dt.days.mean()
    ).rename('mean_gap')

    onl = tran.groupby('customer_id').apply(
        lambda x: (x['biz_type'] == 'Online').mean()
    ).rename('online_ratio')

    install = tran.groupby('customer_id')['is_installment'].mean().rename('install_ratio')

    feats = agg.merge(trend, on='customer_id', how='left') \
        .merge(gap, on='customer_id', how='left') \
        .merge(onl, on='customer_id', how='left') \
        .merge(install, on='customer_id', how='left') \
        .merge(cat_nunique, on='customer_id', how='left')

    if target_df is not None:
        df = target_df.merge(cust, on='customer_id', how='left')
    else:
        df = cust.copy()

    df = df.merge(fin, on='customer_id', how='left') \
           .merge(feats, on='customer_id', how='left')

    df['join_date'] = pd.to_datetime(df['join_date'])
    df['join_days'] = (ref_date - df['join_date']).dt.days
    df['tx_density'] = df['frequency'] / (df['join_days'] + 1)
    df['loan_asset_ratio'] = df['total_loan_balance'] / (df['total_deposit_balance'] + 1)

    # 추가 변수
    df['net_cash_flow'] = df['total_deposit_balance'] - df['card_loan_amt']
    df['debt_pressure'] = df['total_loan_balance'] / (df['credit_score'] + 1)

    return df.drop(columns=['join_date']).fillna(0)

# ==========================================
# 3. 데이터 로드
# ==========================================
train_target = pd.read_csv(f'{BASE}/train/train_targets.csv')

train_df = get_features(
    f'{BASE}/train/train_customer_info.csv',
    f'{BASE}/train/train_transaction_history.csv',
    f'{BASE}/train/train_finance_profile.csv',
    train_target
)

test_df = get_features(
    f'{BASE}/test/test_customer_info.csv',
    f'{BASE}/test/test_transaction_history.csv',
    f'{BASE}/test/test_finance_profile.csv'
)

# ==========================================
# 4. 인코딩
# ==========================================
cat_cols = ['gender', 'region_code', 'prefer_category', 'income_group']

for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = test_df[col].astype(str).map(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
  
# ==========================================
# 5. 데이터 분리
# ==========================================
X = train_df.drop(columns=['customer_id', 'target_churn', 'target_ltv'])
y_churn = train_df['target_churn']
y_ltv_sqrt = np.sqrt(train_df['target_ltv'])

X_test = test_df[X.columns]

# stratify 추가
train_df['ltv_bin'] = pd.qcut(train_df['target_ltv'], 5, labels=False, duplicates='drop')
train_df['stratify_col'] = train_df['target_churn'].astype(str) + "_" + train_df['ltv_bin'].astype(str)

# ==========================================
# 6. K-Fold
# ==========================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = []
rmse_scores = []

churn_preds = np.zeros(len(X_test))
ltv_preds = np.zeros(len(X_test))

for fold, (t_idx, v_idx) in enumerate(skf.split(X, train_df['stratify_col'])):
    X_tr, X_vl = X.iloc[t_idx], X.iloc[v_idx]
    yc_tr, yc_vl = y_churn.iloc[t_idx], y_churn.iloc[v_idx]
    yl_tr, yl_vl = y_ltv_sqrt.iloc[t_idx], y_ltv_sqrt.iloc[v_idx]

    # =========================
    # 1. LightGBM (기존)
    # =========================
    m_c = lgb.LGBMClassifier(**CHURN_PARAMS)
    m_c.fit(X_tr, yc_tr, eval_set=[(X_vl, yc_vl)],
            callbacks=[lgb.early_stopping(300), lgb.log_evaluation(-1)])

    lgb_churn_val = m_c.predict_proba(X_vl)[:, 1]
    lgb_churn_test = m_c.predict_proba(X_test)[:, 1]

    m_l = lgb.LGBMRegressor(**LTV_PARAMS)
    m_l.fit(X_tr, yl_tr, eval_set=[(X_vl, yl_vl)],
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(-1)])

    lgb_ltv_val = np.maximum(m_l.predict(X_vl), 0) ** 2
    lgb_ltv_test = np.maximum(m_l.predict(X_test), 0) ** 2

    # =========================
    # 2. CatBoost 추가
    # =========================
    m_c_cb = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.01,
        depth=6,
        eval_metric='AUC',
        random_seed=SEED,
        verbose=0
    )

    m_c_cb.fit(X_tr, yc_tr, eval_set=(X_vl, yc_vl))

    cb_churn_val = m_c_cb.predict_proba(X_vl)[:, 1]
    cb_churn_test = m_c_cb.predict_proba(X_test)[:, 1]

    m_l_cb = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.01,
        depth=8,
        random_seed=SEED,
        verbose=0
    )

    m_l_cb.fit(X_tr, yl_tr, eval_set=(X_vl, yl_vl))

    cb_ltv_val = np.maximum(m_l_cb.predict(X_vl), 0) ** 2
    cb_ltv_test = np.maximum(m_l_cb.predict(X_test), 0) ** 2

    # =========================
    # 3. 앙상블 (평균)
    # =========================
    vl_churn_pred = (lgb_churn_val + cb_churn_val) / 2
    vl_ltv_pred = (lgb_ltv_val + cb_ltv_val) / 2

    fold_auc = roc_auc_score(yc_vl, vl_churn_pred)
    fold_rmse = np.sqrt(mean_squared_error(train_df['target_ltv'].iloc[v_idx], vl_ltv_pred))

    auc_scores.append(fold_auc)
    rmse_scores.append(fold_rmse)

    churn_preds += (lgb_churn_test + cb_churn_test) / 2 / 5
    ltv_preds += (lgb_ltv_test + cb_ltv_test) / 2 / 5

    print(f"Fold {fold+1}: AUC = {fold_auc:.4f}, RMSE = {fold_rmse:,.0f}")

# ==========================================
# 7. 결과
# ==========================================
mean_auc = np.mean(auc_scores)
mean_rmse = np.mean(rmse_scores)

final_score = 0.5 * mean_auc + 0.5 * (1 / (1 + np.log(mean_rmse)))

print(f"\n{'=' * 45}")
print(f"CV AUC  평균: {mean_auc:.4f}")
print(f"CV RMSE 평균: {mean_rmse:,.0f}")
print(f"대회 산식 최종 Score: {final_score:.5f}")
print(f"{'=' * 45}")

    'verbose': -1
}
