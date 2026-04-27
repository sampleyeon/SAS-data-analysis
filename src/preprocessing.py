import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold

# 전체 병합
# transaction 집계 병합
tx_features = agg_base\
    .merge(agg_trend,    on='customer_id', how='left')\
    .merge(agg_gap,      on='customer_id', how='left')\
    .merge(agg_channel,  on='customer_id', how='left')\
    .merge(agg_category, on='customer_id', how='left')\
    .merge(agg_install,  on='customer_id', how='left')

# 전체 병합
df = target\
    .merge(customer,    on='customer_id', how='left')\
    .merge(finance,     on='customer_id', how='left')\
    .merge(tx_features, on='customer_id', how='left')

print(f"최종 shape: {df.shape}")
df.head(3)

# ── join_date → 가입 경과일 ──
df['join_date'] = pd.to_datetime(df['join_date'])
df['join_days'] = (reference_date - df['join_date']).dt.days
df = df.drop(columns=['join_date'])

# ── 범주형 인코딩 ──
cat_cols = ['gender', 'region_code', 'prefer_category', 'income_group', 'top_category']

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print("인코딩 완료")
print(df[cat_cols].head(3))

# ── LTV log 변환 ──
df['target_ltv_log'] = np.log1p(df['target_ltv'])

print(f"LTV 왜도 원본     : {df['target_ltv'].skew():.2f}")
print(f"LTV 왜도 log 변환 : {df['target_ltv_log'].skew():.2f}")

# ── 최종 feature 확인 ──
print(f"최종 shape : {df.shape}")
print(f"컬럼 목록  :\n{df.columns.tolist()}")
