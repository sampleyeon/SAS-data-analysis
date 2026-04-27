import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold

# transaction 집계
# 기준일 설정 (거래 마지막 날 다음날)
reference_date = pd.Timestamp('2024-01-01')

transaction['trans_date'] = pd.to_datetime(transaction['trans_date'])

# ── Recency / Frequency / Monetary ──
agg_base = transaction.groupby('customer_id').agg(
    recency        = ('trans_date', lambda x: (reference_date - x.max()).days),
    frequency      = ('trans_id', 'count'),
    total_amount   = ('trans_amount', 'sum'),
    mean_amount    = ('trans_amount', 'mean'),
    std_amount     = ('trans_amount', 'std'),
    max_amount     = ('trans_amount', 'max'),
    min_amount     = ('trans_amount', 'min'),
).reset_index()

print(agg_base.shape)
agg_base.head(3)

# ── 최근 1개월 vs 이전 소비 비교 (소비 감소 추세 → churn 신호) ──
last_1m = transaction[transaction['trans_date'] >= pd.Timestamp('2023-12-01')]
prev_5m = transaction[transaction['trans_date'] <  pd.Timestamp('2023-12-01')]

last_1m_agg = last_1m.groupby('customer_id')['trans_amount'].sum().rename('amt_last_1m')
prev_5m_agg = prev_5m.groupby('customer_id')['trans_amount'].sum().rename('amt_prev_5m')

agg_trend = pd.concat([last_1m_agg, prev_5m_agg], axis=1).fillna(0)

# 소비 감소율
agg_trend['amt_trend_ratio'] = (
    agg_trend['amt_last_1m'] - agg_trend['amt_prev_5m'] / 5
) / (agg_trend['amt_prev_5m'] / 5 + 1)

agg_trend = agg_trend.reset_index()
print(agg_trend.shape)
agg_trend.head(3)

# ── 구매 간격 (gap) ──
agg_gap = transaction.groupby('customer_id')['trans_date'].apply(
    lambda x: x.sort_values().diff().dt.days.mean()
).rename('mean_purchase_gap').reset_index()

print(agg_gap.shape)
agg_gap.head(3)

# ── 채널 비율 (Online 비율) ──
agg_channel = transaction.groupby('customer_id').apply(
    lambda x: (x['biz_type'] == 'Online').mean(), include_groups=False
).rename('online_ratio').reset_index()

print(agg_channel.shape)

# ── 카테고리 다양성 ──
agg_category = transaction.groupby('customer_id').agg(
    category_nunique = ('item_category', 'nunique'),
    top_category     = ('item_category', lambda x: x.value_counts().index[0]),
).reset_index()

print(agg_category.shape)

# ── 할부 비율 ──
agg_install = transaction.groupby('customer_id')['is_installment'].mean()\
              .rename('installment_ratio').reset_index()

print(agg_install.shape)
