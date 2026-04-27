import pandas as pd
import numpy as np


# =========================
# 1. test 데이터 로드
# =========================
def load_test_data():

    test_customer    = pd.read_csv('/test_customer_info.csv')
    test_transaction = pd.read_csv('/test_transaction_history.csv')
    test_finance     = pd.read_csv('/test_finance_profile.csv')

    return test_customer, test_transaction, test_finance


# =========================
# 2. feature 생성 (train과 동일)
# =========================
def make_test_features(test_customer, test_transaction, test_finance, reference_date):

    test_transaction['trans_date'] = pd.to_datetime(test_transaction['trans_date'])
    test_transaction['month']      = test_transaction['trans_date'].dt.to_period('M')

    # RFM
    test_agg_base = test_transaction.groupby('customer_id').agg(
        recency      = ('trans_date', lambda x: (reference_date - x.max()).days),
        frequency    = ('trans_id', 'count'),
        total_amount = ('trans_amount', 'sum'),
        mean_amount  = ('trans_amount', 'mean'),
        std_amount   = ('trans_amount', 'std'),
        max_amount   = ('trans_amount', 'max'),
        min_amount   = ('trans_amount', 'min'),
    ).reset_index()

    # 트렌드
    test_last_1m = test_transaction[test_transaction['trans_date'] >= pd.Timestamp('2023-12-01')]
    test_prev_5m = test_transaction[test_transaction['trans_date'] <  pd.Timestamp('2023-12-01')]

    test_agg_trend = pd.concat([
        test_last_1m.groupby('customer_id')['trans_amount'].sum().rename('amt_last_1m'),
        test_prev_5m.groupby('customer_id')['trans_amount'].sum().rename('amt_prev_5m')
    ], axis=1).fillna(0).reset_index()

    test_agg_trend['amt_trend_ratio'] = (
        test_agg_trend['amt_last_1m'] - test_agg_trend['amt_prev_5m'] / 5
    ) / (test_agg_trend['amt_prev_5m'] / 5 + 1)

    # 병합
    test_tx = test_agg_base.merge(test_agg_trend, on='customer_id', how='left')

    test_df = test_customer \
        .merge(test_finance, on='customer_id', how='left') \
        .merge(test_tx, on='customer_id', how='left')

    return test_df


# =========================
# 3. 전처리 (train과 동일)
# =========================
def preprocess_test(test_df, cat_cols, reference_date):

    test_df['join_date'] = pd.to_datetime(test_df['join_date'])
    test_df['join_days'] = (reference_date - test_df['join_date']).dt.days
    test_df = test_df.drop(columns=['join_date'])

    for col in cat_cols:
        test_df[col] = test_df[col].astype(str)  # ⚠️ fit_transform X

    return test_df


# =========================
# 4. 예측
# =========================
def predict(test_df, churn_model, ltv_model, feature_cols):

    X_test = test_df[feature_cols]

    churn_pred = churn_model.predict_proba(X_test)[:, 1]
    ltv_pred   = np.expm1(ltv_model.predict(X_test))

    return churn_pred, ltv_pred
