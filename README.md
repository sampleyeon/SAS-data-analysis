# SAS 데이터 분석 공모전

## 공모전 개요

<details>
  <summary><b>문제 정의</b></summary>

 ### Task 1: Churn (이탈)
  
  * 목표: 고객 이탈 여부 확률 예측   
  * 출력: 0~1 확률값
  * 평가: **ROC-AUC**
  - 누가 이탈할지보다 **이탈할 가능성이 높은 순서 정렬**이 중요
  
  ---------------------------------------------------------------------
  
  ### Task 2: LTV (고객 생애 가치)
  
  * 목표: 향후 1년 LTV 금액 예측
  * 평가: **RMSE**
  - 절대적인 금액 예측 정확도 중요 (회귀 문제)
  
  -----------------------------------------------------------------------
  
  ### 최종 점수
  
  * AUC 50% + LTV 변환 점수 50%
  * **분류 + 회귀 동시에 최적화 필요**
  
  ---
</details>

<details>
  <summary><b>데이터 구조</b></summary>
  
총 3개 입력 테이블 + 1개 타겟

  ### 1) 고객 정보 (정적 데이터)
  
  * 나이, 성별, 지역, 결혼 여부, 소득 수준 등
  -> * **기본 세그먼트 분류**
  
  ---
  
  ### 2) 거래 이력 (시계열 데이터)
  
  * 기간: 2023-07 ~ 2023-12
  * 구매 금액, 채널, 카테고리, 할부 여부
  * 반드시 aggregation + feature engineering 필요
  
  예:
  
  * 최근 1개월 소비금액
  * 구매 빈도
  * 카테고리 다양성
  * 온라인 vs 오프라인 비율
  * 소비 감소 추세
  
  ---
  
  ### 3) 타겟
  
  * churn: binary
  * ltv: continuous

---

  ### 4) 데이터 규모

  * Train: 60,000명
  * Test: 40,000명
  * feature engineering 영향 매우 큼
</details>

<details>
  <summary><b>모델</b></summary>
  
  ### 1) 모델 전략
  
  #### Churn
  
  * LightGBM / XGBoost (AUC 최적화)
  
  #### LTV
  
  * LightGBM Regressor
  * 또는 log 변환 후 회귀
  
  ---
  
  ### 2) 멀티태스크 전략
  
  * churn 예측값을 LTV feature로 사용
  * churn=1 그룹 / 0 그룹 나눠 LTV 모델 분리
  
  ---
  
  ### 3) 검증 전략 (중요)
  
  * 단순 KFold
  * 추천:
    * Stratified KFold (churn 기준)
    * Group KFold (customer 기반)
  
  ---
  
  ### 4) 리더보드 전략
  
  * 과적합 방지
  * seed averaging
  * ensemble 필수

</details>

**“고객 행동 + 금융 상태를 기반으로 이탈 확률과 미래 가치를 동시에 예측하는 고난도 고객 분석 문제”**

---
## 실험 및 제출 기록 (Submission History)

<details>
  <summary><b>exp_v1: baseline model</b></summary>
  
  - **모델:** LightGBM
  - **결과:**

======= [ V1_Baseline RESULT ] =======

Final Score: 0.4230 (AUC: 0.7804, LTV_Score: 0.0657)
LTV RMSE: 1,515,256


[ V1_Baseline Churn Top 10 ] 

| feature | importance |
|--------|-----------|
| total_deposit_balance | 25 |
| card_loan_amt | 17 |
| credit_score | 15 |
| card_cash_service_amt | 8 |
| fin_asset_trend_score | 4 |
| amt_prev_5m | 3 |
| mean_purchase_gap | 2 |
| amt_trend_ratio | 2 |
| join_days | 2 |
| installment_ratio | 2 |
  
    
[ V1_Baseline LTV Top 10 ]
| feature | importance |
|--------|-----------|
| total_deposit_balance | 311 |
| card_loan_amt | 217 |
| credit_score | 111 |
| fin_asset_trend_score | 106 |
| installment_ratio | 74 |
| card_cash_service_amt | 65 |
| mean_amount | 61 |
| join_days | 52 |
| min_amount | 51 |
| amt_trend_ratio | 49 |

=========================================================================
</details>

<details>
  <summary><b>exp_v2: sqrt 변환</b></summary>
  
  - **모델:** LightGBM
  - **변경 사항:** LTV sqrt 변환
  - **결과:** Churn AUC v2 : 0.7829 , LTV RMSE v2 (sqrt 변환) : 1,418,370

  [ V2_Churn Top 10 ] 

| feature | importance |
|--------|-----------|
| total_deposit_balance | 175 |
| card_loan_amt | 165 |
| credit_score | 117 |
| card_cash_service_amt | 82 |
| fin_asset_trend_score | 54 |
| amt_prev_1m | 35 |
| installment_ratio | 32 |
| online_ratio | 30 |
| join_days | 29 |
| min_amount | 25 |
  
    
[ V2_LTV Top 10 ]
| feature | importance |
|--------|-----------|
| fin_asset_trend_score | 110 |
| min_amount | 85 |
| join_days | 84 |
| mean_amount | 80 |
| amt_prev_5m | 79 |
| max_amount | 70 |
| total_deposit_balance | 66 |
| amt_last_1m | 65 |
| credit_score | 58 |
| online_ratio | 54 |
  
</details>

<details>
  <summary><b>exp_v3: feature engineering </b></summary>
  
  - **모델:** LightGBM
  - **변경 사항:**
    - amt_last_2m: 최근 2개월 총 소비금액
    - amt_prev_4m: 이전 4개월 총 소비금액
    - spend_drop_ratio: 과거 대비 최근 소비 감소 정도 (핵심 churn signal)
    - monthly_amt_std: 월별 소비금액의 표준편차
    - last_month_freq: 최근 1개월 거래 횟수
    - [제거 피처]
        - amt_last_2m
        - amt_prev_4m
        - spend_drop_ratio
        - monthly_amt_std
        - last_month_freq
  - **결과:** Churn AUC v3 : 0.7830, LTV RMSE  v3 : 1,406,781

  [ V3_Churn Top 10 ] 

| feature | importance |
|--------|-----------|
| total_deposit_balance | 165 |
| card_loan_amt | 139 |
| credit_score | 93 |
| card_cash_service_amt | 59 |
| fin_asset_trend_score | 38 |
| installment_ratio | 24 |
| mean_purchase_gap | 22 |
| amt_last_1m | 19 |
| amt_prev_4m | 29 |
| spend_drop_ratio | 17 |
  
    
  [ V3_LTV Top 10 ]

| feature | importance |
|--------|-----------|
| total_deposit_balance | 1003 |
| card_loan_amt | 820 |
| fin_asset_trend_score | 405 |
| credit_score | 380 |
| mean_amount | 262 |
| age | 299 |
| min_amount | 262 |
| installment_ratio | 250 |
| online_ratio | 249 |
| monthly_amt_std | 233 |


</details>

<details>
  <summary><b>exp_v4: 하이퍼파라미터 최적화 및 교차검증 </b></summary>
  
  - **모델:**
    - **XGBoost**: early_stopping_rounds를 포함한 생성자 최적화로 학습 안정성 확보.
    - **lightGBM**: 트리 깊이 및 scale_pos_weight 조정을 통한 이탈 클래스 불균형 해소    
  - **변경 사항:**
    - **과적합(Overfitting) 방지**: 단순 RFM(Recency, Frequency, Monetary)을 넘어, 최근 1개월 대 이전 5개월의 소비 트렌드 및 구매 가속도(velocity_amt) 피처를 추가하여 고객의 이탈 징후를 조기에 포착.
    - **앙상블 예측 시스템 구축**: LGBM과 XGBoost를 결합한 소프트 보팅(Soft Voting) 방식의 앙상블을 적용하여 단일 모델의 편향(Bias)을 줄이고 예측 성능을 극대화.
    - **타겟 데이터 분포 최적화**: LTV(Life Time Value) 예측 시 왜도(Skewness) 문제를 해결하기 위해 Square Root(sqrt) 변환을 적용, 모델이 롱테일 분포의 데이터를 더 효과적으로 학습하도록 개선.
  
  - **결과:** Churn AUC v4 : 0.78923, LTV RMSE  v4 : 1,401,187
</details>

<details>
  <summary><b>exp_v5: 하이퍼파라미터 최적화 및 교차검증 </b></summary>
  
  - **모델:** LightGBM
  - **변경 사항:**
    - **가속도 기반 피처 엔지니어링**: L1 규제(`reg_alpha`)와 L2 규제(`reg_lambda`)를 추가하여 복잡한 모델이 학습 데이터에만 치중되지 않도록 제어.
    - **학습 정밀도 최적화**: `learning_rate`를 기존보다 낮은 `0.005`로 설정하고, `num_leaves`를 `63`으로 늘려 데이터의 미세한 패턴을 더 깊게 학습하도록 유도.
  
  - **결과:** Churn AUC v5 : 0.7924, LTV RMSE  v5 : 1,385,450
</details>

=============================================
