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
  <summary><b>Submission 4: 최종 최적화 (현재 버전)</b></summary>
  
  - **모델:** XGBoost + Hyperparameter Tuning
  - **변경 사항:** - `is_weekend` 파생 피처 추가
    - 과적합 방지를 위해 `max_depth` 조정
  - **결과:** Public Score 0.85
</details>



총 3개 입력 테이블 + 1개 타겟

  ## 1) 고객 정보 (정적 데이터)
  
  * 나이, 성별, 지역, 결혼 여부, 소득 수준 등
  -> * **기본 세그먼트 분류**
  
  ---
  
  ## 2) 거래 이력 (시계열 데이터)
  
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
  
  ## 3) 타겟
  
  * churn: binary
  * ltv: continuous

---

# 3. 데이터 규모

* Train: 60,000명
* Test: 40,000명
* feature engineering 영향 매우 큼

  ---
  
  ## 1) 모델 전략
  
  ### Churn
  
  * LightGBM / XGBoost (AUC 최적화)
  
  ### LTV
  
  * LightGBM Regressor
  * 또는 log 변환 후 회귀
  
  ---
  
  ## 2) 멀티태스크 전략
  
  * churn 예측값을 LTV feature로 사용
  * churn=1 그룹 / 0 그룹 나눠 LTV 모델 분리
  
  ---
  
  ## 3) 검증 전략 (중요)
  
  * 단순 KFold
  * 추천:
    * Stratified KFold (churn 기준)
    * Group KFold (customer 기반)
  
  ---
  
  ## 4) 리더보드 전략
  
  * 과적합 방지
  * seed averaging
  * ensemble 필수

---

# 요약
**“고객 행동 + 금융 상태를 기반으로 이탈 확률과 미래 가치를 동시에 예측하는 고난도 고객 분석 문제”**

---
# 1~3주차 중간 결과
* CV AUC  평균: 0.7923  std: 0.0080  (기존: 0.7887)
* CV RMSE 평균: 1,401,187  std: 17,666  (기존: 1,376,715)
* 신규 Score : 0.42916

## 피처 중요도
[ Churn 중요도 전체 ]
              feature  importance
total_deposit_balance         467
         credit_score         399
        card_loan_amt         390
          amt_last_1m         209
           min_amount         207
          mean_amount         206
fin_asset_trend_score         205
         online_ratio         199
         total_amount         193
    mean_purchase_gap         190
            join_days         190
      amt_trend_ratio         182
                  age         180
card_cash_service_amt         173
    installment_ratio         171
          amt_prev_5m         161
           std_amount         150
           max_amount         146
   total_loan_balance         127
            frequency         109
              recency          93
          region_code          61
      prefer_category          40
         top_category          34
         income_group          30
     fin_overdue_days          13
               gender           1
           is_married           0
     num_active_cards           0
     category_nunique           0

[ LTV 중요도 전체 ]
              feature  importance
total_deposit_balance        3754
fin_asset_trend_score        2619
         credit_score        2600
        card_loan_amt        2418
            join_days        2229
          mean_amount        2071
           min_amount        2055
      amt_trend_ratio        2009
         online_ratio        1986
    installment_ratio        1848
          amt_last_1m        1693
                  age        1692
    mean_purchase_gap        1690
           std_amount        1686
           max_amount        1579
         total_amount        1492
          amt_prev_5m        1490
              recency        1279
   total_loan_balance        1264
card_cash_service_amt         989
            frequency         782
          region_code         653
     num_active_cards         632
      prefer_category         615
         top_category         589
         income_group         464
     fin_overdue_days         236
           is_married         167
               gender         163
     category_nunique          98
     
---

# 4주차 모델
* 모델 결합
  
공식 기준 최종 결과

=============================================

평균 AUC (Churn) : 0.7924

평균 RMSE (LTV)   : 1,385,450

최종 통합 Score   : 0.42924

=============================================
