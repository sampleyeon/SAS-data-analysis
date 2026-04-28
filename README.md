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
============================== [ V1_Baseline RESULT ] ==============================
Final Score: 0.4230 (AUC: 0.7804, LTV_Score: 0.0657)
LTV RMSE: 1,515,256

        [ V1_Baseline Churn Top 10 ]
              feature        importance
    total_deposit_balance          25
            card_loan_amt          17
             credit_score          15
    card_cash_service_amt           8
    fin_asset_trend_score           4
              amt_prev_5m           3
        mean_purchase_gap           2
          amt_trend_ratio           2
                join_days           2
        installment_ratio           2
    
    [ V1_Baseline LTV Top 10 ]
              feature        importance
    total_deposit_balance         311
            card_loan_amt         217
             credit_score         111
    fin_asset_trend_score         106
        installment_ratio          74
    card_cash_service_amt          65
              mean_amount          61
                join_days          52
               min_amount          51
          amt_trend_ratio          49
    ======================================================================
</details>

<details>
  <summary><b>exp_v2: sqrt 변환</b></summary>
  
  - **모델:** LightGBM
  - **변경 사항:** LTV sqrt 변환
  - **결과:** Churn AUC v2 : 0.7829 , LTV RMSE v2 (sqrt 변환) : 1,418,370
</details>

<details>
  <summary><b>exp_v3: feature engineering 실험 → 실패 → 정리된 최종 모델</b></summary>
  
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
</details>

=============================================
