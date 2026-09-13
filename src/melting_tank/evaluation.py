"""모델 평가 지표 산출, 임계값(Threshold) 최적화 및 배포 승인 게이트 모듈.

혼동행렬 기반 지표와 NG 재현율·정밀도·F2-score를 평가하여
KPI 조건을 만족하는 후보 중 F2-score가 가장 높은 임계값을 선택합니다.
미탐·오경보 가중 패널티는 참고 지표로 계산합니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, fbeta_score, precision_score, recall_score

from .config import KpiConfig


@dataclass(frozen=True)
class ThresholdSelection:
    """검증 데이터셋에서 탐색된 최적의 분류 임계값 및 성능 지표 컨테이너."""

    ## 탐색을 통해 최종 선택된 이진 분류 판정 확률 임계값
    threshold: float
    ## 선택된 임계값 기준의 평가 지표 모음 (재현율, 정밀도, F2, 비즈니스 비용 등)
    metrics: dict[str, float | None]
    ## 사전 정의된 비즈니스 KPI 최소 기준을 통과한 임계값이 존재했는지 여부
    validation_kpi_satisfied: bool


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    kpi: KpiConfig,
) -> dict[str, float | None]:
    """주어진 임계값 기준으로 혼동행렬과 통계적 지표 및 비즈니스 손실 비용을 계산합니다.

    Parameters
    ----------
    y_true : np.ndarray
        실제 정답 레이블 배열 (0: 정상, 1: NG 불량).
    probabilities : np.ndarray
        모델이 예측한 양성(NG 불량) 클래스 확률값 배열 (0.0 ~ 1.0).
    threshold : float
        불량(1)으로 분류하기 위한 기준 확률 임계값.
    kpi : KpiConfig
        미탐 비용(FN Cost) 및 오탐 비용(FP Cost) 가중치 설정을 담은 객체.

    Returns
    -------
    dict[str, float | None]
        재현율, 정밀도, F2, PR-AUC, 오경보율, 비즈니스 비용 및 혼동행렬 수치를 담은 딕셔너리.
    """
    ## 입력 확률값이 임계값 이상이면 1(NG), 미만이면 0(정상)으로 이진화 판정
    predictions = (probabilities >= threshold).astype(int)

    ## 2x2 혼동행렬(Confusion Matrix) 계산 후 1차원 평탄화: [TN, FP, FN, TP]
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return {
        ## 불량 재현율 (Recall = TP / (TP + FN)): 실제 불량 중 모델이 잡아낸 비율
        "ng_recall": float(recall_score(y_true, predictions, zero_division=0)),
        ## 불량 정밀도 (Precision = TP / (TP + FP)): 불량이라고 예측한 것 중 실제 불량인 비율
        "ng_precision": float(precision_score(y_true, predictions, zero_division=0)),
        ## F2-score: beta=2를 적용하여 정밀도보다 재현율을 더 중시하는 조화평균 (미탐 방지 중심)
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        ## PR-AUC (Precision-Recall Area Under Curve): 클래스 불균형 분류 평가에 적합한 Precision-Recall 곡선 아래 면적
        ## 단일 클래스만 존재하는 특수 상황(예: test_normal)에서는 계산 불가하므로 None 처리
        "pr_auc": float(average_precision_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else None,
        ## 오경보율 (False Alarm Rate / FPR = FP / (FP + TN)): 정상 공정 중 헛알람이 울린 비율
        "false_alarm_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        ## 상대적 업무 패널티 = (미탐 건수 × 미탐 가중치) + (오경보 건수 × 오경보 가중치)
        "business_cost": float(kpi.false_negative_cost * fn + kpi.false_positive_cost * fp),
        ## 혼동행렬 원시 집계 수치
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray, kpi: KpiConfig) -> ThresholdSelection:
    """0.05부터 0.95까지 임계값을 그리드 탐색하여 KPI 기준을 충족하는 최적의 임계값을 도출합니다.

    Parameters
    ----------
    y_true : np.ndarray
        검증 데이터셋의 실제 레이블 배열.
    probabilities : np.ndarray
        검증 데이터셋에 대한 모델의 불량 예측 확률 배열.
    kpi : KpiConfig
        최소 요구 재현율(ng_recall_min)과 최소 정밀도(ng_precision_min) 명세.

    Returns
    -------
    ThresholdSelection
        도출된 최적 임계값, 산출 지표 및 KPI 충족 여부를 담은 불변 데이터클래스 인스턴스.
    """
    ## 0.05부터 0.95까지 0.01 간격으로 총 91개의 후보 임계값별 성능 지표 전수 계산
    candidates = [
        (float(threshold), calculate_metrics(y_true, probabilities, float(threshold), kpi))
        for threshold in np.arange(0.05, 0.96, 0.01)
    ]

    ## 비즈니스 필수 조건(최소 재현율 및 최소 정밀도)을 동시에 만족하는 적격 후보 필터링
    eligible = [
        item
        for item in candidates
        if item[1]["ng_recall"] >= kpi.ng_recall_min and item[1]["ng_precision"] >= kpi.ng_precision_min
    ]

    ## 적격 후보가 있으면 eligible에서 선택, 없으면 전체 후보(candidates)에서 차선책 선택
    pool = eligible or candidates

    ## 1순위: F2-score 최대화, 2순위: 불량 재현율(ng_recall) 최대화 기준으로 최적 임계값 확정
    threshold, metrics = max(pool, key=lambda item: (item[1]["f2"], item[1]["ng_recall"]))

    return ThresholdSelection(
        threshold=threshold,
        metrics=metrics,
        ## 적격 후보(eligible)가 단 하나라도 존재했는지 여부를 불리언 플래그로 저장
        validation_kpi_satisfied=bool(eligible),
    )


def deployment_gate(
    test_ng: dict[str, float | None],
    test_normal: dict[str, float | None],
    kpi: KpiConfig,
) -> dict[str, object]:
    """테스트 데이터셋 성능을 기준으로 후속 서빙 단계의 배포 후보 여부를 판정합니다.

    Parameters
    ----------
    test_ng : dict[str, float | None]
        불량 발생 테스트셋(test_ng)에서 산출된 지표 딕셔너리.
    test_normal : dict[str, float | None]
        순수 정상 공정 테스트셋(test_normal)에서 산출된 지표 딕셔너리.
    kpi : KpiConfig
        배포 승인 게이트의 기준 임계 수치 객체.

    Returns
    -------
    dict[str, object]
        최종 배포 승인 여부('approved': bool) 및 3가지 세부 지표 검사 결과 딕셔너리.
    """
    ## 배포 적격성 3대 관문 검사
    checks = {
        ## 관문 1: 불량이 포함된 상황에서 최소 요구 재현율을 달성하였는가? (미탐 방지)
        "ng_recall": test_ng["ng_recall"] >= kpi.ng_recall_min,
        ## 관문 2: 불량 예측 중 실제 불량의 비율이 최소 정밀도를 넘었는가? (오경보 신뢰도)
        "ng_precision": test_ng["ng_precision"] >= kpi.ng_precision_min,
        ## 관문 3: 순수 정상 공정 상태에서 허용 오경보율(False Alarm Rate) 이하를 유지했는가?
        "normal_false_alarm_rate": test_normal["false_alarm_rate"] <= kpi.normal_false_alarm_rate_max,
    }

    ## 세 가지 KPI를 모두 통과한 경우에만 배포 후보로 승인
    return {"approved": all(checks.values()), "checks": checks}