"""프로젝트 전역 설정(Configuration) 관리 모듈.

YAML 파일로부터 데이터 전처리, 딥러닝 모델 아키텍처, 
비즈니스 KPI 제약 조건을 파싱하여 불변(Frozen) 데이터클래스로 로드합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    """시계열 데이터 전처리, 레이블링 및 기간 분할 설정을 담는 데이터클래스."""

    ## LSTM/RNN 모델 입력을 위한 과거 관측 윈도우 크기 (타임스텝 수)
    sequence_length: int
    ## 모델 학습에 사용할 센서 수집 특성(Feature) 컬럼 명칭 목록 (불변 튜플)
    features: tuple[str, ...]
    ## 지도학습 대상 종속변수(Target) 컬럼명
    target: str
    ## 양성(Positive)으로 판정할 이상/불량 클래스 레이블 명칭 (예: 'NG')
    positive_label: str
    ## 타깃 생성 비즈니스 로직 및 판정 규칙 명세
    target_rule: str
    ## 훈련 데이터셋 종료 타임스탬프 (시계열 데이터 누수 방지용 기준선)
    train_end: str
    ## 검증 데이터셋 종료 타임스탬프 (모델 조기 종료 및 튜닝 기준선)
    validation_end: str
    ## 테스트 데이터셋 종료 타임스탬프 (최종 불량 예측 성능 평가 기준선)
    test_ng_end: str


@dataclass(frozen=True)
class ModelConfig:
    """LSTM 신경망 아키텍처 및 학습 하이퍼파라미터 설정을 담는 데이터클래스."""

    ## 순환신경망 LSTM 계층의 은닉 유닛(Hidden Units) 차원 수
    lstm_units: int
    ## LSTM 출력단 이후 배치되는 완전연결(Dense) 계층의 노드 수
    dense_units: int
    ## 신경망 과적합(Overfitting) 방지를 위한 드롭아웃 비율 (0.0 ~ 1.0)
    dropout: float
    ## 옵티마이저의 가중치 업데이트 보폭을 결정하는 학습률 (Learning Rate)
    learning_rate: float
    ## 경사하강법 1회 반복 시 모델에 동시 주입되는 미니배치 샘플 수
    batch_size: int
    ## 전체 훈련 데이터셋에 대한 총 반복 학습 횟수 (Epochs)
    epochs: int
    ## 클래스 불균형이 손실함수에 미치는 영향을 완화하기 위한 NG 클래스 가중치
    ng_class_weight: float


@dataclass(frozen=True)
class KpiConfig:
    """비즈니스 목표 달성 및 경제적 비용 최적화를 위한 KPI 평가 지표 설정."""

    ## 모델 배포를 위해 반드시 만족해야 하는 불량(NG) 최소 재현율 (Target Recall)
    ng_recall_min: float
    ## 오경보로 인한 현장 피로도를 방지하기 위한 불량(NG) 최소 정밀도 (Target Precision)
    ng_precision_min: float
    ## 정상 공정 데이터 중 잘못된 알람이 발생하는 최대 허용 오경보율 (False Positive Rate)
    normal_false_alarm_rate_max: float
    ## NG를 탐지하지 못한 경우에 부여하는 상대적 미탐 패널티
    false_negative_cost: float
    ## 정상 상태를 NG로 잘못 예측한 경우에 부여하는 상대적 오경보 패널티
    false_positive_cost: float


@dataclass(frozen=True)
class ProjectConfig:
    """하위 설정 모듈(Data, Model, KPI)을 통합 총괄하는 최상위 프로젝트 설정 클래스."""

    ## MLOps 프로젝트 고유 식별 명칭 (예: 'melting-tank-kpi-mlops-modeling')
    name: str
    ## 시맨틱 버저닝 규칙에 따른 프로젝트 버전 (예: '0.2.0')
    version: str
    ## 모델이 현재 시점으로부터 사전 예측해야 하는 미래 목표 시간 간격 (분 단위)
    forecast_horizon_minutes: int
    ## 시계열 데이터 관련 세부 파라미터 객체
    data: DataConfig
    ## 신경망 학습 관련 세부 파라미터 객체
    model: ModelConfig
    ## 평가 및 비즈니스 KPI 관련 세부 파라미터 객체
    kpi: KpiConfig


def load_config(path: str | Path = "config/project.yaml") -> ProjectConfig:
    """지정된 경로의 YAML 설정 파일을 읽어 ProjectConfig 객체로 파싱 및 반환합니다.

    Parameters
    ----------
    path : str | Path, default="config/project.yaml"
        파싱 대상 프로젝트 YAML 설정 파일의 로컬 파일 시스템 경로.

    Returns
    -------
    ProjectConfig
        타입 힌트가 적용되고 무결성이 보장된 최상위 프로젝트 불변 설정 인스턴스.

    Raises
    ------
    FileNotFoundError
        주어진 경로에 YAML 설정 파일이 존재하지 않는 경우 발생합니다.
    yaml.YAMLError
        YAML 문법 형식이 잘못되었거나 파싱이 불가능한 경우 발생합니다.
    KeyError
        YAML 명세서 내에 필수 설정 키가 누락되었을 경우 발생합니다.
    """
    ## 지정된 경로에서 UTF-8 인코딩으로 텍스트를 읽고 안전하게 파싱하여 딕셔너리로 변환
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    ## 파싱된 딕셔너리를 각 전용 불변 데이터클래스로 조립하여 최종 객체 생성
    return ProjectConfig(
        name=raw["project"]["name"],
        version=raw["project"]["version"],
        forecast_horizon_minutes=raw["project"]["forecast_horizon_minutes"],
        ## DataConfig 인스턴스 생성: 튜플 변환을 통한 런타임 컬럼 목록 수정 원천 차단
        data=DataConfig(
            sequence_length=raw["data"]["sequence_length"],
            features=tuple(raw["data"]["features"]),
            target=raw["data"]["target"],
            positive_label=raw["data"]["positive_label"],
            target_rule=raw["data"]["target_rule"],
            train_end=raw["data"]["train_end"],
            validation_end=raw["data"]["validation_end"],
            test_ng_end=raw["data"]["test_ng_end"],
        ),
        ## ModelConfig 인스턴스 생성: 언패킹 연산자를 통해 딕셔너리 키-값을 일괄 주입
        model=ModelConfig(**raw["model"]),
        ## KpiConfig 인스턴스 생성: 비즈니스 비용 및 평가 제약 조건 주입
        kpi=KpiConfig(**raw["kpi"]),
    )