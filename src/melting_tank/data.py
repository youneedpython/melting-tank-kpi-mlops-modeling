"""용해로(Melting Tank) 시계열 센서 데이터 전처리 및 시퀀스 생성 모듈.

원천 CSV 데이터의 유효성을 검증하고, 1분 단위 센서 시퀀스를 생성하여 
다음 1분의 관측값 중 NG가 절반 이상인 상태를 예측하는 지도학습 데이터셋(Train, Validation, Test)을 분할합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import ProjectConfig


@dataclass
class DatasetSplits:
    """시계열 분할 및 스케일링이 완료된 훈련, 검증, 테스트 데이터셋 컨테이너."""

    ## 훈련용 입력 특성 시퀀스 배열 (Shape: [N_train, sequence_length, n_features])
    X_train: np.ndarray
    ## 훈련용 타깃 레이블 배열 (Shape: [N_train,])
    y_train: np.ndarray
    ## 하이퍼파라미터 튜닝 및 조기 종료용 검증 입력 특성 시퀀스 배열
    X_validation: np.ndarray
    ## 검증용 타깃 레이블 배열
    y_validation: np.ndarray
    ## 불량(NG) 발생 패턴이 포함된 테스트셋 입력 특성 시퀀스 배열
    X_test_ng: np.ndarray
    ## 불량 발생 테스트셋 타깃 레이블 배열
    y_test_ng: np.ndarray
    ## 순수 정상 공정 상태만 모아둔 테스트셋 입력 특성 시퀀스 배열 (오경보율 계측용)
    X_test_normal: np.ndarray
    ## 순수 정상 공정 테스트셋 타깃 레이블 배열 (전부 0)
    y_test_normal: np.ndarray
    ## 훈련 데이터셋의 통계량(Min, Max)으로만 학습된 정규화 스케일러 인스턴스
    scaler: MinMaxScaler


def load_raw_data(path: str | Path, config: ProjectConfig) -> pd.DataFrame:
    """원천 CSV 파일을 로드하여 필수 컬럼 존재 여부, 결측치를 검증하고 정제된 데이터프레임을 반환합니다.

    Parameters
    ----------
    path : str | Path
        불러올 원천 데이터 CSV 파일 경로.
    config : ProjectConfig
        필수 특성(features), 타깃(target), 양성 레이블(positive_label) 정의를 포함하는 설정 객체.

    Returns
    -------
    pd.DataFrame
        시간순으로 정렬되고 'target_ng' 이진 레이블이 추가된 정제 데이터프레임.

    Raises
    ------
    ValueError
        필수 컬럼이 누락되었거나, 결측값(NaN)이 존재하는 경우 발생합니다.
    """
    ## 원천 CSV 데이터 로드
    df = pd.read_csv(path)

    ## 기준 시간(STD_DT), 타깃, 센서 특성들이 데이터에 온전히 존재하는지 집합(Set)으로 검증
    required = {"STD_DT", config.data.target, *config.data.features}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"필수 열이 없습니다: {missing}")

    ## 필요한 컬럼만 추출하여 명시적 복사본 생성
    df = df[["STD_DT", *config.data.features, config.data.target]].copy()

    ## 날짜/시간 문자열을 판다스 Timestamp 형식으로 변환 (형식 오류 시 예외 발생)
    df["STD_DT"] = pd.to_datetime(df["STD_DT"], errors="raise")

    ## 양성 레이블(예: 'NG') 일치 여부를 불리언 판정 후 1바이트 정수형(int8)으로 변환
    df["target_ng"] = (df[config.data.target] == config.data.positive_label).astype("int8")

    ## 센서 특성값 또는 생성된 타깃에 결측치(NaN)가 하나라도 존재하면 파이프라인 중단
    if df[[*config.data.features, "target_ng"]].isna().any().any():
        raise ValueError("입력 데이터에 결측값이 있습니다.")

    ## 시계열 순서 보장을 위해 안정 정렬(Stable Sort)을 수행하고 인덱스 초기화
    return df.sort_values("STD_DT", kind="stable").reset_index(drop=True)


def build_minute_sequences(
    df: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """현재 1분의 10개 센서값으로 다음 1분의 majority-NG를 예측하기 위한 시계열 시퀀스를 생성합니다.

    Parameters
    ----------
    df : pd.DataFrame
        load_raw_data()를 통해 검증 및 시간순 정렬된 데이터프레임.
    config : ProjectConfig
        sequence_length, target_rule, features 명세를 포함하는 설정 객체.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - X_seq: 3차원 시퀀스 배열 (Shape: [N, sequence_length, n_features])
        - y_seq: 다음 1분의 NG가 절반 이상인지 나타내는 타깃 배열 (Shape: [N,])
        - times_seq: 각 시퀀스의 기준 시점 타임스탬프 배열 (Shape: [N,])

    Raises
    ------
    ValueError
        지원하지 않는 target_rule이거나 시퀀스를 생성하기에 데이터 길이가 부족한 경우 발생합니다.
    """
    ## 비즈니스 룰 정의 검증 (현재 1분 센서값 -> 다음 1분 다수결 NG 판정 룰만 허용)
    if config.data.target_rule != "next-minute-majority":
        raise ValueError(f"지원하지 않는 target_rule입니다: {config.data.target_rule}")

    feature_names = list(config.data.features)
    timestamps: list[np.datetime64] = []
    sequences: list[np.ndarray] = []
    minute_targets: list[int] = []

    ## NG가 절반 이상인지 판정할 기준 계산
    ## sequence_length=10이면 NG가 5개 이상일 때 1로 판정 (예: 1분간 센서 수집 10개 기준, (10 + 1) // 2 = 5개 이상일 때 NG 판정)
    majority_count = (config.data.sequence_length + 1) // 2

    ## 동일 분(STD_DT)으로 데이터를 그룹화하여 순회
    for timestamp, group in df.groupby("STD_DT", sort=True):
        ## 센서 샘플링 수가 규격(sequence_length)과 맞지 않는 불완전한 분 데이터는 스킵
        if len(group) != config.data.sequence_length:
            continue
        ## 1분간의 다차원 센서 측정값을 float32 2차원 넘파이 배열로 변환하여 수집
        sequences.append(group[feature_names].to_numpy(dtype="float32"))
        ## 1분 동안 발생한 NG 개수가 판정 기준 이상인지 확인
        minute_targets.append(int(group["target_ng"].sum() >= majority_count))
        ## 시점 메타데이터 저장
        timestamps.append(np.datetime64(timestamp))

    ## 리스트를 고속 연산을 위한 넘파이 배열로 캐스팅
    X_current = np.asarray(sequences, dtype="float32")
    y_current = np.asarray(minute_targets, dtype="int8")
    times_current = np.asarray(timestamps)
    if len(X_current) < 2:
        raise ValueError("시퀀스를 만들 데이터가 부족합니다.")

    ## 누락된 분(Time gap)을 건너뛰어 물리적으로 연속되지 않은 시점이 강제로 연결되는 것을 방지
    ## 현재 분(t)과 다음 분(t+1) 사이의 시간 차이가 정확히 1분(1m)인 경우만 True로 마스킹
    is_next_minute = (times_current[1:] - times_current[:-1]) == np.timedelta64(1, "m")

    ## 현재 분 X[t]와 다음 분의 절반 이상 NG 여부 y[t+1]을 매핑
    return X_current[:-1][is_next_minute], y_current[1:][is_next_minute], times_current[:-1][is_next_minute]


def _scale_sequences(train: np.ndarray, *others: np.ndarray) -> tuple[MinMaxScaler, list[np.ndarray]]:
    """훈련 데이터의 2차원 평탄화 분포로 MinMaxScaler를 학습시키고 모든 시퀀스를 3차원 형태로 복원 변환합니다.

    Parameters
    ----------
    train : np.ndarray
        스케일러 fitting의 기준이 되는 훈련용 3차원 시퀀스 배열 (Shape: [N_train, Seq_len, Features]).
    *others : np.ndarray
        동일한 스케일러로 변환만 수행할 검증 및 테스트 3차원 시퀀스 배열들.

    Returns
    -------
    tuple[MinMaxScaler, list[np.ndarray]]
        훈련 완료된 MinMaxScaler 인스턴스와 스케일링이 적용된 3차원 넘파이 배열 리스트.
    """
    scaler = MinMaxScaler()
    n_features = train.shape[-1]

    ## 3차원 시퀀스(샘플 수, 타임스텝, 특성 수)를 2차원(총 타임스텝 수, 특성 수)으로 펼쳐서 훈련 데이터만 fit
    ## [중요] 검증·테스트 데이터의 통계량이 scaler 학습에 포함되는 누수를 방지
    scaler.fit(train.reshape(-1, n_features))

    ## fit된 통계량을 기반으로 훈련셋 및 검증/테스트셋을 동일하게 transform하고 원래 3차원 텐서로 재구성
    transformed = [
        scaler.transform(array.reshape(-1, n_features)).reshape(array.shape).astype("float32")
        for array in (train, *others)
    ]
    return scaler, transformed


def create_splits(df: pd.DataFrame, config: ProjectConfig) -> DatasetSplits:
    """시계열 타임스탬프 기준으로 Train, Validation, Test-NG, Test-Normal 4개 구간으로 엄격히 분할합니다.

    Parameters
    ----------
    df : pd.DataFrame
        정제된 원천 데이터프레임.
    config : ProjectConfig
        각 분할 구간의 종료 시점(train_end, validation_end, test_ng_end)을 담은 설정 객체.

    Returns
    -------
    DatasetSplits
        구간 분할 및 스케일링이 완료된 4개 데이터셋과 스케일러를 담은 객체.

    Raises
    ------
    ValueError
        분할된 구간 중 데이터가 단 하나도 존재하지 않는 빈 세트가 발생할 경우 발생합니다.
    """
    ## 1분 단위 시계열 시퀀스와 다음 1분 타깃 생성
    X, y, times = build_minute_sequences(df, config)
    train_end = np.datetime64(config.data.train_end)
    validation_end = np.datetime64(config.data.validation_end)
    test_ng_end = np.datetime64(config.data.test_ng_end)

    ## 시점 경계선에 따른 불리언 마스크 사전 정의 (과거 -> 미래 순서 유지, 무작위 셔플 금지)
    masks = {
        "train": times <= train_end,
        "validation": (times > train_end) & (times <= validation_end),
        "test_ng": (times > validation_end) & (times <= test_ng_end),
        "test_normal": times > test_ng_end,
    }

    ## 어떤 분할이라도 샘플 수가 0개이면 조기 감지하여 예외 처리
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"{name} 분할이 비어 있습니다.")

    ## 4개 분할의 입력 시퀀스 텐서 슬라이싱
    names = ("train", "validation", "test_ng", "test_normal")
    arrays = [X[masks[name]] for name in names]

    ## 데이터 누수 없는 정규화 수행 (훈련 데이터 기준 스케일링)
    scaler, scaled = _scale_sequences(*arrays)

    ## 스케일링된 데이터 분할 컨테이너를 생성하여 반환
    return DatasetSplits(
        X_train=scaled[0],
        y_train=y[masks["train"]],
        X_validation=scaled[1],
        y_validation=y[masks["validation"]],
        X_test_ng=scaled[2],
        y_test_ng=y[masks["test_ng"]],
        X_test_normal=scaled[3],
        y_test_normal=y[masks["test_normal"]],
        scaler=scaler,
    )


def limit_splits(splits: DatasetSplits, limit: int) -> None:
    """스모크 테스트 및 신속한 파이프라인 연결 검증을 위해 각 구간별 표본을 균등 간격으로 다운샘플링합니다.

    Parameters
    ----------
    splits : DatasetSplits
        데이터가 담긴 분할 객체 (내부 속성이 In-place로 변경됨).
    limit : int
        구간별로 추출할 최대 표본 분(Minutes) 수.

    Raises
    ------
    ValueError
        limit 값이 1 미만일 경우 발생합니다.
    """
    if limit < 1:
        raise ValueError("sample-minutes는 1 이상이어야 합니다.")

    ## 4개 분할 데이터셋을 순회하며 균등 간격 인덱스를 추출하여 슬라이싱 적용
    for name in ("train", "validation", "test_ng", "test_normal"):
        X_part = getattr(splits, f"X_{name}")
        y_part = getattr(splits, f"y_{name}")
        ## np.linspace를 통해 시계열 데이터의 시작점부터 끝점까지 고른 간격으로 인덱스 샘플링
        indices = np.linspace(0, len(X_part) - 1, min(limit, len(X_part)), dtype=int)
        ## 추출된 서브셋으로 DatasetSplits 속성을 제자리에서(In-place) 교체
        setattr(splits, f"X_{name}", X_part[indices])
        setattr(splits, f"y_{name}", y_part[indices])