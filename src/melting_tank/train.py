"""용해로(Melting Tank) LSTM 모델 학습 및 MLOps 파이프라인 실행 모듈.

시계열 센서 데이터 전처리, LSTM 신경망 훈련, 최적 임계값 도출, 
비즈니스 KPI 기반 배포 승인 게이트 검증 및 MLflow 실험 추적을 총괄 오케스트레이션합니다.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import joblib
import mlflow
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import ProjectConfig, load_config
from .data import DatasetSplits, create_splits, limit_splits, load_raw_data
from .evaluation import calculate_metrics, deployment_gate, select_threshold


def build_model(config: ProjectConfig, input_shape: tuple[int, int]) -> keras.Model:
    """시계열 특성 분류를 위한 LSTM 신경망 아키텍처를 정의하고 컴파일합니다.

    Parameters
    ----------
    config : ProjectConfig
        LSTM 유닛 수, 드롭아웃 비율, 학습률 등 모델 하이퍼파라미터를 담은 설정 객체.
    input_shape : tuple[int, int]
        (sequence_length, n_features) 형태의 2차원 입력 텐서 형태.

    Returns
    -------
    keras.Model
        Adam 옵티마이저와 이진 교차 엔트로피 손실 함수로 컴파일된 Keras 시퀀셜 모델.
    """
    ## 순차적 신경망 계층(Sequential Model) 구성
    model = keras.Sequential(
        [
            ## 입력 계층: (시퀀스 길이, 피처 개수) 형태의 입력을 정의
            keras.layers.Input(shape=input_shape),
            ## 순환신경망 계층: 시계열 패턴 추출을 위한 LSTM 유닛 (Tanh 활성화 함수)
            keras.layers.LSTM(config.model.lstm_units, activation="tanh"),
            ## 드롭아웃 계층: 과적합 방지를 위해 무작위로 뉴런 연결을 비활성화
            keras.layers.Dropout(config.model.dropout),
            ## 완전연결 계층: 특징 비선형 결합을 위한 Dense 레이어 (ReLU 활성화 함수)
            keras.layers.Dense(config.model.dense_units, activation="relu"),
            ## 최종 출력 계층: 0~1 사이의 불량(NG) 발생 확률을 출력하는 시그모이드 뉴런
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    ## 모델 컴파일: 옵티마이저, 손실함수, 평가 지표(Accuracy, PR-AUC) 등록
    ## 클래스 불균형 데이터셋에 특화된 Precision-Recall 곡선 아래 면적(PR-AUC) 모니터링
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.model.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def split_summary(splits: DatasetSplits) -> dict[str, dict[str, float]]:
    """데이터 분할별 총 샘플 수, 불량(NG) 건수, 불량 발생 비율 통계를 산출합니다.

    Parameters
    ----------
    splits : DatasetSplits
        Train, Validation, Test-NG, Test-Normal 데이터셋이 포함된 분할 객체.

    Returns
    -------
    dict[str, dict[str, float]]
        각 데이터 분할 구간별 샘플 수(samples), 불량 수(ng), 불량 비율(ng_ratio) 요약본.
    """
    summary: dict[str, dict[str, float]] = {}
    ## 4개 분할 구간을 순회하며 실제 타깃 레이블(y)의 기초 통계량 집계
    for name in ("train", "validation", "test_ng", "test_normal"):
        y = getattr(splits, f"y_{name}")
        summary[name] = {
            "samples": int(len(y)),
            "ng": int(y.sum()),
            "ng_ratio": float(y.mean()),
        }
    return summary


def parse_args() -> argparse.Namespace:
    """CLI(명령줄 인터페이스)로부터 실행 인자들을 파싱합니다.

    Returns
    -------
    argparse.Namespace
        설정 경로, 데이터 경로, 산출물 디렉토리, 에포크, 샘플 수 등이 담긴 인자 네임스페이스.
    """
    parser = argparse.ArgumentParser(description="용해탱크 LSTM 학습 및 MLflow 기록")
    ## 프로젝트 YAML 설정 파일 경로 인자
    parser.add_argument("--config", default="config/project.yaml")
    ## 원천 데이터 CSV 경로 (환경변수 DATA_PATH 우선 적용)
    parser.add_argument("--data", default=os.getenv("DATA_PATH", "data/raw/melting_tank.csv"))
    ## 학습 산출물 로컬 저장 디렉토리 (환경변수 ARTIFACT_DIR 우선 적용)
    parser.add_argument("--artifact-dir", default=os.getenv("ARTIFACT_DIR", "artifacts"))
    ## 설정 파일의 epochs 값을 명령줄에서 동적으로 재정의하기 위한 인자
    parser.add_argument("--epochs", type=int, help="설정 파일의 epoch를 임시 재정의")
    ## 신속한 파이프라인 검증(Smoke Test)을 위한 분할별 최대 표본 분(minutes) 제한 인자
    parser.add_argument("--sample-minutes", type=int, help="각 분할의 최대 표본 수를 제한하는 smoke test 옵션")
    ## MLflow UI 대시보드에 표시될 고유 실행(Run) 명칭
    parser.add_argument("--run-name", help="MLflow run 이름")
    return parser.parse_args()


def main() -> None:
    """전체 훈련 워크플로우를 관장하는 메인 진입점 함수.

    시드 고정, 데이터 로드, 모델 빌드, 가중치 학습, 임계값 최적화, 
    배포 승인 검증 및 아티팩트 저장을 순차적으로 실행합니다.
    """
    ## CLI 인자 파싱
    args = parse_args()

    ## 실행 환경에 따른 차이를 줄이고 결과의 반복 가능성을 높이기 위해 난수 시드를 고정
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    ## 전역 YAML 설정 파일 로드
    config = load_config(args.config)

    ## 아티팩트 저장 폴더 생성 (없을 경우 자동 생성)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    ## 원천 센서 데이터 로드 및 시계열 타임스탬프 기준 분할 수행
    df = load_raw_data(args.data, config)
    splits = create_splits(df, config)

    ## 스모크 테스트 옵션이 주어진 경우 데이터셋을 균등 다운샘플링하여 빠른 검증 지원
    if args.sample_minutes is not None:
        limit_splits(splits, args.sample_minutes)

    ## 명령줄 에포크 값이 있으면 우선 적용하고, 없으면 설정 파일의 기본값 사용
    epochs = args.epochs if args.epochs is not None else config.model.epochs

    ## 입력 텐서 형태 (sequence_length, n_features)를 기반으로 LSTM 모델 인스턴스 빌드
    model = build_model(config, splits.X_train.shape[1:])

    ## MLflow 서버 연결 URI 및 실험(Experiment) 이름 환경변수 바인딩
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "melting-tank-kpi"))

    ## MLflow 컨텍스트 진입: 고유 Run 세션 시작
    with mlflow.start_run(run_name=args.run_name) as run:
        ## 1. 훈련 파라미터 및 메타데이터 일괄 로깅
        mlflow.log_params(
            {
                "project_version": config.version,
                "sequence_length": config.data.sequence_length,
                "features": ",".join(config.data.features),
                "target_rule": config.data.target_rule,
                "forecast_horizon_minutes": config.forecast_horizon_minutes,
                "epochs": epochs,
                "batch_size": config.model.batch_size,
                "ng_class_weight": config.model.ng_class_weight,
                "sample_minutes": args.sample_minutes or "full",
            }
        )

        ## 2. LSTM 신경망 학습 실행
        ## - 클래스 불균형이 학습에 미치는 영향을 완화하기 위해 class_weight 적용
        ## - 검증 손실(val_loss) 기준 조기 종료(EarlyStopping) 및 최고 가중치 자동 복원
        history = model.fit(
            splits.X_train,
            splits.y_train,
            validation_data=(splits.X_validation, splits.y_validation),
            epochs=epochs,
            batch_size=config.model.batch_size,
            class_weight={0: 1.0, 1: config.model.ng_class_weight},
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
            verbose=2,
        )

        ## 3. 검증 데이터셋에 대한 추론 확률 계산 및 최적 임계값(Threshold) 그리드 탐색
        validation_prob = model.predict(splits.X_validation, batch_size=512, verbose=0).ravel()
        selection = select_threshold(splits.y_validation, validation_prob, config.kpi)

        ## 4. 독립된 불량 테스트셋과 순수 정상 테스트셋에 대한 추론 수행
        test_ng_prob = model.predict(splits.X_test_ng, batch_size=512, verbose=0).ravel()
        test_normal_prob = model.predict(splits.X_test_normal, batch_size=512, verbose=0).ravel()

        ## 5. 도출된 최적 임계값을 적용하여 각 테스트셋의 성능 지표 산출
        test_ng_metrics = calculate_metrics(splits.y_test_ng, test_ng_prob, selection.threshold, config.kpi)
        test_normal_metrics = calculate_metrics(
            splits.y_test_normal,
            test_normal_prob,
            selection.threshold,
            config.kpi,
        )

        ## 6. 비즈니스 KPI 제약 조건(재현율, 정밀도, 오경보율) 기반 배포 승인 게이트 판정
        gate = deployment_gate(test_ng_metrics, test_normal_metrics, config.kpi)

        ## 최종 결과 종합 딕셔너리 조립
        results = {
            "selected_threshold": selection.threshold,
            "validation_kpi_satisfied": selection.validation_kpi_satisfied,
            "deployment_approved": gate["approved"],
            "deployment_checks": gate["checks"],
            "splits": split_summary(splits),
            "metrics": {
                "validation": selection.metrics,
                "test_ng": test_ng_metrics,
                "test_normal": test_normal_metrics,
            },
            "mlflow_run_id": run.info.run_id,
        }

        ## 학습 이력(Loss, Accuracy 곡선) 실수형 직렬화
        history_data = {key: [float(value) for value in values] for key, values in history.history.items()}

        ## 모델 서빙 API가 참조할 추론 메타데이터 딕셔너리
        model_config = {
            "model_version": config.version,
            "features": list(config.data.features),
            "sequence_length": config.data.sequence_length,
            "target_rule": config.data.target_rule,
            "forecast_horizon_minutes": config.forecast_horizon_minutes,
            "threshold": selection.threshold,
        }

        ## 7. 모델·전처리·평가 아티팩트를 로컬 디스크에 저장
        ## - Keras 모델 가중치 및 구조
        model.save(artifact_dir / "model.keras")
        ## - 전처리에 사용된 훈련 MinMaxScaler 인스턴스 피클링
        joblib.dump(splits.scaler, artifact_dir / "scaler.joblib")
        ## - 최종 평가 지표 JSON 파일
        (artifact_dir / "metrics.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        ## - 에포크별 수렴 이력 JSON 파일
        (artifact_dir / "history.json").write_text(json.dumps(history_data, indent=2), encoding="utf-8")
        ## - 후속 serving 저장소에서 사용할 모델 입력·임계값 명세 JSON 파일
        (artifact_dir / "model_config.json").write_text(
            json.dumps(model_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        ## 8. MLflow 지표 로깅 (무한대 또는 결측값 예외 필터링)
        for section, values in results["metrics"].items():
            mlflow.log_metrics(
                {
                    f"{section}_{key}": value
                    for key, value in values.items()
                    if value is not None and np.isfinite(value)
                }
            )

        ## 핵심 KPI 배포 판정 지표 단독 로깅
        mlflow.log_metric("selected_threshold", selection.threshold)
        mlflow.log_metric("validation_kpi_satisfied", int(selection.validation_kpi_satisfied))
        mlflow.log_metric("deployment_approved", int(gate["approved"]))

        ## 이번 학습에서 생성한 핵심 아티팩트 5개를 MLflow Artifact Repository에 기록
        artifact_files = [
            artifact_dir / "model.keras",
            artifact_dir / "scaler.joblib",
            artifact_dir / "metrics.json",
            artifact_dir / "history.json",
            artifact_dir / "model_config.json",
        ]

        for artifact_file in artifact_files:
            mlflow.log_artifact(str(artifact_file), artifact_path="model_bundle")

        ## 터미널 표준 출력에 JSON 형식으로 최종 리포트 출력
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()