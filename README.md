# Melting Tank KPI MLOps Modeling

용해탱크 시계열 데이터로 다음 1분의 majority-NG를 예측하는 교육용 모델링 저장소입니다. **v2는 v1의 Notebook을 유지하면서**, 동일한 실험을 재현 가능한 Python 학습 파이프라인으로 모듈화하고 MLflow 추적 기능을 추가한 버전입니다.

## 버전별 학습 흐름

| 버전 | 학습 목표 | 주요 구성 |
|---|---|---|
| v1.0.0 | 모델 설계와 평가 | 실행 결과가 포함된 Notebook 4개 |
| v2.0.0 | 실험 재현과 추적 | v1 전체 + `src/`, `config/`, `tests/`, MLflow |

FastAPI, 대시보드, SQLite 및 API 컨테이너는 별도의 `melting-tank-kpi-mlops-serving` 저장소에서 다룹니다.

## 공통 모델링 정의

| 항목 | 설정 |
|---|---|
| 입력 | MELT_TEMP, MOTORSPEED, MELT_WEIGHT |
| 시퀀스 | 현재 1분의 관측값 10개 |
| 타깃 | 다음 1분의 관측값 중 NG가 5개 이상이면 1 |
| 양성 클래스 | NG=1 |
| 예측 선행시간 | 1분 |
| 데이터 분할 | 시간순 Train/Validation/Test-NG/Test-Normal |
| 모델 | LSTM(32) → Dropout(0.2) → Dense(16) → Sigmoid |
| 임계값 | 검증 KPI 후보 중 F2-score 최대 |

## 프로젝트 구조

```text
notebooks/                  v1 Notebook 4개와 실행 결과
├── 01_data_understanding.ipynb
├── 02_preprocessing.ipynb
├── 03_lstm_training.ipynb
└── 04_model_evaluation.ipynb
config/project.yaml         v2 데이터·모델·KPI 설정
src/melting_tank/           v2 전처리·학습·평가 모듈
tests/                      v2 단위 테스트
data/raw/                   원본 CSV(Git 추적 제외)
data/processed/             Notebook 전처리 결과(Git 추적 제외)
artifacts/                  모델·평가 결과(Git 추적 제외)
mlflow-data/                MLflow 데이터(Git 추적 제외)
docs/                       문제 정의·평가 기준·실행 문서
docker-compose.yml          MLflow Tracking Server 실행
```

## 환경 구성

Python 3.11 또는 3.12를 사용합니다.

```bash
conda create -n melting-tank-v2 python=3.12 -y
conda activate melting-tank-v2
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Python 3.13은 프로젝트 지원 범위에서 제외합니다.

## 데이터 준비

`melting_tank.csv`를 다음 위치에 복사합니다.

```text
data/raw/melting_tank.csv
```

원본 데이터와 학습 산출물은 `.gitignore`에 의해 GitHub에 올라가지 않습니다. Notebook 내부에 저장된 표와 그래프 출력은 버전관리됩니다.

## v1 Notebook 실행

다음 순서로 실행합니다.

1. `01_data_understanding.ipynb`
2. `02_preprocessing.ipynb`
3. `03_lstm_training.ipynb`
4. `04_model_evaluation.ipynb`

## v2 테스트

```bash
pytest -q
ruff check .
```

## MLflow 실행

```bash
docker compose up -d mlflow
```

MLflow UI: `http://127.0.0.1:5000`

PowerShell에서는 다음 환경변수를 지정합니다.

```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

환경변수를 지정하지 않으면 로컬 `mlruns/`에 기록됩니다.

## v2 학습 실행

빠른 연결 확인:

```bash
python -m melting_tank.train --epochs 1 --sample-minutes 1000 --run-name smoke-v2
```

전체 데이터 학습:

```bash
python -m melting_tank.train --run-name baseline-v2-full
```

생성되는 모델과 평가 파일은 `artifacts/`에 저장되고 MLflow에도 기록됩니다. KPI 미달 시 `deployment_approved=false`로 기록하며, v2에서는 모델을 등록하거나 서빙하지 않습니다.

자세한 내용은 다음 문서를 참고합니다.

- [문제 정의](docs/01_문제정의.md)
- [모델 평가 기준](docs/02_모델평가기준.md)
- [v1·v2 정합성](docs/03_v1_v2_정합성.md)
- [v2 실행 절차](docs/04_v2_실행_절차.md)

