# Melting Tank KPI MLOps — v1 모델 개발

용해탱크 시계열 데이터로 **현재 1분의 센서값을 이용해 다음 1분의 NG 여부를 예측**하는 LSTM 모델링 학습 프로젝트입니다.  
v1의 범위는 Notebook 기반 데이터 이해, 전처리, 모델 학습, KPI 평가까지입니다.

## 학습 목표

- 시계열 데이터에서 랜덤 분할을 피하고 시간순으로 분할한다.
- 한 분에 수집된 10개 관측값을 하나의 LSTM 입력 시퀀스로 만든다.
- 다음 1분에 NG가 다수 발생하는지를 예측한다.
- Accuracy뿐 아니라 NG 정밀도·재현율·F2와 정상 구간 오경보율을 평가한다.
- 운영 KPI를 만족하지 못한 모델은 배포하지 않는다는 판단을 내린다.

## 실행 환경

Python 3.11 또는 3.12를 권장합니다. Python 3.13은 TensorFlow 등 의존성 호환 문제를 피하기 위해 사용하지 않습니다.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

## 데이터 준비

`melting_tank.csv`를 `data/raw/melting_tank.csv`에 복사합니다.  
원본 데이터는 용량과 배포 정책 때문에 Git 추적 대상에서 제외했습니다.

## Notebook 실행 순서

1. `01_data_understanding.ipynb` — 구조, 결측치, 분포, 시간 패턴 확인
2. `02_preprocessing.ipynb` — 분 단위 시퀀스, 다음 1분 타깃, 시간순 분할과 스케일링
3. `03_lstm_training.ipynb` — Baseline LSTM 학습과 모델 저장
4. `04_model_evaluation.ipynb` — 임계값 탐색, KPI 평가와 배포 가능 여부 판단

Notebook은 반드시 위 순서대로 실행합니다. 빠른 코드 점검은 각 Notebook의 `QUICK_RUN = True`로 바꾸고, 최종 평가는 `False`로 수행합니다.

## 문제 정의

- 입력: `MELT_TEMP`, `MOTORSPEED`, `MELT_WEIGHT`
- 시퀀스: 현재 1분의 관측값 10개
- 타깃: 다음 1분 관측값 10개 중 NG가 5개 이상이면 `1`, 아니면 `0`
- 양성 클래스: NG = 1
- 제외 변수: `INSP`는 실제 추론 시점의 가용성과 데이터 누수 가능성을 확인하기 전까지 사용하지 않음

## KPI 기준

- NG recall ≥ 0.90
- NG precision ≥ 0.60
- 정상 구간 false alarm rate ≤ 0.05
- F2-score: 미탐을 더 크게 반영하는 보조 지표

이 저장소는 교육용 v1입니다. 


