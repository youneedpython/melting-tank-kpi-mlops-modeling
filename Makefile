.PHONY: install test lint train smoke mlflow

install:
	python -m pip install -r requirements-dev.txt
	python -m pip install -e .

test:
	pytest -q

lint:
	ruff check .

train:
	python -m melting_tank.train

smoke:
	python -m melting_tank.train --epochs 1 --sample-minutes 1000 --run-name smoke-v2

mlflow:
	docker compose up -d mlflow

