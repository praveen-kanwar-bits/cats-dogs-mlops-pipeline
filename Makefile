PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

test:
	pytest -v

lint:
	ruff check src scripts tests
	$(PYTHON) -m compileall src scripts tests

preprocess:
	$(PYTHON) scripts/preprocess.py

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

promote:
	$(PYTHON) scripts/promote_model.py

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

docker-build:
	docker build -t cats-dogs-mlops-api:local .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

smoke:
	$(PYTHON) scripts/smoke_test.py

simulate:
	$(PYTHON) scripts/simulate_traffic.py

post-deploy-eval:
	$(PYTHON) scripts/post_deployment_evaluation.py --manifest tests/fixtures/post_deploy_manifest.csv

package:
	$(PYTHON) scripts/create_submission.py
