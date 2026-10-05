install:
	pip install -r requirements.txt

lint:
	python -m compileall src

test:
	pytest -q

run:
	python scripts/run_pipeline.py --config configs/default.yaml --mode smoke
