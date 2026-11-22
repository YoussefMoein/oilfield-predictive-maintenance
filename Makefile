.PHONY: install all test notebook clean

install:
	pip install -r requirements.txt

all:            ## data -> validation -> features -> models -> evaluation -> risk ranking -> replication
	python scripts/run_pipeline.py

test:
	python -m pytest -q

notebook:       ## rebuild and execute the narrative notebook
	python scripts/build_notebook.py

clean:
	rm -f data/processed/*.parquet models/*.joblib
