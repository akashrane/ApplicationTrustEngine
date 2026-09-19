.PHONY: install data pipeline app all

PYTHON ?= .venv/bin/python

install:
	test -x .venv/bin/python || python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) generate.py

pipeline:
	$(PYTHON) integrity.py
	$(PYTHON) population.py
	$(PYTHON) provenance.py
	$(PYTHON) score.py

app:
	$(PYTHON) -m streamlit run app.py

all: data pipeline
