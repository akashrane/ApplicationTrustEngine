.PHONY: install data pipeline app all

PYTHON ?= python3

install:
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

