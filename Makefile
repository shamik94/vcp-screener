.PHONY: screen trend heuristic calibrate test install ui venv

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit
PYTEST := $(VENV)/bin/pytest

venv:
	test -d $(VENV) || python3 -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt

screen:
	$(PY) -m src.screener.run --countries us

trend:
	$(PY) -m src.screener.run --countries us --mode trend

heuristic:
	$(PY) -m src.screener.run --countries us --mode heuristic

calibrate:
	$(PY) -m src.screener.calibrate

ui:
	$(STREAMLIT) run src/screener/ui.py

test:
	$(PYTEST) -q
