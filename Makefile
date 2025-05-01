PYTHON := python3
VENV_DIR := .venv
PIP := $(VENV_DIR)/bin/pip
PYTHON_EXEC := $(VENV_DIR)/bin/python

.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  make venv        Create virtual environment"
	@echo "  make install     Install dependencies into virtual environment"
	@echo "  make run         Run the main forecasting script"
	@echo "  make clean       Remove virtual environment and __pycache__"

$(VENV_DIR)/bin/activate:
	$(PYTHON) -m venv $(VENV_DIR)

venv: $(VENV_DIR)/bin/activate
	@echo "Virtual environment created in $(VENV_DIR)/"

install: venv
	@echo "Installing dependencies from pyproject.toml..."
	$(PIP) install --upgrade pip
	$(PIP) install . # Installs project defined in pyproject.toml and its dependencies
	@echo "Installation complete."

run: venv
	@echo "Running the forecasting script (ai_model.py)..."
	$(PYTHON_EXEC) ai_model.py
	@echo "Script finished."

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -f *.png
	@echo "Cleanup complete."

.PHONY: help venv install run clean