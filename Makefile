# Reproducible entry points. Run `make help` to see what is available.
.PHONY: help setup verify clean-pyc

PYTHON := python3
VENV   := .venv

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtual environment and install dependencies
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	@echo "Done. Activate with: source $(VENV)/bin/activate"

verify: ## Smoke-test that config and logging load correctly
	$(VENV)/bin/python scripts/verify_setup.py

clean-pyc: ## Remove Python bytecode caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
