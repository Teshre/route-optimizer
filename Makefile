# Route Optimizer — developer tasks
# Usage: `make help` to list targets.

# Use the venv's interpreters when present, fall back to system Python otherwise.
VENV        ?= .venv
PY          := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
STREAMLIT   := $(VENV)/bin/streamlit

.DEFAULT_GOAL := help
.PHONY: help setup data app analysis clean

help: ## Show this help
	@echo "Route Optimizer — available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quickstart:  make setup && make data && make app"

setup: ## Create a virtualenv and install dependencies
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Environment ready. Next: make data && make app"

data: ## Generate the synthetic client dataset (data/clientes.csv)
	$(PY) data/generate_data.py

app: ## Launch the interactive Streamlit dashboard
	$(STREAMLIT) run app.py

analysis: ## Run the standalone CLI analysis (KPIs, charts, map)
	$(PY) route_optimizer.py

clean: ## Remove generated output and Python caches
	rm -rf resultados out
	find . -type d -name '__pycache__' -not -path './$(VENV)/*' -exec rm -rf {} +
	@echo "✓ Cleaned generated output and caches"
