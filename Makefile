# The repo path may contain characters that break `source .venv/bin/activate`
# (a colon ends up in PATH), so every target calls the venv binaries directly.
VENV ?= .venv
PY := $(VENV)/bin/python
TASKS ?= evals/data/tasks.sample.jsonl
PROVIDER ?= stub

.PHONY: install run mcp test lint fmt eval gate baseline calibrate migrate up down clean

install:
	@if command -v uv >/dev/null 2>&1; then \
		(test -x $(PY) || uv venv $(VENV)) && uv pip install --python $(PY) -e ".[dev]"; \
	else \
		(test -x $(PY) || python3 -m venv $(VENV)) && $(PY) -m pip install -e ".[dev]"; \
	fi

run:
	$(PY) -m uvicorn relay.api.app:create_app --factory --reload --port 8000

mcp:
	$(PY) -m relay.mcp_server

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

eval:
	$(PY) -m evals.run --tasks $(TASKS) --provider $(PROVIDER) --out reports

gate:
	$(PY) -m evals.gate --report reports/latest.json --baseline evals/baseline.json --config evals/gate.yaml

baseline: eval
	$(PY) -m evals.gate --report reports/latest.json --write-baseline evals/baseline.json

calibrate:
	$(PY) -m evals.calibrate_judge --labels evals/data/judge_labels.jsonl --provider $(PROVIDER)

migrate:
	$(PY) -m relay.db.migrate

up:
	docker compose up -d postgres opensearch

down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache reports/*.json reports/*.md
