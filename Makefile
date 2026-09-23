.PHONY: setup lint test up down ingest agent replay eval-smoke eval-full ablation report

PROFILE ?= naive
MODE ?= simulated

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy polyglot/core

test:
	uv run pytest

up:
	docker compose up -d

down:
	docker compose down

ingest:
	uv run python -m polyglot.retrieval.ingest

agent:
	@echo "make agent: not implemented until M4"

replay:
	uv run python -m eval.replay --scenario $(SCENARIO) --profile $(PROFILE) --mode $(MODE)

eval-smoke:
	@echo "make eval-smoke: not implemented until M7"

eval-full:
	@echo "make eval-full: not implemented until M7"

ablation:
	@echo "make ablation: not implemented until M5"

report:
	@echo "make report: not implemented until M7"
