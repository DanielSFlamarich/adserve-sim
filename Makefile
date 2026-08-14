.PHONY: install lint typecheck test check

install:
	uv sync --all-extras --dev
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src/

test:
	uv run pytest

check: lint typecheck test
