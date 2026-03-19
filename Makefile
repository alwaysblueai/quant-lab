.PHONY: lint typecheck test check format clean

UV_RUN = uv run --no-sync --frozen

lint:
	$(UV_RUN) ruff check .

format:
	$(UV_RUN) ruff format .
	$(UV_RUN) ruff check . --fix

typecheck:
	$(UV_RUN) mypy src

test:
	$(UV_RUN) pytest

check:
	$(UV_RUN) ruff check .
	$(UV_RUN) mypy src
	$(UV_RUN) pytest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
