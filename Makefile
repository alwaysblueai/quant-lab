.PHONY: lint typecheck test check format clean clean-outputs

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

clean-outputs:
	rm -rf -- outputs/real_cases/_web_runs outputs/model_lab_tutorial
	find outputs/real_cases -maxdepth 1 -type d \( -name "*smoke*" -o -name "*labdemo*" \) -exec rm -rf {} + 2>/dev/null || true
