.PHONY: test test-cov lint format clean install help

# Default target
help:
	@echo "Available commands:"
	@echo "  test      - Run tests"
	@echo "  test-cov  - Run tests with coverage"
	@echo "  lint      - Run linting (flake8, mypy)"
	@echo "  format    - Format code (black, isort)"
	@echo "  check     - Run all checks (test + lint)"
	@echo "  install   - Install development dependencies"
	@echo "  clean     - Clean cache files"

# Run tests
test:
	/data/lyc/conda_envs/ms-swift/bin/python -m pytest tests/ -v

# Run tests with coverage
test-cov:
	/data/lyc/conda_envs/ms-swift/bin/python -m pytest tests/ --cov=. --cov-report=html --cov-report=term

# Run linting
lint:
	/data/lyc/conda_envs/ms-swift/bin/flake8 *.py tests/ reward.py dataset.py
	/data/lyc/conda_envs/ms-swift/bin/mypy reward.py dataset.py --ignore-missing-imports

# Format code
format:
	/data/lyc/conda_envs/ms-swift/bin/black *.py tests/ reward.py dataset.py
	/data/lyc/conda_envs/ms-swift/bin/isort *.py tests/ reward.py dataset.py

# Run all checks
check: test lint

# Install development dependencies
install:
	/data/lyc/conda_envs/ms-swift/bin/pip install -r requirements.txt

# Clean cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
