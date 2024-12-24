# Variables
PYTHON = python
PIP = pip
CODE_DIRS = models scripts data/*.py

# Default target
.PHONY: all
all: install test lint format

# Install dependencies
.PHONY: install
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Lint and format check using Black
.PHONY: lint
lint:
	$(PYTHON) -m black --check $(CODE_DIRS)

# Format the code using Black
.PHONY: format
format:
	$(PYTHON) -m black $(CODE_DIRS)

# Clean up
.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Help
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make install  - Set up the virtual environment and install dependencies"
	@echo "  make lint     - Check code style using Black"
	@echo "  make format   - Format code using Black"
	@echo "  make clean    - Clean up generated files"
