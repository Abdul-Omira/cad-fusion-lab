.PHONY: help install install-dev test lint format clean docker-build docker-run docs

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
FLAKE8 := $(PYTHON) -m flake8
MYPY := $(PYTHON) -m mypy
DOCKER_IMAGE := cad-fusion-lab
DOCKER_TAG := latest

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo '$(BLUE)CAD Fusion Lab - Development Commands$(NC)'
	@echo ''
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ''

##@ Installation

install: ## Install production dependencies
	@echo '$(BLUE)Installing production dependencies...$(NC)'
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .

install-dev: ## Install development dependencies
	@echo '$(BLUE)Installing development dependencies...$(NC)'
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"
	$(PIP) install -e ".[data-collection]"
	$(PIP) install -e ".[monitoring]"
	pre-commit install

install-all: ## Install all dependencies including optional ones
	@echo '$(BLUE)Installing all dependencies...$(NC)'
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[all]"
	pre-commit install

##@ Development

format: ## Format code with black and isort
	@echo '$(BLUE)Formatting code...$(NC)'
	$(BLACK) src scripts tests
	$(ISORT) src scripts tests
	@echo '$(GREEN)Code formatting complete!$(NC)'

lint: ## Run linters (flake8, mypy, bandit)
	@echo '$(BLUE)Running linters...$(NC)'
	$(FLAKE8) src scripts tests
	$(MYPY) src --ignore-missing-imports || true
	bandit -r src scripts || true
	@echo '$(GREEN)Linting complete!$(NC)'

test: ## Run tests with pytest
	@echo '$(BLUE)Running tests...$(NC)'
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo '$(GREEN)Tests complete! Coverage report in htmlcov/index.html$(NC)'

test-fast: ## Run tests without coverage
	@echo '$(BLUE)Running fast tests...$(NC)'
	$(PYTEST) tests/ -v -x
	@echo '$(GREEN)Tests complete!$(NC)'

test-integration: ## Run integration tests
	@echo '$(BLUE)Running integration tests...$(NC)'
	$(PYTEST) tests/ -v -m integration
	@echo '$(GREEN)Integration tests complete!$(NC)'

pre-commit: ## Run pre-commit hooks on all files
	@echo '$(BLUE)Running pre-commit hooks...$(NC)'
	pre-commit run --all-files

##@ Training and Evaluation

train-small: ## Train small model for development
	@echo '$(BLUE)Training small model...$(NC)'
	$(PYTHON) scripts/train.py --config configs/small_config.yaml --output-dir checkpoints/small

train-base: ## Train base model
	@echo '$(BLUE)Training base model...$(NC)'
	$(PYTHON) scripts/train.py --config configs/base_config.yaml --output-dir checkpoints/base

train-large: ## Train large model
	@echo '$(BLUE)Training large model...$(NC)'
	$(PYTHON) scripts/train.py --config configs/large_config.yaml --output-dir checkpoints/large

evaluate: ## Evaluate model
	@echo '$(BLUE)Evaluating model...$(NC)'
	$(PYTHON) scripts/evaluate.py --model checkpoints/final_model.pt --output-dir outputs/evaluation

##@ Data Processing

prepare-data: ## Prepare dataset
	@echo '$(BLUE)Preparing dataset...$(NC)'
	$(PYTHON) scripts/prepare_dataset.py --output-dir data/processed --num-samples 1000

generate-sample: ## Generate a sample CAD model
	@echo '$(BLUE)Generating sample CAD model...$(NC)'
	$(PYTHON) scripts/inference.py \
		--model checkpoints/final_model.pt \
		--text "Create a rectangular bracket with mounting holes" \
		--output outputs/sample.step

##@ Docker

docker-build: ## Build Docker image
	@echo '$(BLUE)Building Docker image...$(NC)'
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo '$(GREEN)Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)'

docker-run: ## Run Docker container
	@echo '$(BLUE)Running Docker container...$(NC)'
	docker run -p 8000:8000 -v $(PWD)/outputs:/app/outputs $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-compose-up: ## Start all services with docker-compose
	@echo '$(BLUE)Starting services with docker-compose...$(NC)'
	docker-compose up -d
	@echo '$(GREEN)Services started! API: http://localhost:8000, Grafana: http://localhost:3000$(NC)'

docker-compose-down: ## Stop all docker-compose services
	@echo '$(BLUE)Stopping docker-compose services...$(NC)'
	docker-compose down

docker-compose-logs: ## View docker-compose logs
	docker-compose logs -f

##@ Deployment

serve: ## Run development server
	@echo '$(BLUE)Starting development server...$(NC)'
	$(PYTHON) src/deployment/server.py --reload

serve-secure: ## Run secure server with authentication
	@echo '$(BLUE)Starting secure server...$(NC)'
	$(PYTHON) src/deployment/server_secure.py

##@ Cleaning

clean: ## Clean build artifacts and caches
	@echo '$(BLUE)Cleaning build artifacts...$(NC)'
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.mypy_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	find . -type f -name '*.pyo' -delete 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage
	@echo '$(GREEN)Clean complete!$(NC)'

clean-data: ## Clean data directories
	@echo '$(YELLOW)Warning: This will delete all processed data!$(NC)'
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/processed data/raw outputs; \
		echo '$(GREEN)Data cleaned!$(NC)'; \
	fi

##@ Git

git-setup: ## Set up git hooks and configuration
	@echo '$(BLUE)Setting up git hooks...$(NC)'
	pre-commit install
	git config --local commit.gpgsign false
	@echo '$(GREEN)Git setup complete!$(NC)'

##@ Release

bump-patch: ## Bump patch version (0.0.X)
	@echo '$(BLUE)Bumping patch version...$(NC)'
	bump2version patch

bump-minor: ## Bump minor version (0.X.0)
	@echo '$(BLUE)Bumping minor version...$(NC)'
	bump2version minor

bump-major: ## Bump major version (X.0.0)
	@echo '$(BLUE)Bumping major version...$(NC)'
	bump2version major

##@ Other

check: format lint test ## Run all quality checks (format, lint, test)
	@echo '$(GREEN)All checks passed!$(NC)'

init-dev: install-dev git-setup ## Initialize development environment
	@echo '$(GREEN)Development environment initialized!$(NC)'
	@echo ''
	@echo 'Next steps:'
	@echo '  1. Copy .env.example to .env and configure'
	@echo '  2. Run: make prepare-data'
	@echo '  3. Run: make train-small'
	@echo '  4. Run: make serve'

.DEFAULT_GOAL := help
