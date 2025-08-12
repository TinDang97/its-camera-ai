# Makefile for ITS Camera AI
# AI-powered camera traffic monitoring system

.PHONY: help install install-dev install-ml install-gpu install-edge install-all
.PHONY: test test-unit test-integration test-ml test-gpu test-slow test-coverage test-infra
.PHONY: lint format type-check security-check code-quality pre-commit
.PHONY: docker-build docker-build-gpu docker-build-prod docker-build-edge
.PHONY: docker-up docker-down docker-logs docker-shell docker-clean
.PHONY: infra-up infra-down infra-status infra-test infra-logs infra-reset
.PHONY: db-migrate db-seed db-reset db-backup db-restore db-init
.PHONY: deploy-dev deploy-prod monitoring jupyter
.PHONY: clean clean-cache clean-logs clean-models clean-all
.PHONY: docs docs-serve benchmark profile status

# Platform Detection
OS := $(shell uname -s)
ARCH := $(shell uname -m)
ifeq ($(OS),Darwin)
    PLATFORM := macos
    ifeq ($(ARCH),arm64)
        PLATFORM_ARCH := apple-silicon
    else
        PLATFORM_ARCH := intel
    endif
else ifeq ($(OS),Linux)
    PLATFORM := linux
    PLATFORM_ARCH := $(ARCH)
else
    PLATFORM := unknown
    PLATFORM_ARCH := unknown
endif

# Configuration
PYTHON := python3.12
UV := uv
DOCKER := docker
DOCKER_COMPOSE := docker compose
DOCKER_BUILDX := docker buildx
PROJECT_NAME := its-camera-ai
ENVIRONMENT ?= development
BUILD_TARGET ?= development
DOCKER_PLATFORM ?= linux/amd64,linux/arm64
REGISTRY ?= ghcr.io/your-org

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m # No Color

# Platform info target
platform-info:
	@echo "$(CYAN)Platform Information$(NC)"
	@echo "OS: $(YELLOW)$(OS)$(NC)"
	@echo "Architecture: $(YELLOW)$(ARCH)$(NC)"
	@echo "Platform: $(YELLOW)$(PLATFORM)$(NC)"
	@echo "Platform Architecture: $(YELLOW)$(PLATFORM_ARCH)$(NC)"
	@echo "Docker Platform: $(YELLOW)$(DOCKER_PLATFORM)$(NC)"
	@echo ""

# Default help target
help:
	@echo "$(CYAN)ITS Camera AI Development Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Environment Setup:$(NC)"
	@echo "  install         Install core dependencies"
	@echo "  install-dev     Install development dependencies"
	@echo "  install-ml      Install ML/AI dependencies"
	@echo "  install-gpu     Install GPU dependencies"
	@echo "  install-edge    Install edge deployment dependencies"
	@echo "  install-all     Install all dependencies"
	@echo ""
	@echo "$(YELLOW)Code Quality & Testing:$(NC)"
	@echo "  test            Run all tests"
	@echo "  test-unit       Run unit tests only"
	@echo "  test-integration Run integration tests"
	@echo "  test-ml         Run ML model tests"
	@echo "  test-gpu        Run GPU tests (requires CUDA)"
	@echo "  test-slow       Run slow tests"
	@echo "  test-coverage   Generate coverage report"
	@echo "  test-infra      Test infrastructure components"
	@echo "  lint            Run all linters"
	@echo "  format          Format code with black and isort"
	@echo "  type-check      Run mypy type checking"
	@echo "  security-check  Run security scans"
	@echo "  code-quality    Run all code quality checks"
	@echo "  pre-commit      Run pre-commit hooks"
	@echo ""
	@echo "$(YELLOW)Infrastructure Commands:$(NC)"
	@echo "  infra-up        Start all infrastructure services"
	@echo "  infra-down      Stop all infrastructure services"
	@echo "  infra-status    Show infrastructure status"
	@echo "  infra-test      Test infrastructure connectivity"
	@echo "  infra-logs      View infrastructure logs"
	@echo "  infra-reset     Reset all infrastructure data"
	@echo ""
	@echo "$(YELLOW)Platform Commands:$(NC)"
	@echo "  platform-info   Show platform information"
	@echo "  setup-platform  Setup platform-specific optimizations"
	@echo "  optimize-macos  Optimize for macOS (Docker Desktop tuning)"
	@echo "  optimize-linux  Optimize for Linux (kernel parameters)"
	@echo ""
	@echo "$(YELLOW)Docker Commands:$(NC)"
	@echo "  docker-build    Build development Docker image"
	@echo "  docker-build-multiarch Build multi-architecture images"
	@echo "  docker-build-gpu Build GPU Docker image"
	@echo "  docker-build-prod Build production Docker image"
	@echo "  docker-build-edge Build edge Docker image"
	@echo "  docker-up       Start all services"
	@echo "  docker-down     Stop all services"
	@echo "  docker-logs     View service logs"
	@echo "  docker-shell    Open shell in app container"
	@echo "  docker-clean    Clean Docker resources"
	@echo ""
	@echo "$(YELLOW)Inference Optimization:$(NC)"
	@echo "  setup-triton    Setup Triton Inference Server"
	@echo "  setup-tensorrt  Setup TensorRT optimization (Linux/NVIDIA)"
	@echo "  setup-coreml    Setup CoreML optimization (macOS)"
	@echo "  benchmark-inference Benchmark inference performance"
	@echo "  optimize-models Convert models for target platform"
	@echo ""
	@echo "$(YELLOW)Kubernetes Commands:$(NC)"
	@echo "  k8s-setup       Setup local Kubernetes (Kind/k3s)"
	@echo "  k8s-deploy      Deploy to Kubernetes"
	@echo "  k8s-scale       Scale inference services"
	@echo "  k8s-monitoring  Setup monitoring stack"
	@echo ""
	@echo "$(YELLOW)Database Commands:$(NC)"
	@echo "  db-migrate      Run database migrations"
	@echo "  db-seed         Seed database with test data"
	@echo "  db-reset        Reset database"
	@echo "  db-backup       Backup database"
	@echo "  db-restore      Restore database from backup"
	@echo "  db-init         Initialize all databases with schemas"
	@echo ""
	@echo "$(YELLOW)Deployment & Services:$(NC)"
	@echo "  deploy-dev      Deploy to development environment"
	@echo "  deploy-prod     Deploy to production environment"
	@echo "  monitoring      Start monitoring stack (Prometheus/Grafana)"
	@echo "  jupyter         Start Jupyter Lab for ML development"
	@echo ""
	@echo "$(YELLOW)Utilities:$(NC)"
	@echo "  clean           Clean temporary files"
	@echo "  clean-cache     Clean Python cache files"
	@echo "  clean-logs      Clean log files"
	@echo "  clean-models    Clean model files"
	@echo "  clean-all       Clean everything"
	@echo "  docs            Build documentation"
	@echo "  docs-serve      Serve documentation locally"
	@echo "  benchmark       Run performance benchmarks"
	@echo "  profile         Profile application performance"
	@echo "  status          Show project implementation status"

# Environment Setup
install:
	@echo "$(BLUE)Installing core dependencies...$(NC)"
	$(UV) sync --no-dev

install-dev:
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(UV) sync --group dev

install-ml:
	@echo "$(BLUE)Installing ML/AI dependencies...$(NC)"
	$(UV) sync --group ml

install-gpu:
	@echo "$(BLUE)Installing GPU dependencies for $(PLATFORM)...$(NC)"
ifeq ($(PLATFORM),macos)
	$(UV) sync --group gpu --group macos
else ifeq ($(PLATFORM),linux)
	$(UV) sync --group gpu --group linux
else
	$(UV) sync --group gpu
endif

install-edge:
	@echo "$(BLUE)Installing edge deployment dependencies...$(NC)"
	$(UV) sync --group edge

install-all:
	@echo "$(BLUE)Installing all dependencies for $(PLATFORM)...$(NC)"
ifeq ($(PLATFORM),macos)
	$(UV) sync --group dev --group ml --group gpu --group macos --group edge
else ifeq ($(PLATFORM),linux)
	$(UV) sync --group dev --group ml --group gpu --group linux --group edge
else
	$(UV) sync --group dev --group ml --group gpu --group edge
endif

# Code Quality & Testing
test:
	@echo "$(BLUE)Running all tests...$(NC)"
	$(UV) run pytest --cov=src/its_camera_ai --cov-report=html --cov-report=term-missing --cov-fail-under=90

test-unit:
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(UV) run pytest -m "not integration and not slow and not gpu and not ml"

test-integration:
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(UV) run pytest -m integration

test-ml:
	@echo "$(BLUE)Running ML model tests...$(NC)"
	$(UV) run pytest -m ml

test-gpu:
	@echo "$(BLUE)Running GPU tests...$(NC)"
	$(UV) run pytest -m gpu

test-slow:
	@echo "$(BLUE)Running slow tests...$(NC)"
	$(UV) run pytest -m slow

test-coverage:
	@echo "$(BLUE)Generating coverage report...$(NC)"
	$(UV) run pytest --cov=src/its_camera_ai --cov-report=html --cov-report=xml --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

test-infra:
	@echo "$(BLUE)Testing infrastructure components...$(NC)"
	$(UV) run python test_infrastructure.py
	@echo "$(GREEN)Infrastructure tests completed!$(NC)"

lint:
	@echo "$(BLUE)Running linters...$(NC)"
	$(UV) run ruff check src/ tests/
	$(UV) run ruff format --check src/ tests/

format:
	@echo "$(BLUE)Formatting code...$(NC)"
	$(UV) run black src/ tests/
	$(UV) run isort src/ tests/
	$(UV) run ruff format src/ tests/

type-check:
	@echo "$(BLUE)Running type checks...$(NC)"
	$(UV) run mypy src/

security-check:
	@echo "$(BLUE)Running security scans...$(NC)"
	$(UV) run bandit -r src/
	$(UV) run safety check
	$(UV) run pip-audit

code-quality: lint type-check security-check
	@echo "$(GREEN)All code quality checks passed!$(NC)"

pre-commit:
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	$(UV) run pre-commit run --all-files

# Platform Setup Commands
setup-platform: platform-info
ifeq ($(PLATFORM),macos)
	$(MAKE) optimize-macos
else ifeq ($(PLATFORM),linux)
	$(MAKE) optimize-linux
endif
	@echo "$(GREEN)Platform setup completed!$(NC)"

optimize-macos:
	@echo "$(BLUE)Optimizing for macOS...$(NC)"
	@echo "Configuring Docker Desktop resource limits..."
	@if command -v docker >/dev/null 2>&1; then \
		echo "Docker Desktop detected. Consider increasing:"; \
		echo "- Memory: 8GB+ for ML workloads"; \
		echo "- CPUs: 4+ cores"; \
		echo "- Disk: 64GB+ for models"; \
	fi

optimize-linux:
	@echo "$(BLUE)Optimizing for Linux...$(NC)"
	@echo "Checking system optimizations..."
	@if [ -f /proc/sys/vm/swappiness ]; then \
		echo "Current swappiness: $$(cat /proc/sys/vm/swappiness)"; \
		echo "Recommended for ML: 10"; \
	fi
	@if [ -f /proc/sys/net/core/somaxconn ]; then \
		echo "Current somaxconn: $$(cat /proc/sys/net/core/somaxconn)"; \
		echo "Recommended for high-load: 1024"; \
	fi

# Docker Commands
docker-build:
	@echo "$(BLUE)Building development Docker image for $(PLATFORM)...$(NC)"
ifeq ($(PLATFORM_ARCH),apple-silicon)
	$(DOCKER) build --platform linux/arm64 --target development -t $(PROJECT_NAME):dev .
else
	$(DOCKER) build --target development -t $(PROJECT_NAME):dev .
endif

docker-build-multiarch:
	@echo "$(BLUE)Building multi-architecture Docker images...$(NC)"
	$(DOCKER_BUILDX) create --use --name multiarch-builder --driver docker-container || true
	$(DOCKER_BUILDX) build --platform $(DOCKER_PLATFORM) --target development -t $(REGISTRY)/$(PROJECT_NAME):dev --push .
	$(DOCKER_BUILDX) build --platform $(DOCKER_PLATFORM) --target production -t $(REGISTRY)/$(PROJECT_NAME):latest --push .
	$(DOCKER_BUILDX) build --platform $(DOCKER_PLATFORM) --target edge -t $(REGISTRY)/$(PROJECT_NAME):edge --push .

docker-build-gpu:
	@echo "$(BLUE)Building GPU Docker image...$(NC)"
ifeq ($(PLATFORM),linux)
	$(DOCKER) build --target gpu-development -t $(PROJECT_NAME):gpu .
else
	@echo "$(YELLOW)GPU support primarily for Linux. Building CPU version...$(NC)"
	$(DOCKER) build --target development -t $(PROJECT_NAME):gpu .
endif

docker-build-prod:
	@echo "$(BLUE)Building production Docker image...$(NC)"
	$(DOCKER) build --target production -t $(PROJECT_NAME):prod .

docker-build-edge:
	@echo "$(BLUE)Building edge Docker image...$(NC)"
	$(DOCKER) build --target edge -t $(PROJECT_NAME):edge .

docker-up:
	@echo "$(BLUE)Starting all services...$(NC)"
	ENVIRONMENT=$(ENVIRONMENT) BUILD_TARGET=$(BUILD_TARGET) $(DOCKER_COMPOSE) --profile dev up -d

docker-up-gpu:
	@echo "$(BLUE)Starting services with GPU support...$(NC)"
	ENVIRONMENT=$(ENVIRONMENT) BUILD_TARGET=gpu-development $(DOCKER_COMPOSE) --profile gpu up -d

docker-up-prod:
	@echo "$(BLUE)Starting production services...$(NC)"
	ENVIRONMENT=production BUILD_TARGET=production $(DOCKER_COMPOSE) --profile prod up -d

docker-down:
	@echo "$(BLUE)Stopping all services...$(NC)"
	$(DOCKER_COMPOSE) down

docker-logs:
	@echo "$(BLUE)Viewing service logs...$(NC)"
	$(DOCKER_COMPOSE) logs -f

docker-shell:
	@echo "$(BLUE)Opening shell in app container...$(NC)"
	$(DOCKER_COMPOSE) exec app bash || $(DOCKER) run -it --rm -v $(PWD):/app $(PROJECT_NAME):dev bash

docker-clean:
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	$(DOCKER_COMPOSE) down -v --remove-orphans
	$(DOCKER) system prune -af

# Infrastructure Commands
infra-up:
	@echo "$(BLUE)Starting infrastructure services...$(NC)"
	@if [ -f docker-compose.complete.yml ]; then \
		$(DOCKER_COMPOSE) -f docker-compose.complete.yml up -d postgres timescaledb redis zookeeper kafka minio prometheus grafana; \
	elif [ -f start-infrastructure.sh ]; then \
		./start-infrastructure.sh; \
	else \
		@echo "$(RED)No infrastructure configuration found!$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Infrastructure services started!$(NC)"
	@echo "Run 'make infra-status' to check service status"

infra-down:
	@echo "$(BLUE)Stopping infrastructure services...$(NC)"
	@if [ -f docker-compose.complete.yml ]; then \
		$(DOCKER_COMPOSE) -f docker-compose.complete.yml down; \
	else \
		$(DOCKER) stop $$($(DOCKER) ps -q --filter name=its-) 2>/dev/null || true; \
	fi
	@echo "$(GREEN)Infrastructure services stopped!$(NC)"

infra-status:
	@echo "$(CYAN)Infrastructure Services Status$(NC)"
	@echo "========================================"
	@$(DOCKER) ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "its-|NAME" || echo "No infrastructure services running"
	@echo ""
	@echo "$(CYAN)Service Access Points:$(NC)"
	@echo "PostgreSQL:    localhost:5432"
	@echo "TimescaleDB:   localhost:5433"
	@echo "Redis:         localhost:6379"
	@echo "Kafka:         localhost:9092"
	@echo "MinIO Console: http://localhost:9001"
	@echo "Prometheus:    http://localhost:9090"
	@echo "Grafana:       http://localhost:3000"

infra-test: test-infra

infra-logs:
	@echo "$(BLUE)Viewing infrastructure logs...$(NC)"
	@if [ -f docker-compose.complete.yml ]; then \
		$(DOCKER_COMPOSE) -f docker-compose.complete.yml logs -f --tail=100; \
	else \
		$(DOCKER) logs -f --tail=100 $$($(DOCKER) ps -q --filter name=its- | head -1); \
	fi

infra-reset:
	@echo "$(YELLOW)WARNING: This will delete all infrastructure data!$(NC)"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ]; then \
		$(MAKE) infra-down; \
		$(DOCKER) rm -f $$($(DOCKER) ps -aq --filter name=its-) 2>/dev/null || true; \
		$(DOCKER) volume rm $$($(DOCKER) volume ls -q | grep its) 2>/dev/null || true; \
		@echo "$(GREEN)Infrastructure reset completed!$(NC)"; \
	else \
		@echo "$(YELLOW)Reset cancelled$(NC)"; \
	fi

# Database Commands
db-migrate:
	@echo "$(BLUE)Running database migrations...$(NC)"
	$(UV) run alembic upgrade head

db-seed:
	@echo "$(BLUE)Seeding database with test data...$(NC)"
	$(UV) run python scripts/seed_database.py

db-reset:
	@echo "$(BLUE)Resetting database...$(NC)"
	$(DOCKER_COMPOSE) exec postgres dropdb -U its_user its_camera_ai || true
	$(DOCKER_COMPOSE) exec postgres createdb -U its_user its_camera_ai
	$(MAKE) db-migrate
	$(MAKE) db-seed

db-backup:
	@echo "$(BLUE)Backing up database...$(NC)"
	mkdir -p backups
	$(DOCKER_COMPOSE) exec postgres pg_dump -U its_user its_camera_ai > backups/db-backup-$(shell date +%Y%m%d-%H%M%S).sql

db-restore:
	@echo "$(BLUE)Restoring database...$(NC)"
	@read -p "Enter backup file path: " backup_file; \
	$(DOCKER_COMPOSE) exec -T postgres psql -U its_user its_camera_ai < $$backup_file

db-init:
	@echo "$(BLUE)Initializing all databases with schemas...$(NC)"
	@if [ -f infrastructure/database/postgresql-init.sql ]; then \
		$(DOCKER) exec -i its-postgres psql -U postgres its_camera_ai < infrastructure/database/postgresql-init.sql 2>/dev/null || \
		echo "$(YELLOW)PostgreSQL schema already initialized or file not found$(NC)"; \
	fi
	@if [ -f infrastructure/database/timescale-init.sql ]; then \
		$(DOCKER) exec -i its-timescaledb psql -U postgres its_metrics < infrastructure/database/timescale-init.sql 2>/dev/null || \
		echo "$(YELLOW)TimescaleDB schema already initialized or file not found$(NC)"; \
	fi
	@echo "$(GREEN)Database initialization completed!$(NC)"

# Deployment & Services
deploy-dev: docker-build docker-up
	@echo "$(GREEN)Development environment deployed!$(NC)"
	@echo "$(CYAN)Application: http://localhost:8000$(NC)"
	@echo "$(CYAN)API Docs: http://localhost:8000/docs$(NC)"

deploy-prod: docker-build-prod
	@echo "$(BLUE)Deploying to production...$(NC)"
	ENVIRONMENT=production BUILD_TARGET=production $(DOCKER_COMPOSE) --profile prod up -d
	@echo "$(GREEN)Production environment deployed!$(NC)"

monitoring:
	@echo "$(BLUE)Starting monitoring stack...$(NC)"
	$(DOCKER_COMPOSE) --profile monitoring up -d
	@echo "$(GREEN)Monitoring stack started!$(NC)"
	@echo "$(CYAN)Prometheus: http://localhost:9090$(NC)"
	@echo "$(CYAN)Grafana: http://localhost:3000 (admin/admin123)$(NC)"

jupyter:
	@echo "$(BLUE)Starting Jupyter Lab...$(NC)"
	$(DOCKER_COMPOSE) --profile ml up -d jupyter
	@echo "$(GREEN)Jupyter Lab started!$(NC)"
	@echo "$(CYAN)Access: http://localhost:8888$(NC)"
	@echo "$(CYAN)Token: jupyter-token-12345$(NC)"

# Utilities
clean:
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

clean-cache: clean
	@echo "$(BLUE)Cleaning cache files...$(NC)"
	rm -rf .coverage htmlcov/
	rm -rf .pytest_cache .mypy_cache .ruff_cache

clean-logs:
	@echo "$(BLUE)Cleaning log files...$(NC)"
	rm -rf logs/*.log
	rm -rf logs/*.log.*

clean-models:
	@echo "$(BLUE)Cleaning model files...$(NC)"
	rm -rf models/*.pt models/*.onnx models/*.trt

clean-all: clean-cache clean-logs clean-models docker-clean
	@echo "$(GREEN)All cleaned!$(NC)"

docs:
	@echo "$(BLUE)Building documentation...$(NC)"
	$(UV) run sphinx-build -b html docs/ docs/_build/html

docs-serve: docs
	@echo "$(BLUE)Serving documentation...$(NC)"
	$(UV) run python -m http.server 8080 -d docs/_build/html

benchmark:
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	$(UV) run pytest --benchmark-only --benchmark-sort=mean

profile:
	@echo "$(BLUE)Profiling application performance...$(NC)"
	$(UV) run python -m cProfile -o profile.stats main.py
	$(UV) run python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Inference Optimization Commands
setup-triton:
	@echo "$(BLUE)Setting up Triton Inference Server...$(NC)"
	$(DOCKER) pull nvcr.io/nvidia/tritonserver:24.02-py3
	@echo "$(GREEN)Triton Inference Server ready!$(NC)"

setup-tensorrt:
	@echo "$(BLUE)Setting up TensorRT optimization...$(NC)"
ifeq ($(PLATFORM),linux)
	@echo "Installing TensorRT for Linux..."
	$(UV) sync --group tensorrt
	@echo "$(GREEN)TensorRT setup completed!$(NC)"
else
	@echo "$(YELLOW)TensorRT is only available on Linux/NVIDIA platforms$(NC)"
endif

setup-coreml:
	@echo "$(BLUE)Setting up CoreML optimization...$(NC)"
ifeq ($(PLATFORM),macos)
	@echo "Installing CoreML tools..."
	$(UV) sync --group coreml
	@echo "$(GREEN)CoreML setup completed!$(NC)"
else
	@echo "$(YELLOW)CoreML is only available on macOS platforms$(NC)"
endif

benchmark-inference:
	@echo "$(BLUE)Running inference benchmarks for $(PLATFORM)...$(NC)"
	$(UV) run python scripts/benchmark_inference.py --platform $(PLATFORM) --arch $(PLATFORM_ARCH)

optimize-models:
	@echo "$(BLUE)Optimizing models for $(PLATFORM)...$(NC)"
ifeq ($(PLATFORM),macos)
	$(UV) run python scripts/optimize_models.py --target coreml --arch $(PLATFORM_ARCH)
else ifeq ($(PLATFORM),linux)
	$(UV) run python scripts/optimize_models.py --target tensorrt onnx --arch $(PLATFORM_ARCH)
else
	$(UV) run python scripts/optimize_models.py --target onnx --arch $(PLATFORM_ARCH)
endif

# Kubernetes Commands
k8s-setup:
	@echo "$(BLUE)Setting up local Kubernetes for $(PLATFORM)...$(NC)"
ifeq ($(PLATFORM),macos)
	@echo "Setting up Kind on macOS..."
	brew install kind kubectl helm || true
	kind create cluster --config infrastructure/kubernetes/kind-config-$(PLATFORM).yaml --name $(PROJECT_NAME)
else ifeq ($(PLATFORM),linux)
	@echo "Setting up k3s on Linux..."
	curl -sfL https://get.k3s.io | sh - || true
	mkdir -p ~/.kube && sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config && sudo chown $(shell id -u):$(shell id -g) ~/.kube/config
endif
	@echo "$(GREEN)Kubernetes cluster ready!$(NC)"

k8s-deploy:
	@echo "$(BLUE)Deploying to Kubernetes...$(NC)"
	helm upgrade --install $(PROJECT_NAME) infrastructure/helm/its-camera-ai \
		--set platform=$(PLATFORM) \
		--set architecture=$(PLATFORM_ARCH) \
		--set image.repository=$(REGISTRY)/$(PROJECT_NAME) \
		--set image.tag=latest

k8s-scale:
	@echo "$(BLUE)Scaling inference services...$(NC)"
	kubectl scale deployment $(PROJECT_NAME)-inference --replicas=3
	kubectl get hpa

k8s-monitoring:
	@echo "$(BLUE)Setting up Kubernetes monitoring...$(NC)"
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts || true
	helm repo add grafana https://grafana.github.io/helm-charts || true
	helm repo update
	helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
		--set grafana.adminPassword=admin123 \
		--namespace monitoring --create-namespace

# Development workflow shortcuts
dev: install-dev docker-up
	@echo "$(GREEN)Development environment ready!$(NC)"
	@echo "$(CYAN)Run 'make test' to run tests$(NC)"
	@echo "$(CYAN)Run 'make format' to format code$(NC)"
	@echo "$(CYAN)Run 'make jupyter' for ML development$(NC)"

gpu-dev: install-all docker-up-gpu
	@echo "$(GREEN)GPU development environment ready!$(NC)"

ci: install-dev code-quality test
	@echo "$(GREEN)CI checks passed!$(NC)"

# Check if required tools are installed
check-tools:
	@command -v $(UV) >/dev/null 2>&1 || { echo "$(RED)uv is required but not installed. Please install it first.$(NC)" >&2; exit 1; }
	@command -v $(DOCKER) >/dev/null 2>&1 || { echo "$(RED)Docker is required but not installed. Please install it first.$(NC)" >&2; exit 1; }
	@command -v $(DOCKER_COMPOSE) >/dev/null 2>&1 || { echo "$(RED)Docker Compose is required but not installed. Please install it first.$(NC)" >&2; exit 1; }

# Project status command
status:
	@echo "$(CYAN)ITS Camera AI - Implementation Status$(NC)"
	@echo "========================================"
	@echo ""
	@echo "$(YELLOW)Infrastructure Status:$(NC)"
	@$(DOCKER) ps --format "table {{.Names}}\t{{.Status}}" | grep -E "its-|NAME" | head -10 || echo "No services running"
	@echo ""
	@echo "$(YELLOW)Completed Components:$(NC)"
	@echo "✅ PostgreSQL Database Service"
	@echo "✅ TimescaleDB Time-Series Service"
	@echo "✅ Redis Cache/Queue Service"
	@echo "✅ Kafka Message Broker"
	@echo "✅ MinIO Object Storage"
	@echo "✅ Prometheus Monitoring"
	@echo "✅ Grafana Dashboards"
	@echo "✅ Infrastructure Testing Suite"
	@echo "✅ Docker Compose Configurations"
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "1. Start infrastructure: $(GREEN)make infra-up$(NC)"
	@echo "2. Check status: $(GREEN)make infra-status$(NC)"
	@echo "3. Run tests: $(GREEN)make test-infra$(NC)"
	@echo "4. View logs: $(GREEN)make infra-logs$(NC)"
	@echo ""
	@echo "For detailed status, see: docs/INFRASTRUCTURE_STATUS.md"

# Auto-run check-tools before main targets
test lint format type-check security-check docker-build docker-up: check-tools
