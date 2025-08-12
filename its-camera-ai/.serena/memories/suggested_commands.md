# ITS Camera AI - Essential Development Commands

## Environment Setup
```bash
# Install core dependencies
uv sync

# Install development dependencies  
uv sync --group dev

# Install ML/AI dependencies
uv sync --group ml

# Install GPU dependencies (platform-specific)
uv sync --group ml --group gpu

# Install all dependencies
make install-all
```

## Development Workflow
```bash
# Start development environment
make dev

# Start development with GPU support
make gpu-dev

# Run the main application
python main.py

# Run FastAPI development server
uvicorn its_camera_ai.app.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Testing
```bash
# Run all tests with coverage (must maintain >90% coverage)
pytest --cov=src/its_camera_ai --cov-report=html --cov-report=term-missing --cov-fail-under=90

# Run specific test types
pytest -m "not slow"          # Skip slow tests
pytest -m integration         # Integration tests only
pytest -m ml                  # ML model tests only
pytest -m gpu                 # GPU-dependent tests

# Quick testing commands
make test              # All tests
make test-unit         # Unit tests only
make test-integration  # Integration tests
make test-ml          # ML model tests
make test-gpu         # GPU tests
```

## Code Quality (Required before commits)
```bash
# Format code
make format
black src/ tests/
isort src/ tests/
ruff format src/ tests/

# Lint code
make lint
ruff check src/ tests/

# Type checking
make type-check
mypy src/

# Security checks
make security-check
bandit -r src/
safety check
pip-audit

# All code quality checks
make code-quality

# Pre-commit hooks
make pre-commit
```

## Docker Operations
```bash
# Build and start development environment
make docker-build
make docker-up

# Build production images
make docker-build-prod

# Start services
make docker-up         # Development
make docker-up-gpu     # With GPU support
make docker-up-prod    # Production

# View logs and debugging
make docker-logs
make docker-shell

# Clean up
make docker-clean
```

## Database Operations
```bash
# Run migrations
make db-migrate

# Seed test data
make db-seed

# Reset database
make db-reset

# Backup/restore
make db-backup
make db-restore
```

## Deployment
```bash
# Deploy development environment
make deploy-dev

# Deploy production environment  
make deploy-prod

# Start monitoring stack
make monitoring

# Start Jupyter Lab for ML development
make jupyter
```

## Utilities
```bash
# Clean temporary files
make clean
make clean-all

# Build documentation
make docs
make docs-serve

# Performance analysis
make benchmark
make profile

# Platform optimization
make platform-info
make setup-platform
```

## Essential Daily Commands
1. `uv sync --group dev` - Update dependencies
2. `make format` - Format code before committing
3. `make test` - Run tests (required >90% coverage)
4. `make code-quality` - Run all quality checks
5. `make docker-up` - Start local environment