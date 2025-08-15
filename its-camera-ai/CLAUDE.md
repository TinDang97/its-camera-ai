# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **ITS Camera AI** project - an AI-powered camera traffic monitoring system for real-time traffic analytics and vehicle tracking. The system uses computer vision with YOLO11 models, PyTorch, and FastAPI in a microservices architecture.

### Current Production Status

The system is in **Phase 2.0 of Production Deployment** with the following completed infrastructure:

**Completed Infrastructure (Phases 1.1-1.5.3):**
- ✅ **Unified ML Pipeline**: CoreVisionEngine with gRPC services, CUDA memory management, <100ms inference
- ✅ **Production Kubernetes**: EKS cluster with GPU nodes (NVIDIA Tesla V100)
- ✅ **Database Infrastructure**: Citus distributed PostgreSQL (64 shards, 90% compression, 10TB/day capacity)
- ✅ **Connection Pooling**: PgBouncer supporting 10,000+ concurrent connections
- ✅ **High Availability**: 3 coordinator + 6 worker PostgreSQL cluster with Patroni
- ✅ **Monitoring Stack**: Prometheus, Grafana (5 dashboards), NVIDIA DCGM exporter, PagerDuty alerts
- ✅ **Container Infrastructure**: Docker Compose for local development and production environments

**Current Phase (In Progress):**
- 🚧 **Phase 2.0: ML Pipeline Microservices Architecture** - Decomposing monolithic ML pipeline into scalable microservices

**Upcoming Phases:**
- Phase 3.0: Production Resilience & Security (Istio, Falco, Velero)
- Phase 4.0: Edge Computing & Federated Learning (NVIDIA Jetson, FedML)
- Phase 5.0: Multi-tenancy & Global Scaling (10,000+ tenants)
- Phase 6.0: Cost Optimization & FinOps (Karpenter, 70% cost savings)
- Phase 7.0: Advanced AI/ML Capabilities (MLflow, AutoML)
- Phase 8.0: Developer Experience & Platform Engineering (Backstage, golden paths)

## Development Commands

### Environment Setup

```bash
# Install dependencies using uv (Python package manager)
uv sync

# Install development dependencies
uv sync --group dev

# Install ML/GPU dependencies  
uv sync --group ml --group gpu

# Install edge deployment dependencies
uv sync --group edge
```

### Code Quality & Testing

```bash
# Run all tests with coverage (must maintain >90% coverage)
pytest --cov=src/its_camera_ai --cov-report=html --cov-report=term-missing --cov-fail-under=90

# Run specific test categories
pytest -m "not slow"          # Skip slow tests
pytest -m integration         # Integration tests only
pytest -m ml                  # ML model tests only
pytest -m gpu                 # GPU-dependent tests
pytest -m unit                # Unit tests only
pytest -m e2e                 # End-to-end tests
pytest -m benchmark           # Performance benchmark tests
pytest -m performance         # Performance tests
pytest -m asyncio             # Async tests

# Run a single test file
pytest tests/test_streaming_service.py -v

# Run tests matching a pattern
pytest -k "test_stream" -v

# Type checking
mypy src/

# Code formatting and linting (run in order)
black src/ tests/
isort src/ tests/
ruff check src/ tests/
ruff format src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/

# Security scanning
bandit -r src/
safety check
pip-audit
```

### Running the Application

```bash
# Run the main application
python src/its_camera_ai/main.py

# Run FastAPI development server
uvicorn its_camera_ai.api.app:app --reload --host 0.0.0.0 --port 8000

# Run CLI interface
its-camera-ai --help

# Run specific services
its-camera-ai services start
its-camera-ai monitoring dashboard
```

## Architecture Overview

The system follows an **event-driven microservices architecture** with the following key components:

### Core Structure

- **`src/its_camera_ai/`** - Main application code with modular architecture
- **`src/its_camera_ai/ml/`** - Comprehensive ML pipeline with federated learning, model registry, and inference optimization
- **`security/`** - Zero-trust security framework with encryption, privacy controls, and threat detection
- **`src/its_camera_ai/main.py`** - Application entry point
- **`src/its_camera_ai/api/`** - FastAPI web services and routers
- **`src/its_camera_ai/cli/`** - Comprehensive CLI interface with interactive commands

### Key Architectural Patterns

#### ML Pipeline Components

- **DataIngestionPipeline** - Real-time camera stream processing with quality validation
- **ModelRegistry** - Centralized model versioning and deployment management  
- **InferenceEngine** - High-performance batch inference with GPU optimization
- **DistributedTrainingManager** - Federated learning across edge nodes
- **MLMonitoringSystem** - Model drift detection and performance monitoring
- **ExperimentationPlatform** - A/B testing for model variants

#### Security Components

- **EncryptionManager** - AES/RSA encryption for video data and communications
- **PrivacyEngine** - GDPR-compliant anonymization with differential privacy
- **MultiFactorAuthenticator** - JWT-based auth with TOTP/SMS MFA
- **RoleBasedAccessControl** - Fine-grained permission system
- **ThreatDetectionEngine** - Real-time security threat monitoring
- **SecurityAuditLogger** - Comprehensive compliance logging

### Technology Stack

- **Backend**: Python 3.12+, FastAPI, Pydantic v2
- **ML/AI**: PyTorch 2.0+, YOLO11 (Ultralytics), OpenCV 4.8+
- **Data**: PostgreSQL, Redis, TimescaleDB, Apache Kafka
- **Infrastructure**: Docker, Kubernetes, Prometheus/Grafana

## Development Guidelines

### ML Model Development

- Models are managed through the `ModelRegistry` class with semantic versioning
- Use `DeploymentStage` enum for promoting models (development → staging → canary → production)
- Production models require >90% accuracy and <100ms latency
- Implement proper model monitoring with drift detection

### Security Requirements

- All video data must be encrypted using `EncryptionManager`
- Apply privacy protections based on `SecurityLevel` classification
- Use `SecurityContext` for all operations requiring authentication
- Log security events using `SecurityAuditLogger`

### Testing Strategy

- Unit tests for individual components
- Integration tests for service interactions
- ML model tests with performance benchmarks
- GPU tests for inference optimization
- Security tests for authentication/authorization

### Code Organization

- Follow the established `src/its_camera_ai/` modular structure for new components
- Use type hints and Pydantic v2 models for all data structures
- Implement async/await patterns for I/O operations with proper resource cleanup
- Follow the security-first approach with comprehensive error handling
- **ALWAYS use dependency injection patterns with `dependency-injector` for CLI and API components**
- Maintain strict type checking with MyPy configuration
- Follow consistent import organization with isort and black formatting

## Performance Considerations

- The system is designed for **sub-100ms inference latency**
- Uses **GPU batching** for efficient model inference
- Implements **multi-level caching** (L1 in-memory, L2 Redis)
- Supports **horizontal scaling** with Kubernetes HPA
- **Memory optimization** with proper GPU memory management

## Security & Compliance

- Implements **zero-trust architecture** principles
- **Privacy-by-design** with configurable anonymization
- **GDPR/CCPA compliance** with data export/deletion capabilities
- **Comprehensive audit logging** for security events
- **Multi-factor authentication** with role-based access control

## Deployment

### Quick Start Options

#### Docker Compose (Recommended for Development)

```bash
# Local development environment
docker-compose up -d

# Production-like environment with clustering
docker-compose -f docker-compose.prod.yml up -d

# One-command setup with verification
./scripts/quickstart.sh --env development --gpu --monitoring
```

#### Production Kubernetes Deployment

```bash
# 1. Deploy database infrastructure
./scripts/deploy-database.sh

# 2. Deploy monitoring stack  
./scripts/deploy-monitoring.sh

# 3. Deploy GitOps infrastructure
./scripts/deploy-gitops.sh

# 4. Deploy application services
kubectl apply -f k8s/application/
```

### Container Environments

**Development Environment (`docker-compose.yml`):**
- Single PostgreSQL with TimescaleDB
- Single Redis instance
- MinIO object storage
- Apache Kafka (single broker)
- Prometheus + Grafana monitoring
- Application with hot-reload

**Production Environment (`docker-compose.prod.yml`):**
- PostgreSQL with production settings (2GB shared_buffers)
- Redis master-replica setup
- MinIO cluster (4 nodes)
- Kafka cluster (3 brokers)
- Full monitoring stack with retention
- Load-balanced application instances (nginx)
- Background workers with concurrency

### Kubernetes Production Features

- **Database**: Citus distributed PostgreSQL with 64 shards, 90% compression, 10TB/day capacity
- **Connection Pooling**: PgBouncer supporting 10,000+ concurrent connections
- **High Availability**: 3 coordinator + 6 worker PostgreSQL cluster with Patroni + etcd
- **Monitoring**: Prometheus, Grafana (5 dashboards), NVIDIA DCGM exporter, PagerDuty alerts
- **GPU Support**: NVIDIA Tesla V100 nodes with DCGM metrics collection
- **Auto-scaling**: HPA based on inference queue length and GPU utilization
- **Storage**: Persistent volumes with backup and disaster recovery

## CLI Interface

The system includes a comprehensive CLI built with Typer:

```bash
# Interactive dashboard
its-camera-ai dashboard

# ML model management
its-camera-ai ml deploy --model-path ./model.pt
its-camera-ai ml monitor --deployment-stage production

# Service management
its-camera-ai services start --service camera-processor
its-camera-ai services status

# Database operations
its-camera-ai database migrate
its-camera-ai database seed --env development

# Security and monitoring
its-camera-ai security audit
its-camera-ai monitoring metrics --service inference-engine
```

## Important Implementation Notes

### Dependency Injection

- **ALWAYS use `dependency-injector` for CLI and API components** - this is a critical architectural requirement
- Configure containers properly for service dependencies
- Use proper async context managers for resource lifecycle

### File Management

- NEVER create files unless absolutely necessary for achieving the goal
- ALWAYS prefer editing existing files to creating new ones
- NEVER proactively create documentation files (*.md) or README files unless explicitly requested

### Testing Best Practices

- Write tests for all new functionality
- Use appropriate pytest markers (@pytest.mark.asyncio, @pytest.mark.unit, etc.)
- Maintain >90% test coverage requirement
- Mock external dependencies properly in unit tests

### Error Handling

- Implement comprehensive error handling with proper logging
- Use structured logging with `structlog`
- Return appropriate HTTP status codes in API endpoints
- Provide clear error messages for CLI commands

## Development Workflow

### Getting Started (Ultra-Fast Setup)

```bash
# Method 1: One-command setup
./scripts/quickstart.sh --env development --gpu --monitoring

# Method 2: Docker Compose
docker-compose up -d && open http://localhost:8000/docs

# Method 3: Manual setup
uv sync --group dev --group ml
docker-compose up -d postgres redis minio
alembic upgrade head
uvicorn its_camera_ai.api.app:app --reload
```

### Development Commands Reference

**Environment Management:**
```bash
uv sync                           # Install core dependencies
uv sync --group dev               # Add development tools
uv sync --group ml --group gpu    # Add ML and GPU support
uv sync --group edge              # Add edge deployment tools
```

**Code Quality (Always run before commits):**
```bash
black src/ tests/                 # Format code
isort src/ tests/                 # Sort imports
ruff check --fix src/ tests/      # Lint and auto-fix
mypy src/                         # Type checking
bandit -r src/                    # Security scanning
```

**Testing Strategy:**
```bash
pytest                            # Run all tests
pytest -m "not slow"              # Skip slow tests
pytest -m integration             # Integration tests only
pytest -m ml                      # ML model tests
pytest -m gpu                     # GPU-dependent tests (requires GPU)
pytest -m unit                    # Unit tests only
pytest --cov-report=html          # Generate coverage report
```

**Database Operations:**
```bash
alembic upgrade head              # Run migrations
alembic revision --autogenerate   # Create new migration
its-camera-ai database seed       # Seed test data
its-camera-ai database backup     # Backup database
```

**Service Management:**
```bash
its-camera-ai services status     # Check all services
its-camera-ai services logs       # View service logs
its-camera-ai dashboard           # Interactive dashboard
its-camera-ai monitoring metrics  # View metrics
```

**ML Model Operations:**
```bash
its-camera-ai ml deploy           # Deploy model
its-camera-ai ml monitor          # Monitor model performance
its-camera-ai ml benchmark        # Run performance benchmarks
```

### Performance Optimization Commands

```bash
# Profile application performance
python -m cProfile -o profile.stats src/its_camera_ai/main.py

# GPU memory profiling
nvidia-smi -l 1                   # Monitor GPU usage
python -c "import torch; print(torch.cuda.memory_summary())"

# Database performance analysis
its-camera-ai database analyze    # Analyze query performance
```

### Infrastructure Validation

```bash
# Verify Docker Compose setup
docker-compose ps                 # Check service status
docker-compose logs -f app        # Monitor application logs

# Verify Kubernetes deployment
kubectl get pods -A               # Check all pods
kubectl describe pod <pod-name>   # Debug specific pod

# Health checks
curl http://localhost:8000/health # API health
curl http://localhost:9090/-/healthy # Prometheus health
```

### Common Development Tasks

**Adding a new ML model:**
1. Add model file to `models/` directory
2. Update `ModelRegistry` configuration
3. Create model-specific tests in `tests/ml/`
4. Deploy with `its-camera-ai ml deploy --model-path ./models/new_model.pt`

**Adding a new API endpoint:**
1. Create endpoint in `src/its_camera_ai/api/routes/`
2. Add Pydantic models in `src/its_camera_ai/api/schemas/`
3. Update OpenAPI documentation
4. Add tests in `tests/api/`

**Adding a new CLI command:**
1. Create command in `src/its_camera_ai/cli/commands/`
2. Register in `src/its_camera_ai/cli/main.py`
3. Add tests in `tests/cli/`
4. Update help documentation

### Troubleshooting Common Issues

**GPU not detected:**
```bash
nvidia-smi                        # Check driver
python -c "import torch; print(torch.cuda.is_available())"
uv sync --group gpu               # Reinstall GPU dependencies
```

**Database connection issues:**
```bash
docker-compose logs postgres      # Check database logs
its-camera-ai database status     # Check connection
alembic current                   # Check migration status
```

**Performance issues:**
```bash
its-camera-ai monitoring resources # Check system resources
docker stats                     # Check container resources
its-camera-ai profile --duration 60s # Profile application
```
