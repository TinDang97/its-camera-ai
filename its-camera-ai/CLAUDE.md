# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **ITS Camera AI** project - an AI-powered camera traffic monitoring system for real-time traffic analytics and vehicle tracking. The system uses computer vision with YOLO11 models, PyTorch, and FastAPI in a microservices architecture.

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

# Type checking
mypy src/

# Code formatting and linting
black src/ tests/
ruff check src/ tests/
ruff format src/ tests/        # Additional ruff formatting
isort src/ tests/

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
- Use dependency injection patterns with `dependency-injector`
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

- **Hybrid edge-cloud architecture** supporting both local and distributed processing
- **Containerized services** with Kubernetes orchestration  
- **Auto-scaling** based on inference queue length and resource utilization
- **Multi-region deployment** with automated failover
- **CI/CD pipeline** with automated testing and security scanning
- **Production deployment script**: `python deploy_production.py`
- **MinIO deployment**: `./deploy-minio.sh` for object storage setup
- **Database migrations**: Managed with Alembic (`alembic upgrade head`)

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

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.

- always use dependency injector for cli and api