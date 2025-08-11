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
# Run all tests with coverage
pytest --cov=src/its_camera_ai --cov-report=html --cov-report=term-missing --cov-fail-under=90

# Run specific test categories
pytest -m "not slow"          # Skip slow tests
pytest -m integration         # Integration tests only
pytest -m ml                  # ML model tests only
pytest -m gpu                 # GPU-dependent tests

# Type checking
mypy src/

# Code formatting and linting
black src/ tests/
ruff check src/ tests/
isort src/ tests/

# Security scanning
bandit -r src/
safety check
pip-audit
```

### Running the Application

```bash
# Run the main application
python main.py

# Run with development server (if FastAPI service exists)
uvicorn its_camera_ai.app.api.main:app --reload --host 0.0.0.0 --port 8000

# Run CLI interface
its-camera-ai --help
```

## Architecture Overview

The system follows an **event-driven microservices architecture** with the following key components:

### Core Structure

- **`src/tca/`** - Main application code using Test-Driven Architecture (TCA) pattern
- **`ml_architecture.py`** - Comprehensive ML pipeline with federated learning, model registry, and inference optimization
- **`security/zero_trust_architecture.py`** - Zero-trust security framework with encryption, privacy controls, and threat detection
- **`main.py`** - Application entry point

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
- **Data**: PostgreSQL, Redis, InfluxDB, Apache Kafka
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

- Follow the established `src/tca/` structure for new components
- Use type hints and Pydantic models for all data structures
- Implement async/await patterns for I/O operations
- Follow the security-first approach with proper error handling

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
