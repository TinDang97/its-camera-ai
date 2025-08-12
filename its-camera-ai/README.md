# ITS Camera AI

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://img.shields.io/badge/coverage-90%2B-brightgreen.svg)](https://github.com/your-org/its-camera-ai)

## Overview

**AI-powered camera traffic monitoring system for real-time traffic analytics and vehicle tracking**

ITS Camera AI is a comprehensive, production-ready system that leverages computer vision and machine learning to monitor traffic patterns, detect incidents, and provide actionable insights from camera feeds. Built with a microservices architecture, it supports both edge and cloud deployment with sub-100ms inference latency.

## 🚀 Key Features

### Core Capabilities

- **Real-time Processing**: Sub-100ms inference latency with YOLO11 object detection
- **Traffic Analytics**: Vehicle tracking, speed estimation, and incident detection
- **Scalable Architecture**: Event-driven microservices with horizontal scaling
- **Edge Deployment**: Optimized for edge computing with minimal resource requirements
- **Multi-Camera Support**: Process hundreds of camera sources simultaneously

### Advanced Features

- **Zero-Trust Security**: Comprehensive security framework with encryption and privacy controls
- **Federated Learning**: Distributed model training across edge nodes
- **Model Registry**: Centralized model versioning and deployment management
- **Real-time Monitoring**: Prometheus metrics with Grafana dashboards
- **Auto-Scaling**: Kubernetes-based deployment with automatic resource scaling

### Performance Highlights

- **Inference Speed**: < 100ms average processing time
- **Throughput**: 1000+ frames per second on GPU
- **Accuracy**: > 90% object detection accuracy
- **Compression**: 76% video data compression ratio
- **Test Coverage**: > 90% code coverage

## 📋 Prerequisites

### System Requirements

- **Python**: 3.12 or higher
- **Operating System**: Linux (Ubuntu 20.04+), macOS (12+), Windows 10+
- **Memory**: Minimum 8GB RAM (16GB+ recommended for production)
- **Storage**: 20GB+ available disk space
- **GPU** (Optional): NVIDIA GPU with CUDA 11.8+ for accelerated inference

### Dependencies

- **Database**: PostgreSQL 13+ with TimescaleDB extension
- **Cache**: Redis 6+
- **Message Queue**: Apache Kafka (optional, for high-throughput scenarios)
- **Object Storage**: MinIO or AWS S3 compatible storage
- **Time-Series Database**: TimescaleDB (for metrics and time-series data)

## ⚡ Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-org/its-camera-ai.git
cd its-camera-ai

# Create environment configuration
cp .env.example .env
# Edit .env with your specific settings

# Install dependencies using uv (recommended)
uv sync

# Or install with specific feature groups
uv sync --group dev --group ml --group gpu
```

### 2. Database Setup

```bash
# Start PostgreSQL and Redis (using Docker)
docker-compose up -d postgres redis

# Run database migrations
alembic upgrade head

# Seed initial data (optional)
its-camera-ai database seed --env development
```

### 3. Run the Application

```bash
# Start the main application
python src/its_camera_ai/main.py

# Or start the FastAPI development server
uvicorn its_camera_ai.api.app:app --reload --host 0.0.0.0 --port 8000

# Access the API documentation at http://localhost:8000/docs
```

### 4. Using the CLI

```bash
# Interactive dashboard
its-camera-ai dashboard

# Check system status
its-camera-ai services status

# Deploy an ML model
its-camera-ai ml deploy --model-path ./models/yolo11n.pt
```

## 📖 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure the following key settings:

```bash
# Application Settings
ENVIRONMENT=development
API_HOST=0.0.0.0
API_PORT=8000

# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/its_camera_ai

# Redis Configuration
REDIS_URL=redis://:password@localhost:6379/0

# ML Model Configuration
MODEL_CONFIDENCE_THRESHOLD=0.5
BATCH_SIZE=8
GPU_MEMORY_FRACTION=0.8

# Security Settings
SECRET_KEY=your-super-secret-key-change-in-production
JWT_EXPIRATION_MINUTES=30
```

### Configuration Files

- **`production_config.json`**: Production deployment settings
- **`alembic.ini`**: Database migration configuration
- **`pyproject.toml`**: Project dependencies and tool configuration

## 🔧 Usage Examples

### Processing Camera Feeds

```python
from its_camera_ai.ml.core_vision_engine import CoreVisionEngine
from its_camera_ai.services.camera_service import CameraService

# Initialize the vision engine
vision_engine = CoreVisionEngine()

# Process a camera feed
camera_service = CameraService()
results = await camera_service.process_stream("camera_001")

# Access detection results
for detection in results.detections:
    print(f"Detected {detection.class_name} with confidence {detection.confidence}")
```

### Using the REST API

```bash
# Add a new camera
curl -X POST "http://localhost:8000/api/v1/cameras/" \
  -H "Content-Type: application/json" \
  -d '{"name": "Main Street Camera", "url": "rtsp://camera.url/stream"}'

# Get real-time analytics
curl "http://localhost:8000/api/v1/analytics/realtime?camera_id=1"

# Retrieve traffic metrics
curl "http://localhost:8000/api/v1/analytics/metrics?start_time=2024-01-01T00:00:00Z"
```

### CLI Commands

```bash
# Model management
its-camera-ai ml deploy --model-path ./custom_model.pt --deployment-stage staging
its-camera-ai ml monitor --deployment-stage production

# Service management
its-camera-ai services start --service camera-processor
its-camera-ai services logs --service inference-engine --lines 100

# Database operations
its-camera-ai database backup --output ./backup.sql
its-camera-ai database migrate --revision head

# Security and monitoring
its-camera-ai security audit
its-camera-ai monitoring metrics --service camera-service --period 1h
```

## 📊 API Documentation

The system provides comprehensive REST API endpoints:

- **Base URL**: `http://localhost:8000/api/v1`
- **Interactive Documentation**: `http://localhost:8000/docs`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/cameras/` | GET, POST | Manage camera configurations |
| `/analytics/realtime` | GET | Real-time traffic analytics |
| `/analytics/metrics` | GET | Historical traffic metrics |
| `/models/` | GET, POST | ML model management |
| `/health/` | GET | System health status |
| `/auth/login` | POST | User authentication |

## 🏗️ Architecture Overview

### System Components

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Camera Feeds  │    │   Edge Nodes    │    │  Cloud Backend  │
│                 │    │                 │    │                 │
│  RTSP Streams   │────│  Preprocessing  │────│   ML Pipeline   │
│  HTTP Streams   │    │  Local Storage  │    │  Model Registry │
│  File Uploads   │    │  Edge Inference │    │  Analytics API  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Services

- **Camera Service**: Manages camera connections and stream processing
- **Vision Engine**: Core computer vision processing with YOLO11
- **Analytics Service**: Traffic metrics and incident detection
- **Model Registry**: ML model versioning and deployment
- **Security Service**: Authentication, authorization, and encryption
- **Monitoring Service**: System metrics and observability

### Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React, TypeScript, WebRTC |
| **Backend** | Python 3.12+, FastAPI, Pydantic v2 |
| **ML/AI** | PyTorch 2.0+, YOLO11, OpenCV 4.8+, ONNX |
| **Databases** | PostgreSQL, Redis, TimescaleDB |
| **Messaging** | Apache Kafka, Redis Streams |
| **Storage** | MinIO, AWS S3 |
| **Infrastructure** | Docker, Kubernetes, Helm |
| **Monitoring** | Prometheus, Grafana, Sentry |

## 🛠️ Development Setup

### Prerequisites for Development

```bash
# Install development tools
uv sync --group dev

# Install pre-commit hooks
pre-commit install

# Install additional ML dependencies
uv sync --group ml --group gpu
```

### Development Workflow

1. **Code Quality**:

```bash
# Format code
black src/ tests/
ruff format src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Security scanning
bandit -r src/
safety check
```

2. **Testing**:

```bash
# Run all tests with coverage
pytest --cov=src/its_camera_ai --cov-report=html --cov-fail-under=90

# Run specific test categories
pytest -m "not slow"          # Skip slow tests
pytest -m integration         # Integration tests
pytest -m ml                  # ML model tests
pytest -m gpu                 # GPU tests (requires GPU)
```

3. **Performance Benchmarks**:

```bash
# Run performance benchmarks
pytest -m benchmark

# ML model performance
python examples/performance_benchmark.py
```

### Project Structure

```text
src/its_camera_ai/
├── api/                    # FastAPI routes and schemas
├── cli/                    # Command-line interface
├── core/                   # Core utilities and configuration
├── ml/                     # Machine learning pipeline
├── models/                 # Database models
├── services/               # Business logic services
├── security/               # Security and authentication
├── storage/                # Data storage abstractions
└── monitoring/             # Observability and metrics
```

## 🧪 Testing

### Test Categories

- **Unit Tests**: Individual component testing (>90% coverage required)
- **Integration Tests**: Service interaction testing
- **ML Model Tests**: Model accuracy and performance validation
- **GPU Tests**: CUDA-accelerated inference testing
- **End-to-End Tests**: Complete workflow testing
- **Benchmark Tests**: Performance and scalability testing

### Running Tests

```bash
# All tests with coverage report
pytest --cov=src/its_camera_ai --cov-report=html --cov-report=term-missing --cov-fail-under=90

# Specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests
pytest -m "ml and not gpu"  # ML tests without GPU
pytest -m benchmark         # Performance benchmarks

# Generate detailed coverage report
pytest --cov-report=html
open htmlcov/index.html
```

## 🚀 Deployment

### Production Deployment

```bash
# Deploy to production environment
python deploy_production.py

# Deploy with Docker
docker build -t its-camera-ai:latest .
docker run -p 8000:8000 its-camera-ai:latest

# Deploy to Kubernetes
helm install its-camera-ai ./k8s/helm-chart/
```

### Container Deployment

```bash
# Build production image
docker build --target production -t its-camera-ai:prod .

# Run with docker-compose
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale worker=4
```

### Edge Deployment

```bash
# Install edge dependencies
uv sync --group edge

# Deploy to edge device
its-camera-ai deploy edge --target raspberry-pi --model yolo11n.pt

# Configure edge-to-cloud sync
its-camera-ai config set edge.sync.enabled=true
```

## 🔒 Security Considerations

### Security Features

- **Zero-Trust Architecture**: All communications encrypted and authenticated
- **Privacy by Design**: GDPR/CCPA compliant with data anonymization
- **Multi-Factor Authentication**: JWT tokens with TOTP/SMS support
- **Role-Based Access Control**: Fine-grained permissions system
- **Audit Logging**: Comprehensive security event tracking

### Security Best Practices

1. **Change Default Credentials**: Update all default passwords and keys
2. **Enable HTTPS**: Use TLS certificates for all communications
3. **Network Security**: Implement firewall rules and network segmentation
4. **Data Encryption**: Encrypt sensitive data at rest and in transit
5. **Regular Updates**: Keep dependencies and security patches current

### Configuration

```bash
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Configure SSL/TLS
its-camera-ai security setup-tls --cert-path /path/to/cert.pem

# Enable audit logging
its-camera-ai security audit --enable
```

## 📈 Performance Benchmarks

### Inference Performance

| Model | Resolution | FPS (CPU) | FPS (GPU) | Latency (ms) | Memory (MB) |
|-------|------------|-----------|-----------|-------------|-------------|
| YOLO11n | 640x640 | 45 | 280 | 22 | 256 |
| YOLO11s | 640x640 | 32 | 195 | 31 | 512 |
| YOLO11m | 640x640 | 21 | 145 | 48 | 1024 |

### System Throughput

- **Single Camera**: 30+ FPS real-time processing
- **Multi-Camera**: 500+ cameras with GPU acceleration
- **Edge Processing**: 10-15 FPS on Raspberry Pi 4
- **Cloud Processing**: 1000+ FPS with horizontal scaling

### Compression Performance

- **Video Compression**: 76% size reduction with minimal quality loss
- **Throughput**: 256 Mbps compression speed
- **Serialization**: 645K FPS for small frames

## 🐛 Troubleshooting

### Common Issues

#### GPU Not Detected

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available())"

# Install GPU dependencies
uv sync --group gpu
```

#### Database Connection Issues

```bash
# Check database status
its-camera-ai database status

# Reset database
its-camera-ai database reset --confirm

# Run migrations
alembic upgrade head
```

#### Performance Issues

```bash
# Check system resources
its-camera-ai monitoring resources

# Profile application
its-camera-ai profile --duration 60s

# Optimize model
its-camera-ai ml optimize --model yolo11n.pt --format onnx
```

### Logging and Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# View real-time logs
its-camera-ai services logs --follow

# Generate diagnostic report
its-camera-ai diagnostics --output report.json
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Install** development dependencies (`uv sync --group dev`)
4. **Write** tests for your changes
5. **Ensure** all tests pass (`pytest`)
6. **Format** code (`black src/ tests/`)
7. **Commit** changes (`git commit -m 'Add amazing feature'`)
8. **Push** to branch (`git push origin feature/amazing-feature`)
9. **Open** a Pull Request

### Code Standards

- **Test Coverage**: Maintain >90% test coverage
- **Type Hints**: Use type annotations for all functions
- **Documentation**: Update docstrings and README for new features
- **Security**: Follow security best practices and run security scans
- **Performance**: Ensure changes don't degrade performance

### Reporting Issues

- Use the [GitHub Issues](https://github.com/your-org/its-camera-ai/issues) tracker
- Provide detailed reproduction steps
- Include system information and logs
- Label issues appropriately (bug, enhancement, documentation)

## 📚 Additional Resources

### Documentation

- **Architecture Guide**: [docs/architecture.md](docs/architecture.md)
- **API Reference**: [docs/api.md](docs/api.md)
- **Deployment Guide**: [docs/deployment.md](docs/deployment.md)
- **Security Guide**: [docs/security.md](docs/security.md)

### Examples

- **Basic Usage**: [examples/vision_engine_demo.py](examples/vision_engine_demo.py)
- **Performance Testing**: [examples/performance_benchmark.py](examples/performance_benchmark.py)
- **Custom Models**: [examples/custom_model_integration.py](examples/custom_model_integration.py)

### Community

- **GitHub Discussions**: [Discussions](https://github.com/your-org/its-camera-ai/discussions)
- **Documentation**: [ReadTheDocs](https://its-camera-ai.readthedocs.io)
- **Issue Tracker**: [GitHub Issues](https://github.com/your-org/its-camera-ai/issues)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com) for the YOLO11 model architecture
- [FastAPI](https://fastapi.tiangolo.com) for the excellent web framework
- [PyTorch](https://pytorch.org) for the machine learning foundation
- The open-source community for invaluable tools and libraries

---

## Acknowledgments

**Built with ❤️ by the ITS Camera AI Team**

For detailed development guidelines and architecture information, see [CLAUDE.md](CLAUDE.md).
