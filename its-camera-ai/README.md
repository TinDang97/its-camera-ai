# ITS Camera AI

AI-powered camera traffic monitoring system for real-time traffic analytics and vehicle tracking.

## Features

- Real-time camera stream processing with YOLO11 object detection
- Traffic analytics and incident detection
- Scalable microservices architecture
- Edge deployment capabilities
- Zero-trust security framework

## Quick Start

### Environment Setup

```bash
# Install dependencies using uv
uv sync

# Install development dependencies
uv sync --group dev

# Install ML/GPU dependencies  
uv sync --group ml --group gpu
```

### Running the Application

```bash
# Run the main application
python main.py

# Run with development server
uvicorn its_camera_ai.app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Code Quality & Testing

```bash
# Run all tests with coverage
pytest --cov=src/its_camera_ai --cov-report=html --cov-report=term-missing --cov-fail-under=90

# Type checking
mypy src/

# Code formatting and linting
black src/ tests/
ruff check src/ tests/
isort src/ tests/
```

## Architecture

The system follows an event-driven microservices architecture with:

- **ML Pipeline**: Real-time inference with YOLO11 models
- **Security**: Zero-trust architecture with encryption and privacy controls
- **Deployment**: Kubernetes-based with auto-scaling
- **Monitoring**: Comprehensive observability stack

## Technology Stack

- **Backend**: Python 3.12+, FastAPI, Pydantic v2
- **ML/AI**: PyTorch 2.0+, YOLO11 (Ultralytics), OpenCV 4.8+
- **Data**: PostgreSQL, Redis, InfluxDB, Apache Kafka
- **Infrastructure**: Docker, Kubernetes, Prometheus/Grafana

## Development

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines and architecture overview.

## License

See LICENSE file for details.