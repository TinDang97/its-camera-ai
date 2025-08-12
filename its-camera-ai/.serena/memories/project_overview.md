# ITS Camera AI - Project Overview

## Purpose
AI-powered camera traffic monitoring system for real-time traffic analytics and vehicle tracking. The system uses computer vision with YOLO11 models, PyTorch, and FastAPI in a microservices architecture.

## Key Features
- Real-time camera stream processing with YOLO11 object detection
- Traffic analytics and incident detection
- Scalable microservices architecture
- Edge deployment capabilities
- Zero-trust security framework
- Sub-100ms inference latency with >90% accuracy targets
- Federated learning across edge nodes
- Production ML pipeline with model registry and monitoring

## Technology Stack

### Core Technologies
- **Backend**: Python 3.12+, FastAPI, Pydantic v2
- **ML/AI**: PyTorch 2.0+, YOLO11 (Ultralytics), OpenCV 4.8+
- **Data**: PostgreSQL, Redis, InfluxDB, Apache Kafka
- **Infrastructure**: Docker, Kubernetes, Prometheus/Grafana

### Package Management
- **uv**: Fast Python package installer and resolver
- Uses dependency groups for different environments (dev, ml, gpu, edge, etc.)

### Architecture Patterns
- Event-driven microservices architecture
- Test-Driven Architecture (TCA) pattern in src/tca/ structure
- Zero-trust security framework
- ML pipeline with federated learning and model registry
- Edge-cloud hybrid deployment

## Core Components

### ML Pipeline Components
- **DataIngestionPipeline**: Real-time camera stream processing
- **ModelRegistry**: Centralized model versioning and deployment
- **InferenceEngine**: High-performance batch inference with GPU optimization
- **DistributedTrainingManager**: Federated learning across edge nodes
- **MLMonitoringSystem**: Model drift detection and performance monitoring
- **ExperimentationPlatform**: A/B testing for model variants

### Security Components
- **EncryptionManager**: AES/RSA encryption for video data
- **PrivacyEngine**: GDPR-compliant anonymization
- **MultiFactorAuthenticator**: JWT-based auth with TOTP/SMS MFA
- **RoleBasedAccessControl**: Fine-grained permission system
- **ThreatDetectionEngine**: Real-time security threat monitoring
- **SecurityAuditLogger**: Comprehensive compliance logging

### Deployment Modes
- DEVELOPMENT: Local development environment
- STAGING: Pre-production testing
- PRODUCTION: Full production deployment with auto-scaling
- EDGE: Edge node deployment with limited resources