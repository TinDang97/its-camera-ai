# ITS Camera AI - Codebase Architecture

## High-Level Architecture

### System Design Pattern
- **Event-driven microservices architecture** with async communication
- **Test-Driven Architecture (TCA)** pattern in `src/tca/` structure  
- **Zero-trust security framework** with encryption and privacy controls
- **Hybrid edge-cloud deployment** supporting both local and distributed processing

### Core Architectural Principles
1. **Security-first approach** with comprehensive encryption and privacy controls
2. **Performance optimization** targeting <100ms inference latency
3. **Scalability** from single cameras to enterprise-level deployments (1000+ cameras)
4. **Evolvable architecture** with modular components and clear interfaces
5. **Production-ready** with comprehensive monitoring and observability

## Directory Structure

```
its-camera-ai/
├── src/its_camera_ai/              # Main application code
│   ├── core/                       # Core utilities and configuration
│   │   ├── config.py              # Application configuration management
│   │   ├── logging.py             # Structured logging setup
│   │   └── exceptions.py          # Custom exception hierarchy
│   ├── api/                       # FastAPI web services
│   │   ├── routers/               # API route handlers
│   │   │   ├── auth.py           # Authentication endpoints
│   │   │   ├── cameras.py        # Camera management endpoints
│   │   │   ├── analytics.py      # Traffic analytics endpoints
│   │   │   └── models.py         # ML model management endpoints
│   │   ├── schemas/               # Pydantic data models
│   │   └── dependencies.py       # FastAPI dependency injection
│   ├── ml/                        # ML pipeline components
│   │   ├── ml_pipeline.py         # Production ML pipeline orchestration
│   │   ├── inference_optimizer.py # GPU/CPU inference optimization
│   │   ├── federated_learning.py  # Distributed training coordination
│   │   ├── production_monitoring.py # Model performance monitoring
│   │   └── edge_deployment.py     # Edge node deployment strategies
│   ├── data/                      # Data processing and streaming
│   │   ├── streaming_processor.py # Real-time stream processing
│   │   ├── redis_queue_manager.py # Message queue management
│   │   └── grpc_serialization.py # Protocol buffer serialization
│   ├── services/                  # Business logic services
│   │   ├── auth.py               # Authentication service
│   │   └── cache.py              # Caching service
│   ├── storage/                   # Model storage and registry
│   │   ├── model_registry.py     # ML model version management
│   │   ├── minio_service.py      # Object storage service
│   │   └── models.py             # Storage data models
│   ├── models/                    # Database models
│   │   ├── user.py               # User authentication models
│   │   ├── database.py           # Database session management
│   │   └── base.py               # Base model classes
│   ├── proto/                     # gRPC protocol definitions
│   │   ├── processed_frame_pb2.py # Frame processing protocols
│   │   └── streaming_service_pb2.py # Streaming service protocols
│   └── production_pipeline.py     # Main production orchestrator
├── tests/                         # Test suite
├── infrastructure/                # Infrastructure as code
├── docs/                         # Documentation
├── examples/                     # Usage examples
└── scripts/                      # Utility scripts
```

## Key Architectural Components

### 1. ML Pipeline Components (`src/its_camera_ai/ml/`)

#### Production ML Pipeline (`ml_pipeline.py`)
- **ProductionMLPipeline**: Central orchestrator for ML operations
- **DataIngestionPipeline**: Real-time camera stream processing with quality validation
- **ModelRegistry**: Centralized model versioning and deployment management
- **InferenceEngine**: High-performance batch inference with GPU optimization
- **DistributedTrainingManager**: Federated learning across edge nodes
- **MLMonitoringSystem**: Model drift detection and performance monitoring
- **ExperimentationPlatform**: A/B testing for model variants

#### Key Features
- Semantic versioning for models (development → staging → canary → production)
- GPU batching for efficient inference (<100ms latency target)
- Multi-level caching (L1 in-memory, L2 Redis)
- Horizontal scaling with Kubernetes HPA
- Memory optimization with proper GPU management

### 2. Security Framework (`src/its_camera_ai/security/`)

#### Zero-Trust Architecture Components
- **EncryptionManager**: AES/RSA encryption for video data and communications
- **PrivacyEngine**: GDPR-compliant anonymization with differential privacy
- **MultiFactorAuthenticator**: JWT-based auth with TOTP/SMS MFA
- **RoleBasedAccessControl**: Fine-grained permission system
- **ThreatDetectionEngine**: Real-time security threat monitoring
- **SecurityAuditLogger**: Comprehensive compliance logging

#### Security Patterns
- All video data encrypted in transit and at rest
- Privacy-by-design with configurable anonymization
- Comprehensive audit logging for compliance
- Zero-trust principles with continuous verification

### 3. Data Processing Layer (`src/its_camera_ai/data/`)

#### Real-time Streaming (`streaming_processor.py`)
- **StreamProcessor**: High-throughput camera stream processing
- Async frame processing with backpressure handling
- Quality validation and frame rate optimization
- Integration with Redis for queue management

#### Message Queue Management (`redis_queue_manager.py`)  
- **RedisQueueManager**: Reliable message passing between services
- Priority queuing for different camera streams
- Dead letter queue handling for failed processing
- Metrics collection for queue performance

### 4. API Layer (`src/its_camera_ai/api/`)

#### RESTful Web Services
- **FastAPI** with automatic OpenAPI documentation
- **Pydantic v2** for data validation and serialization
- **JWT authentication** with role-based access control
- **Rate limiting** and **request validation**
- **Health checks** and **metrics endpoints**

#### Key Endpoints
- `/auth/*` - Authentication and authorization
- `/cameras/*` - Camera management and configuration  
- `/analytics/*` - Traffic analytics and reporting
- `/models/*` - ML model management and deployment
- `/health` - System health and readiness checks

### 5. Storage Layer (`src/its_camera_ai/storage/`)

#### Model Storage (`model_registry.py`)
- **ModelRegistry**: Centralized model version control
- Support for multiple model formats (PyTorch, ONNX, TensorRT)
- Automated model validation and testing
- Deployment stage management and promotion

#### Object Storage (`minio_service.py`)
- **MinIO integration** for scalable object storage
- Encrypted storage for models and video data
- Backup and disaster recovery capabilities
- Multi-region replication support

## Deployment Architecture

### Edge-Cloud Hybrid Model
- **Edge Nodes**: Local inference with limited models
- **Cloud Services**: Full ML pipeline with training and advanced analytics
- **Federated Learning**: Collaborative training across edge nodes
- **Model Synchronization**: Automated model updates and version control

### Scalability Patterns
- **Horizontal scaling** with Kubernetes and Docker
- **Auto-scaling** based on inference queue length and resource utilization
- **Load balancing** across multiple inference engines
- **Circuit breakers** for fault tolerance

### Performance Targets
- **Inference latency**: <100ms per frame
- **Throughput**: 30,000 fps aggregate across all cameras  
- **Accuracy**: >90% for production models
- **Availability**: 99.9% uptime with automated failover

## Integration Points

### External Services
- **Kafka**: Event streaming and message passing
- **PostgreSQL**: Persistent data storage with async connections
- **Redis**: Caching and session management
- **InfluxDB**: Time-series data for metrics and analytics
- **Prometheus/Grafana**: Monitoring and alerting

### Protocol Support
- **gRPC**: High-performance service-to-service communication
- **WebSocket**: Real-time client connections
- **REST API**: Standard HTTP interfaces
- **Protocol Buffers**: Efficient data serialization

## Development Patterns

### Async Programming
- Extensive use of `async/await` for I/O operations
- Proper resource cleanup with context managers
- Connection pooling for database and external services
- Graceful shutdown handling

### Error Handling
- Custom exception hierarchy for different error types
- Structured logging with correlation IDs
- Circuit breaker pattern for external service calls
- Proper error propagation and recovery strategies

### Testing Strategy
- **Unit tests**: Individual component testing with mocks
- **Integration tests**: Service interaction validation  
- **ML tests**: Model accuracy and performance benchmarks
- **GPU tests**: Hardware-specific optimization validation
- **End-to-end tests**: Complete workflow validation