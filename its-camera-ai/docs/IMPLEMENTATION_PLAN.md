# ITS Camera AI - Comprehensive Implementation Plan

**Document Version:** 1.1  
**Date:** August 12, 2025  
**Author:** Claude Code Analysis  
**Last Updated:** August 12, 2025 - Infrastructure Phase Completed

---

## 🚀 Quick Start

### Infrastructure Status: ✅ OPERATIONAL

All critical infrastructure components are deployed and tested. Get started with:

```bash
# Start all infrastructure
make infra-up

# Check status
make infra-status

# Run tests
make test-infra

# View logs
make infra-logs
```

### Completed Components (Phase 1)
- ✅ **PostgreSQL** - Primary database (port 5432)
- ✅ **TimescaleDB** - Time-series metrics (port 5433)
- ✅ **Redis** - Cache & message queue (port 6379)
- ✅ **Kafka** - Event streaming (port 9092)
- ✅ **MinIO** - Object storage (ports 9000/9001)
- ✅ **Prometheus** - Metrics collection (port 9090)
- ✅ **Grafana** - Monitoring dashboards (port 3000)

### Next Steps
- Begin Phase 2: Core Services implementation
- Initialize databases with schemas: `make db-init`
- Deploy application services on infrastructure

---

## Executive Summary

This document provides a comprehensive implementation plan for the ITS Camera AI system based on the C4 architecture diagrams, sequence diagrams, and high-level design documentation. The plan prioritizes components by criticality, defines clear interfaces, and provides actionable implementation guidance for each architectural component.

## Table of Contents

1. [Component Implementation Matrix](#1-component-implementation-matrix)
2. [Phase-wise Implementation Plan](#2-phase-wise-implementation-plan)
3. [Component Implementation Guides](#3-component-implementation-guides)
4. [Service Implementation Details](#4-service-implementation-details)
5. [Data Flow Implementation](#5-data-flow-implementation)
6. [Integration Points](#6-integration-points)
7. [Testing Strategy](#7-testing-strategy)
8. [Deployment Strategy](#8-deployment-strategy)

---

## 1. Component Implementation Matrix

### 1.1 Infrastructure Components (P0 - Critical)

| Component | Status | Priority | Effort | Dependencies | Team Assignment |
|-----------|--------|----------|--------|--------------|-----------------|
| PostgreSQL Database | ✅ Completed | P0 | 3 days | None | Backend Team |
| Redis Cache/Queue | ✅ Completed | P0 | 2 days | None | Backend Team |
| TimescaleDB Metrics | ✅ Completed | P0 | 2 days | PostgreSQL | Backend Team |
| MinIO Object Storage | ✅ Completed | P0 | 2 days | None | DevOps Team |
| Kafka Message Queue | ✅ Completed | P0 | 3 days | None | Backend Team |

### 1.2 Core Services (P0 - Critical)

| Component | Status | Priority | Effort | Dependencies | Team Assignment |
|-----------|--------|----------|--------|--------------|-----------------|
| Streaming Service | ⚠️ Partial | P0 | 5 days | Redis, gRPC | ML Team |
| Core Vision Engine | ✅ Implemented | P0 | - | GPU, PyTorch | ML Team |
| Analytics Service | ❌ Missing | P0 | 5 days | TimescaleDB | Backend Team |
| Alert Service | ❌ Missing | P0 | 3 days | Kafka, Analytics | Backend Team |
| Authentication Service | ⚠️ Partial | P0 | 3 days | PostgreSQL, Redis | Backend Team |

### 1.3 API Layer (P1 - High)

| Component | Status | Priority | Effort | Dependencies | Team Assignment |
|-----------|--------|----------|--------|--------------|-----------------|
| API Gateway | ✅ Implemented | P1 | - | FastAPI | Backend Team |
| Camera Router | ✅ Implemented | P1 | - | API Gateway | Backend Team |
| Analytics Router | ⚠️ Partial | P1 | 2 days | Analytics Service | Backend Team |
| Model Router | ❌ Missing | P1 | 3 days | Model Registry | ML Team |
| SSE Broadcaster | ✅ Implemented | P1 | - | Redis | Backend Team |

### 1.4 ML Pipeline (P1 - High)

| Component | Status | Priority | Effort | Dependencies | Team Assignment |
|-----------|--------|----------|--------|--------------|-----------------|
| Adaptive Batcher | ✅ Implemented | P1 | - | Core Vision Engine | ML Team |
| GPU Preprocessor | ✅ Implemented | P1 | - | CUDA, Core Vision | ML Team |
| Memory Pool Manager | ✅ Implemented | P1 | - | GPU Hardware | ML Team |
| TensorRT Optimizer | ✅ Implemented | P1 | - | TensorRT | ML Team |
| Model Registry | ❌ Missing | P1 | 4 days | MinIO, Versioning | ML Team |
| Production Monitoring | ✅ Implemented | P1 | - | Metrics Collection | ML Team |

### 1.5 Advanced Features (P2 - Medium)

| Component | Status | Priority | Effort | Dependencies | Team Assignment |
|-----------|--------|----------|--------|--------------|-----------------|
| Federated Learning | ✅ Implemented | P2 | - | Edge Deployment | ML Team |
| Edge Deployment Manager | ✅ Implemented | P2 | - | Kubernetes | DevOps Team |
| Threat Detection Engine | ❌ Missing | P2 | 5 days | Analytics Service | Security Team |
| Privacy Engine | ❌ Missing | P2 | 4 days | Encryption Manager | Security Team |
| Multi-Factor Auth | ❌ Missing | P2 | 3 days | Auth Service | Backend Team |

### 1.6 Monitoring & Operations (P2 - Medium)

| Component | Status | Priority | Effort | Dependencies | Team Assignment |
|-----------|--------|----------|--------|--------------|-----------------|
| Prometheus Metrics | ✅ Completed | P2 | 2 days | Infrastructure | DevOps Team |
| Grafana Dashboards | ✅ Completed | P2 | 3 days | Prometheus | DevOps Team |
| ELK Stack Logging | ❌ Missing | P2 | 4 days | Infrastructure | DevOps Team |
| Security Audit Logger | ❌ Missing | P2 | 2 days | Security Framework | Security Team |
| Health Check System | ✅ Completed | P2 | 2 days | All Services | DevOps Team |

---

## 2. Phase-wise Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2) ✅ COMPLETED
**Objective**: Establish foundational data and messaging infrastructure

#### Sprint 1.1: Database Infrastructure ✅
- ✅ Set up PostgreSQL with optimized schema
- ✅ Configure Redis for caching and message queuing
- ✅ Implement TimescaleDB for metrics storage
- ✅ Set up MinIO for object storage
- **Deliverables**: Functional database layer with migrations

#### Sprint 1.2: Message Infrastructure ✅
- ✅ Deploy Kafka cluster with proper topics
- ✅ Configure Redis pub/sub for real-time events
- ✅ Implement message serialization schemas
- ✅ Set up dead letter queues for error handling
- **Deliverables**: Reliable messaging infrastructure

#### Dependencies Ready For
- All core services can begin development
- Data persistence layer available
- Real-time communication established

### Phase 2: Core Services (Week 3-5)
**Objective**: Implement critical business services

#### Sprint 2.1: Streaming & Analytics Foundation
- Complete Streaming Service implementation
- Build Analytics Service with rule engine
- Implement Alert Service with notification system
- Create basic metrics collection
- **Deliverables**: Working video processing pipeline

#### Sprint 2.2: Authentication & Security
- Complete Authentication Service with JWT
- Implement Role-Based Access Control (RBAC)
- Add session management and security middleware
- Create security audit logging
- **Deliverables**: Secure authentication system

#### Sprint 2.3: Service Integration
- Implement gRPC communication between services
- Add circuit breakers and resilience patterns
- Create service discovery mechanism
- Implement health checks
- **Deliverables**: Resilient service mesh

### Phase 3: API Layer (Week 6-7)
**Objective**: Complete external interfaces

#### Sprint 3.1: API Completion
- Complete Analytics Router implementation
- Build Model Router for ML model management
- Enhance error handling and validation
- Add comprehensive API documentation
- **Deliverables**: Complete REST API

#### Sprint 3.2: Real-time Features
- Optimize SSE Broadcaster for scale
- Implement WebSocket support for dashboards
- Add real-time alert broadcasting
- Create API rate limiting and throttling
- **Deliverables**: Real-time API features

### Phase 4: Advanced ML Features (Week 8-10)
**Objective**: Enhance ML capabilities

#### Sprint 4.1: Model Management
- Build Model Registry with versioning
- Implement A/B testing for models
- Create model performance monitoring
- Add automated model deployment
- **Deliverables**: Complete MLOps pipeline

#### Sprint 4.2: Edge Computing
- Enhance Edge Deployment Manager
- Implement edge-cloud synchronization
- Add offline operation capabilities
- Create federated learning coordination
- **Deliverables**: Edge deployment capabilities

### Phase 5: Security & Privacy (Week 11-12)
**Objective**: Implement advanced security features

#### Sprint 5.1: Threat Detection
- Build Threat Detection Engine
- Implement anomaly detection algorithms
- Add security incident response
- Create security dashboard
- **Deliverables**: Advanced security monitoring

#### Sprint 5.2: Privacy Compliance
- Implement Privacy Engine with anonymization
- Add GDPR compliance features
- Create data export/deletion capabilities
- Implement differential privacy
- **Deliverables**: Privacy-compliant system

### Phase 6: Operations & Monitoring (Week 13-14)
**Objective**: Complete operational readiness

#### Sprint 6.1: Monitoring Stack
- Deploy Prometheus metrics collection
- Create Grafana dashboards
- Implement ELK stack for logging
- Add distributed tracing
- **Deliverables**: Complete observability

#### Sprint 6.2: Production Readiness
- Implement backup and disaster recovery
- Add performance optimization
- Create operational runbooks
- Complete security hardening
- **Deliverables**: Production-ready system

---

## 3. Component Implementation Guides

### 3.1 Streaming Service Implementation

#### Overview
The Streaming Service manages real-time camera stream processing and frame distribution using gRPC and Redis queuing.

#### Interface Definition
```python
class StreamingServiceInterface:
    async def register_camera(self, camera_config: CameraConfig) -> CameraRegistration
    async def process_stream(self, camera_id: str) -> AsyncIterator[ProcessedFrame]
    async def validate_frame_quality(self, frame: RawFrame) -> QualityMetrics
    async def queue_frame_batch(self, frames: List[ProcessedFrame]) -> BatchId
```

#### Data Models
```python
class CameraConfig(BaseModel):
    camera_id: str
    stream_url: str
    resolution: Tuple[int, int]
    fps: int
    protocol: StreamProtocol
    
class ProcessedFrame(BaseModel):
    frame_id: str
    camera_id: str
    timestamp: datetime
    quality_score: float
    metadata: FrameMetadata
```

#### Implementation Checklist
- [ ] Implement gRPC server for camera communication
- [ ] Add RTSP/WebRTC stream handling
- [ ] Create frame quality validation
- [ ] Implement Redis queue integration
- [ ] Add error handling and recovery
- [ ] Create performance metrics collection
- [ ] Add stream health monitoring
- [ ] Implement graceful shutdown

#### Testing Requirements
- Unit tests for stream processing logic
- Integration tests with Redis queues
- Load tests with multiple camera streams
- Error recovery testing
- Performance benchmarking (target: <10ms per frame)

#### Performance Targets
- Support 100+ concurrent camera streams
- Frame processing latency < 10ms
- 99.9% frame processing success rate
- Memory usage < 4GB per service instance

### 3.2 Analytics Service Implementation

#### Overview
The Analytics Service processes vehicle tracking data, evaluates traffic rules, and generates insights using TimescaleDB.

#### Interface Definition
```python
class AnalyticsServiceInterface:
    async def process_detections(self, detections: List[VehicleDetection]) -> AnalyticsResult
    async def evaluate_traffic_rules(self, vehicle_data: VehicleTrajectory) -> List[RuleViolation]
    async def calculate_traffic_metrics(self, time_range: TimeRange) -> TrafficMetrics
    async def detect_anomalies(self, traffic_patterns: TrafficData) -> List[Anomaly]
```

#### Data Models
```python
class VehicleDetection(BaseModel):
    detection_id: str
    camera_id: str
    vehicle_id: Optional[str]
    bounding_box: BoundingBox
    confidence: float
    timestamp: datetime
    
class TrafficMetrics(BaseModel):
    camera_id: str
    time_period: TimeRange
    vehicle_count: int
    average_speed: float
    congestion_level: CongestionLevel
```

#### Implementation Checklist
- [ ] Create rule engine for traffic violations
- [ ] Implement speed calculation algorithms
- [ ] Add trajectory analysis
- [ ] Create anomaly detection using ML
- [ ] Implement TimescaleDB integration
- [ ] Add real-time metrics calculation
- [ ] Create alert generation system
- [ ] Add performance optimization

#### Testing Requirements
- Unit tests for rule evaluation logic
- Integration tests with TimescaleDB
- Performance tests with high-volume data
- Accuracy tests for speed calculations
- End-to-end analytics pipeline tests

#### Performance Targets
- Rule evaluation throughput > 10,000 events/second
- Real-time analytics processing < 50ms
- Alert generation latency < 200ms
- 99.5% accuracy for traffic metrics

### 3.3 Alert Service Implementation

#### Overview
The Alert Service handles incident detection, notification distribution, and escalation management.

#### Interface Definition
```python
class AlertServiceInterface:
    async def create_alert(self, incident: IncidentData) -> AlertId
    async def process_alert(self, alert_id: AlertId) -> ProcessingResult
    async def distribute_notifications(self, alert: Alert) -> NotificationResults
    async def handle_escalation(self, alert_id: AlertId) -> EscalationResult
```

#### Data Models
```python
class Alert(BaseModel):
    alert_id: str
    incident_type: IncidentType
    severity: AlertSeverity
    camera_id: str
    timestamp: datetime
    description: str
    status: AlertStatus
    
class NotificationChannel(BaseModel):
    channel_type: ChannelType
    endpoint: str
    priority: int
    escalation_timeout: int
```

#### Implementation Checklist
- [ ] Implement alert prioritization logic
- [ ] Create notification routing system
- [ ] Add external system integration (emergency services)
- [ ] Implement escalation workflows
- [ ] Create alert acknowledgment handling
- [ ] Add notification delivery tracking
- [ ] Implement alert history and audit
- [ ] Create dashboard integration

#### Testing Requirements
- Unit tests for alert processing logic
- Integration tests with external systems
- Load tests for high-volume alerts
- Escalation workflow testing
- Notification delivery verification

#### Performance Targets
- Alert creation latency < 100ms
- Notification delivery < 200ms
- 99.9% notification delivery success
- Support 10,000+ alerts per hour

### 3.4 Model Registry Implementation

#### Overview
The Model Registry manages ML model versions, deployment, and performance tracking using MinIO storage.

#### Interface Definition
```python
class ModelRegistryInterface:
    async def register_model(self, model: ModelArtifact) -> ModelVersion
    async def deploy_model(self, model_id: str, stage: DeploymentStage) -> DeploymentResult
    async def get_active_model(self, model_type: ModelType) -> ModelVersion
    async def promote_model(self, model_id: str, target_stage: DeploymentStage) -> bool
```

#### Data Models
```python
class ModelArtifact(BaseModel):
    model_id: str
    version: str
    model_type: ModelType
    framework: MLFramework
    metrics: ModelMetrics
    artifacts_path: str
    
class DeploymentStage(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
```

#### Implementation Checklist
- [ ] Create model storage in MinIO
- [ ] Implement semantic versioning
- [ ] Add model validation pipeline
- [ ] Create deployment automation
- [ ] Implement A/B testing framework
- [ ] Add model performance monitoring
- [ ] Create rollback mechanisms
- [ ] Add model lineage tracking

### 3.5 Threat Detection Engine Implementation

#### Overview
The Threat Detection Engine provides real-time security monitoring and anomaly detection capabilities.

#### Interface Definition
```python
class ThreatDetectionInterface:
    async def analyze_behavior(self, behavior_data: BehaviorPattern) -> ThreatAssessment
    async def detect_anomalies(self, traffic_data: TrafficData) -> List[Anomaly]
    async def process_security_event(self, event: SecurityEvent) -> ThreatResponse
    async def update_threat_models(self, training_data: TrainingData) -> bool
```

#### Data Models
```python
class ThreatAssessment(BaseModel):
    threat_level: ThreatLevel
    confidence: float
    threat_type: ThreatType
    recommended_action: str
    evidence: List[Evidence]
    
class SecurityEvent(BaseModel):
    event_id: str
    event_type: SecurityEventType
    timestamp: datetime
    source: str
    severity: SecuritySeverity
```

#### Implementation Checklist
- [ ] Implement ML-based anomaly detection
- [ ] Create behavioral analysis algorithms
- [ ] Add threat intelligence integration
- [ ] Implement automated response system
- [ ] Create security dashboard
- [ ] Add incident investigation tools
- [ ] Implement threat model updates
- [ ] Create compliance reporting

---

## 4. Service Implementation Details

### 4.1 Streaming Service (gRPC, Camera Connections)

#### Architecture Pattern
- **Pattern**: Producer-Consumer with Quality Gates
- **Technology**: Python gRPC, OpenCV, Redis
- **Deployment**: Kubernetes with GPU affinity

#### Key Components
```python
class StreamingDataProcessor:
    """High-throughput video stream processing with quality validation."""
    
    def __init__(self, redis_client: Redis, config: StreamingConfig):
        self.redis = redis_client
        self.quality_validator = QualityValidator()
        self.frame_queue = FrameQueue(redis_client)
        
    async def process_stream(self, camera_stream: CameraStream) -> None:
        async for frame in camera_stream:
            quality_score = await self.quality_validator.validate(frame)
            if quality_score > self.config.quality_threshold:
                await self.frame_queue.enqueue(frame)
```

#### gRPC Service Definition
```protobuf
service StreamingService {
    rpc RegisterCamera(CameraRegistrationRequest) returns (CameraRegistrationResponse);
    rpc ProcessFrameStream(stream FrameData) returns (stream ProcessedFrame);
    rpc GetStreamHealth(StreamHealthRequest) returns (StreamHealthResponse);
}
```

#### Implementation Steps
1. **Week 1**: Implement gRPC server and basic frame processing
2. **Week 2**: Add quality validation and Redis integration
3. **Week 3**: Implement error handling and monitoring
4. **Week 4**: Performance optimization and load testing

### 4.2 Vision Engine (YOLO11, PyTorch)

#### Architecture Pattern
- **Pattern**: Pipeline with Adaptive Batching
- **Technology**: PyTorch, TensorRT, CUDA
- **Status**: ✅ Already Implemented

#### Current Implementation Status
The Vision Engine is already well-implemented with:
- ✅ Adaptive Batcher for dynamic batch optimization
- ✅ GPU Preprocessor for CUDA-accelerated processing
- ✅ Memory Pool Manager for efficient GPU memory
- ✅ TensorRT Optimizer for model optimization
- ✅ Production Monitoring for performance tracking

#### Integration Requirements
- Enhance integration with Streaming Service
- Add Model Registry connection
- Improve metrics collection for TimescaleDB

### 4.3 Analytics Service (Traffic Metrics)

#### Architecture Pattern
- **Pattern**: Event-Driven Analytics with Rule Engine
- **Technology**: Python, TimescaleDB, Apache Kafka
- **Deployment**: Horizontally scalable workers

#### Core Components
```python
class TrafficAnalyzer:
    """Real-time traffic analytics with rule processing."""
    
    def __init__(self, timeseries_db: TimescaleDB, rule_engine: RuleEngine):
        self.db = timeseries_db
        self.rules = rule_engine
        self.metrics_calculator = TrafficMetricsCalculator()
        
    async def process_vehicle_data(self, vehicle_data: VehicleData) -> AnalyticsResult:
        # Calculate traffic metrics
        metrics = await self.metrics_calculator.calculate(vehicle_data)
        
        # Evaluate traffic rules
        violations = await self.rules.evaluate(vehicle_data)
        
        # Store in time-series database
        await self.db.store_metrics(metrics)
        
        return AnalyticsResult(metrics=metrics, violations=violations)
```

#### Rule Engine Implementation
```python
class RuleEngine:
    """Traffic rule evaluation engine with configurable rules."""
    
    def __init__(self, rule_config: RuleConfig):
        self.rules = self._load_rules(rule_config)
        self.violation_detector = ViolationDetector()
    
    async def evaluate(self, vehicle_data: VehicleData) -> List[RuleViolation]:
        violations = []
        for rule in self.rules:
            if await rule.evaluate(vehicle_data):
                violation = self.violation_detector.create_violation(rule, vehicle_data)
                violations.append(violation)
        return violations
```

#### Implementation Steps
1. **Week 1**: Design and implement rule engine
2. **Week 2**: Create traffic metrics calculator
3. **Week 3**: Add TimescaleDB integration
4. **Week 4**: Implement real-time analytics pipeline

### 4.4 Alert Service (Incident Detection)

#### Architecture Pattern
- **Pattern**: Event-Driven Notification with Escalation
- **Technology**: Python, Kafka, External APIs
- **Deployment**: High-availability with message durability

#### Core Components
```python
class AlertManager:
    """Incident detection and alert management system."""
    
    def __init__(self, notification_service: NotificationService, escalation_manager: EscalationManager):
        self.notifications = notification_service
        self.escalation = escalation_manager
        self.alert_prioritizer = AlertPrioritizer()
        
    async def process_incident(self, incident: IncidentData) -> AlertResult:
        # Create and prioritize alert
        alert = await self._create_alert(incident)
        priority = await self.alert_prioritizer.prioritize(alert)
        
        # Send notifications
        notification_result = await self.notifications.send(alert, priority)
        
        # Schedule escalation if needed
        if alert.severity >= AlertSeverity.HIGH:
            await self.escalation.schedule(alert)
            
        return AlertResult(alert=alert, notifications=notification_result)
```

#### Notification System
```python
class NotificationService:
    """Multi-channel notification distribution."""
    
    def __init__(self):
        self.channels = {
            ChannelType.EMAIL: EmailNotifier(),
            ChannelType.SMS: SMSNotifier(),
            ChannelType.API: APINotifier(),
            ChannelType.SSE: SSENotifier()
        }
    
    async def send(self, alert: Alert, priority: Priority) -> NotificationResults:
        results = []
        channels = self._select_channels(alert.severity, priority)
        
        for channel in channels:
            result = await self.channels[channel].send(alert)
            results.append(result)
            
        return NotificationResults(results)
```

### 4.5 Authentication Service (JWT, RBAC)

#### Architecture Pattern
- **Pattern**: JWT-based with Role-Based Access Control
- **Technology**: FastAPI, PostgreSQL, Redis, JWT
- **Security**: Multi-factor authentication, session management

#### Core Components
```python
class AuthenticationService:
    """JWT-based authentication with MFA support."""
    
    def __init__(self, db: Database, cache: Redis, jwt_manager: JWTManager):
        self.db = db
        self.cache = cache
        self.jwt = jwt_manager
        self.mfa = MFAService()
        
    async def authenticate(self, credentials: UserCredentials) -> AuthResult:
        # Verify credentials
        user = await self._verify_credentials(credentials)
        if not user:
            return AuthResult(success=False, error="Invalid credentials")
            
        # Check MFA if required
        if user.mfa_enabled:
            mfa_challenge = await self.mfa.create_challenge(user)
            return AuthResult(success=False, mfa_challenge=mfa_challenge)
            
        # Generate JWT token
        token = await self.jwt.create_token(user)
        await self.cache.store_session(user.id, token)
        
        return AuthResult(success=True, token=token, user=user)
```

#### RBAC Implementation
```python
class RoleBasedAccessControl:
    """Fine-grained permission system with role inheritance."""
    
    def __init__(self, db: Database, cache: Redis):
        self.db = db
        self.cache = cache
        
    async def check_permission(self, user_id: str, resource: str, action: str) -> bool:
        # Check cached permissions first
        cache_key = f"permissions:{user_id}:{resource}:{action}"
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result == "true"
            
        # Query database for permissions
        permissions = await self.db.get_user_permissions(user_id)
        has_permission = self._evaluate_permissions(permissions, resource, action)
        
        # Cache result
        await self.cache.setex(cache_key, 300, "true" if has_permission else "false")
        
        return has_permission
```

### 4.6 API Gateway (FastAPI Routes)

#### Current Implementation Status
The API Gateway is already well-implemented with:
- ✅ FastAPI Application with proper routing
- ✅ Middleware stack for CORS, security, logging
- ✅ Dependency injection system
- ✅ Multiple routers (auth, camera, analytics)
- ✅ SSE Broadcaster for real-time events

#### Enhancement Requirements
```python
class APIGateway:
    """Enhanced API Gateway with advanced features."""
    
    def __init__(self):
        self.app = FastAPI(title="ITS Camera AI API")
        self.rate_limiter = RateLimiter()
        self.circuit_breaker = CircuitBreaker()
        
    def setup_middleware(self):
        # Add rate limiting middleware
        self.app.add_middleware(
            RateLimitMiddleware,
            rate_limiter=self.rate_limiter
        )
        
        # Add circuit breaker middleware
        self.app.add_middleware(
            CircuitBreakerMiddleware,
            circuit_breaker=self.circuit_breaker
        )
```

#### Additional Features to Implement
- API versioning strategy
- Request/response transformation
- Advanced rate limiting
- Circuit breaker patterns
- API analytics and monitoring

---

## 5. Data Flow Implementation

### 5.1 Message Schemas

#### Camera Stream Processing Messages
```python
class CameraStreamMessage(BaseModel):
    message_id: str = Field(..., description="Unique message identifier")
    camera_id: str = Field(..., description="Camera source identifier")
    frame_id: str = Field(..., description="Frame sequence identifier")
    timestamp: datetime = Field(..., description="Frame capture timestamp")
    frame_data: bytes = Field(..., description="Base64 encoded frame data")
    metadata: FrameMetadata = Field(..., description="Frame processing metadata")

class FrameMetadata(BaseModel):
    resolution: Tuple[int, int]
    quality_score: float
    compression_ratio: float
    processing_stage: ProcessingStage
```

#### Vehicle Detection Messages
```python
class VehicleDetectionMessage(BaseModel):
    detection_id: str = Field(..., description="Unique detection identifier")
    camera_id: str = Field(..., description="Source camera")
    frame_id: str = Field(..., description="Source frame")
    timestamp: datetime = Field(..., description="Detection timestamp")
    detections: List[VehicleDetection] = Field(..., description="Detected vehicles")
    inference_metadata: InferenceMetadata = Field(..., description="ML inference data")

class VehicleDetection(BaseModel):
    vehicle_id: Optional[str] = Field(None, description="Tracked vehicle ID")
    bounding_box: BoundingBox = Field(..., description="Vehicle location")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    vehicle_type: VehicleType = Field(..., description="Vehicle classification")
    speed: Optional[float] = Field(None, description="Calculated speed in km/h")
```

#### Alert and Incident Messages
```python
class AlertMessage(BaseModel):
    alert_id: str = Field(..., description="Unique alert identifier")
    incident_type: IncidentType = Field(..., description="Type of incident")
    severity: AlertSeverity = Field(..., description="Alert severity level")
    camera_id: str = Field(..., description="Source camera")
    location: Optional[GeographicLocation] = Field(None, description="Geographic location")
    timestamp: datetime = Field(..., description="Incident timestamp")
    description: str = Field(..., description="Incident description")
    evidence: List[Evidence] = Field(default_factory=list, description="Supporting evidence")
    
class Evidence(BaseModel):
    evidence_type: EvidenceType
    reference_id: str
    description: str
    confidence: float
```

### 5.2 Event Definitions

#### System Events
```python
class SystemEventType(str, Enum):
    CAMERA_CONNECTED = "camera.connected"
    CAMERA_DISCONNECTED = "camera.disconnected"
    STREAM_STARTED = "stream.started"
    STREAM_STOPPED = "stream.stopped"
    FRAME_PROCESSED = "frame.processed"
    INFERENCE_COMPLETED = "inference.completed"
    ALERT_CREATED = "alert.created"
    ALERT_RESOLVED = "alert.resolved"

class SystemEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: SystemEventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_service: str
    correlation_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

#### Business Events
```python
class BusinessEventType(str, Enum):
    TRAFFIC_VIOLATION_DETECTED = "traffic.violation.detected"
    SPEED_LIMIT_EXCEEDED = "traffic.speed.exceeded"
    LANE_VIOLATION_DETECTED = "traffic.lane.violation"
    UNAUTHORIZED_AREA_ACCESS = "security.unauthorized.access"
    VEHICLE_TRACKING_STARTED = "tracking.vehicle.started"
    VEHICLE_TRACKING_LOST = "tracking.vehicle.lost"

class BusinessEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: BusinessEventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    camera_id: str
    vehicle_id: Optional[str] = None
    severity: EventSeverity = EventSeverity.MEDIUM
    data: Dict[str, Any] = Field(default_factory=dict)
    requires_alert: bool = False
```

### 5.3 API Contracts

#### RESTful API Specifications
```python
# Camera Management API
class CameraAPI:
    @app.post("/cameras", response_model=CameraResponse, status_code=201)
    async def create_camera(self, camera: CameraCreateRequest) -> CameraResponse:
        """Register a new camera in the system."""
        
    @app.get("/cameras/{camera_id}", response_model=CameraResponse)
    async def get_camera(self, camera_id: str) -> CameraResponse:
        """Retrieve camera information and status."""
        
    @app.put("/cameras/{camera_id}", response_model=CameraResponse)
    async def update_camera(self, camera_id: str, camera: CameraUpdateRequest) -> CameraResponse:
        """Update camera configuration."""
        
    @app.delete("/cameras/{camera_id}", status_code=204)
    async def delete_camera(self, camera_id: str) -> None:
        """Remove camera from the system."""

# Analytics API
class AnalyticsAPI:
    @app.get("/analytics/traffic-metrics", response_model=TrafficMetricsResponse)
    async def get_traffic_metrics(
        self,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        aggregation: AggregationType = AggregationType.HOURLY
    ) -> TrafficMetricsResponse:
        """Retrieve traffic analytics data."""
        
    @app.get("/analytics/violations", response_model=ViolationsResponse)
    async def get_violations(
        self,
        camera_id: Optional[str] = None,
        violation_type: Optional[ViolationType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> ViolationsResponse:
        """Retrieve traffic violation data."""
```

#### gRPC Service Contracts
```protobuf
// Streaming Service
service StreamingService {
  // Register a camera for stream processing
  rpc RegisterCamera(CameraRegistrationRequest) returns (CameraRegistrationResponse);
  
  // Process continuous frame stream
  rpc ProcessFrameStream(stream FrameData) returns (stream ProcessedFrameResponse);
  
  // Get stream health and performance metrics
  rpc GetStreamHealth(StreamHealthRequest) returns (StreamHealthResponse);
  
  // Control stream processing (start/stop/pause)
  rpc ControlStream(StreamControlRequest) returns (StreamControlResponse);
}

// Vision Engine Service
service VisionEngineService {
  // Process single frame or batch
  rpc ProcessFrame(FrameProcessingRequest) returns (FrameProcessingResponse);
  
  // Process frame batch for optimal GPU utilization
  rpc ProcessBatch(BatchProcessingRequest) returns (BatchProcessingResponse);
  
  // Get inference engine status and performance
  rpc GetEngineStatus(EngineStatusRequest) returns (EngineStatusResponse);
  
  // Update model configuration
  rpc UpdateModel(ModelUpdateRequest) returns (ModelUpdateResponse);
}
```

### 5.4 Database Schemas

#### PostgreSQL Schema Design
```sql
-- Users and Authentication
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE
);

-- Role-Based Access Control
CREATE TABLE roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource VARCHAR(100) NOT NULL,
    action VARCHAR(50) NOT NULL,
    description TEXT,
    UNIQUE(resource, action)
);

CREATE TABLE role_permissions (
    role_id UUID REFERENCES roles(id) ON DELETE CASCADE,
    permission_id UUID REFERENCES permissions(id) ON DELETE CASCADE,
    PRIMARY KEY (role_id, permission_id)
);

CREATE TABLE user_roles (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID REFERENCES roles(id) ON DELETE CASCADE,
    assigned_at TIMESTAMPTZ DEFAULT NOW(),
    assigned_by UUID REFERENCES users(id),
    PRIMARY KEY (user_id, role_id)
);

-- Camera Management
CREATE TABLE cameras (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    stream_url VARCHAR(500) NOT NULL,
    location JSONB,
    configuration JSONB NOT NULL DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Incident Management
CREATE TABLE incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id VARCHAR(100) UNIQUE NOT NULL,
    incident_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    camera_id VARCHAR(100) REFERENCES cameras(camera_id),
    location JSONB,
    description TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'OPEN',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    resolved_by UUID REFERENCES users(id)
);

-- Audit Logging
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(255),
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);
```

#### TimescaleDB Schema for Time-Series Data
```sql
-- Performance Metrics
CREATE TABLE performance_metrics (
    time TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    service_name VARCHAR(100) NOT NULL,
    instance_id VARCHAR(100),
    tags JSONB DEFAULT '{}'
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('performance_metrics', 'time', chunk_time_interval => INTERVAL '1 hour');

-- Traffic Analytics
CREATE TABLE traffic_metrics (
    time TIMESTAMPTZ NOT NULL,
    camera_id VARCHAR(100) NOT NULL,
    vehicle_count INTEGER NOT NULL DEFAULT 0,
    average_speed DOUBLE PRECISION,
    traffic_density DOUBLE PRECISION,
    congestion_level VARCHAR(20),
    metadata JSONB DEFAULT '{}'
);

SELECT create_hypertable('traffic_metrics', 'time', chunk_time_interval => INTERVAL '1 hour');

-- Vehicle Detections (High Volume)
CREATE TABLE vehicle_detections (
    time TIMESTAMPTZ NOT NULL,
    detection_id VARCHAR(100) NOT NULL,
    camera_id VARCHAR(100) NOT NULL,
    vehicle_id VARCHAR(100),
    vehicle_type VARCHAR(50),
    confidence DOUBLE PRECISION,
    bounding_box JSONB,
    speed DOUBLE PRECISION,
    metadata JSONB DEFAULT '{}'
);

SELECT create_hypertable('vehicle_detections', 'time', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes for common queries
CREATE INDEX idx_performance_metrics_service_time ON performance_metrics (service_name, time DESC);
CREATE INDEX idx_traffic_metrics_camera_time ON traffic_metrics (camera_id, time DESC);
CREATE INDEX idx_vehicle_detections_camera_time ON vehicle_detections (camera_id, time DESC);
CREATE INDEX idx_vehicle_detections_vehicle_time ON vehicle_detections (vehicle_id, time DESC) WHERE vehicle_id IS NOT NULL;
```

### 5.5 Cache Strategies

#### Redis Cache Architecture
```python
class CacheStrategy:
    """Multi-tier caching strategy implementation."""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.local_cache = LocalCache(max_size=1000)
        
    # L1 Cache: In-Memory (5-second TTL)
    async def get_l1_cache(self, key: str) -> Optional[Any]:
        return self.local_cache.get(key)
    
    async def set_l1_cache(self, key: str, value: Any, ttl: int = 5) -> None:
        self.local_cache.set(key, value, ttl)
    
    # L2 Cache: Redis (5-minute TTL)
    async def get_l2_cache(self, key: str) -> Optional[Any]:
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None
    
    async def set_l2_cache(self, key: str, value: Any, ttl: int = 300) -> None:
        await self.redis.setex(key, ttl, json.dumps(value, cls=DateTimeEncoder))
    
    # Cache patterns for different data types
    async def cache_user_session(self, user_id: str, session_data: dict) -> None:
        key = f"session:{user_id}"
        await self.set_l2_cache(key, session_data, ttl=3600)  # 1 hour
    
    async def cache_camera_config(self, camera_id: str, config: dict) -> None:
        key = f"camera:config:{camera_id}"
        await self.set_l2_cache(key, config, ttl=1800)  # 30 minutes
    
    async def cache_inference_result(self, frame_id: str, result: dict) -> None:
        key = f"inference:{frame_id}"
        await self.set_l1_cache(key, result, ttl=30)  # 30 seconds
```

#### Cache Invalidation Strategy
```python
class CacheInvalidation:
    """Intelligent cache invalidation system."""
    
    def __init__(self, cache: CacheStrategy, event_bus: EventBus):
        self.cache = cache
        self.event_bus = event_bus
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Set up event-driven cache invalidation."""
        self.event_bus.subscribe("camera.updated", self._invalidate_camera_cache)
        self.event_bus.subscribe("user.role_changed", self._invalidate_user_cache)
        self.event_bus.subscribe("model.deployed", self._invalidate_model_cache)
    
    async def _invalidate_camera_cache(self, event: SystemEvent):
        camera_id = event.payload.get("camera_id")
        if camera_id:
            await self.cache.redis.delete(f"camera:*:{camera_id}")
    
    async def _invalidate_user_cache(self, event: SystemEvent):
        user_id = event.payload.get("user_id")
        if user_id:
            await self.cache.redis.delete(f"user:*:{user_id}")
            await self.cache.redis.delete(f"session:{user_id}")
```

---

## 6. Integration Points

### 6.1 Internal Service Communication

#### gRPC Service Mesh
```python
class ServiceMeshClient:
    """Centralized service communication with circuit breakers."""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.registry = service_registry
        self.circuit_breakers = {}
        self.load_balancers = {}
    
    async def call_service(
        self,
        service_name: str,
        method: str,
        request: Message,
        timeout: float = 30.0
    ) -> Message:
        # Get service instance with load balancing
        instance = await self.load_balancers[service_name].get_instance()
        
        # Apply circuit breaker pattern
        circuit_breaker = self.circuit_breakers.get(service_name)
        if circuit_breaker and circuit_breaker.is_open():
            raise CircuitBreakerOpenError(f"Circuit breaker open for {service_name}")
        
        try:
            # Create gRPC channel
            channel = grpc.aio.insecure_channel(f"{instance.host}:{instance.port}")
            
            # Make the call with timeout
            response = await asyncio.wait_for(
                self._make_grpc_call(channel, method, request),
                timeout=timeout
            )
            
            # Record success
            if circuit_breaker:
                circuit_breaker.record_success()
                
            return response
            
        except Exception as e:
            # Record failure
            if circuit_breaker:
                circuit_breaker.record_failure()
            raise ServiceCallError(f"Failed to call {service_name}.{method}: {e}")
        finally:
            await channel.close()
```

#### Service Discovery Implementation
```python
class ServiceRegistry:
    """Service discovery with health checking."""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.health_checker = HealthChecker()
        
    async def register_service(
        self,
        service_name: str,
        instance: ServiceInstance
    ) -> None:
        """Register service instance."""
        key = f"services:{service_name}"
        instance_data = {
            "host": instance.host,
            "port": instance.port,
            "health_check_url": instance.health_check_url,
            "registered_at": datetime.utcnow().isoformat(),
            "status": "healthy"
        }
        
        await self.redis.hset(key, instance.id, json.dumps(instance_data))
        
        # Start health checking
        await self.health_checker.start_monitoring(instance)
    
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Discover healthy service instances."""
        key = f"services:{service_name}"
        instances_data = await self.redis.hgetall(key)
        
        healthy_instances = []
        for instance_id, instance_json in instances_data.items():
            instance_data = json.loads(instance_json)
            if instance_data["status"] == "healthy":
                healthy_instances.append(
                    ServiceInstance(
                        id=instance_id,
                        host=instance_data["host"],
                        port=instance_data["port"],
                        health_check_url=instance_data["health_check_url"]
                    )
                )
        
        return healthy_instances
```

### 6.2 External System Interfaces

#### Camera Integration Layer
```python
class CameraIntegrationService:
    """Universal camera integration supporting multiple protocols."""
    
    def __init__(self):
        self.protocol_handlers = {
            StreamProtocol.RTSP: RTSPHandler(),
            StreamProtocol.WEBRTC: WebRTCHandler(),
            StreamProtocol.HTTP: HTTPHandler(),
            StreamProtocol.ONVIF: ONVIFHandler()
        }
        self.auth_manager = CameraAuthManager()
    
    async def connect_camera(self, camera_config: CameraConfig) -> CameraConnection:
        """Establish connection to camera with appropriate protocol."""
        protocol = camera_config.protocol
        handler = self.protocol_handlers.get(protocol)
        
        if not handler:
            raise UnsupportedProtocolError(f"Protocol {protocol} not supported")
        
        # Authenticate with camera
        auth_credentials = await self.auth_manager.get_credentials(camera_config.camera_id)
        
        # Establish connection
        connection = await handler.connect(camera_config, auth_credentials)
        
        # Validate stream
        await self._validate_stream(connection)
        
        return connection
    
    async def _validate_stream(self, connection: CameraConnection) -> None:
        """Validate camera stream quality and configuration."""
        test_frame = await connection.get_frame()
        
        if test_frame is None:
            raise CameraConnectionError("Unable to receive frames from camera")
        
        # Validate frame properties
        if test_frame.width < 640 or test_frame.height < 480:
            raise CameraConfigurationError("Camera resolution too low")
        
        if test_frame.fps < 10:
            raise CameraConfigurationError("Camera FPS too low")
```

#### Emergency Services Integration
```python
class EmergencyServicesIntegration:
    """Integration with external emergency services."""
    
    def __init__(self, config: EmergencyServicesConfig):
        self.config = config
        self.notification_clients = {
            EmergencyServiceType.POLICE: PoliceAPIClient(config.police_api),
            EmergencyServiceType.FIRE: FireDepartmentAPIClient(config.fire_api),
            EmergencyServiceType.MEDICAL: MedicalServicesAPIClient(config.medical_api)
        }
    
    async def send_emergency_alert(self, alert: EmergencyAlert) -> EmergencyResponse:
        """Send alert to appropriate emergency services."""
        service_type = self._determine_service_type(alert)
        client = self.notification_clients[service_type]
        
        # Prepare alert data
        emergency_data = EmergencyData(
            incident_id=alert.alert_id,
            location=alert.location,
            incident_type=alert.incident_type,
            severity=alert.severity,
            description=alert.description,
            timestamp=alert.timestamp,
            evidence_urls=await self._prepare_evidence_urls(alert.evidence)
        )
        
        # Send to emergency service
        try:
            response = await client.submit_incident(emergency_data)
            
            # Log the interaction
            await self._log_emergency_interaction(alert, response)
            
            return response
            
        except Exception as e:
            await self._handle_emergency_service_error(alert, e)
            raise EmergencyServiceError(f"Failed to contact emergency services: {e}")
    
    def _determine_service_type(self, alert: EmergencyAlert) -> EmergencyServiceType:
        """Determine which emergency service to contact."""
        incident_mappings = {
            IncidentType.ACCIDENT: EmergencyServiceType.POLICE,
            IncidentType.FIRE: EmergencyServiceType.FIRE,
            IncidentType.MEDICAL_EMERGENCY: EmergencyServiceType.MEDICAL,
            IncidentType.SECURITY_THREAT: EmergencyServiceType.POLICE
        }
        
        return incident_mappings.get(alert.incident_type, EmergencyServiceType.POLICE)
```

#### Traffic Management System Integration
```python
class TrafficManagementIntegration:
    """Integration with external traffic management systems."""
    
    def __init__(self, tms_config: TMSConfig):
        self.tms_client = TMSAPIClient(tms_config)
        self.data_transformer = TMSDataTransformer()
    
    async def share_traffic_data(self, traffic_data: TrafficData) -> TMSResponse:
        """Share real-time traffic data with TMS."""
        # Transform data to TMS format
        tms_data = await self.data_transformer.transform(traffic_data)
        
        # Send to TMS
        response = await self.tms_client.update_traffic_data(tms_data)
        
        return response
    
    async def receive_signal_controls(self, camera_id: str) -> SignalControlData:
        """Receive traffic signal control information."""
        controls = await self.tms_client.get_signal_controls(camera_id)
        return controls
```

### 6.3 Event Bus Implementation

#### Apache Kafka Event Streaming
```python
class KafkaEventBus:
    """Kafka-based event bus for reliable event streaming."""
    
    def __init__(self, kafka_config: KafkaConfig):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=kafka_config.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, cls=DateTimeEncoder).encode('utf-8')
        )
        self.consumer_groups = {}
        self.topic_schemas = self._load_topic_schemas()
    
    async def publish_event(self, topic: str, event: SystemEvent) -> None:
        """Publish event to Kafka topic."""
        # Validate event against schema
        schema = self.topic_schemas.get(topic)
        if schema:
            schema.validate(event.dict())
        
        # Add metadata
        event_data = event.dict()
        event_data['__metadata__'] = {
            'published_at': datetime.utcnow().isoformat(),
            'schema_version': schema.version if schema else '1.0',
            'producer_service': self._get_service_name()
        }
        
        # Publish to Kafka
        await self.producer.send(topic, event_data)
    
    async def subscribe_to_events(
        self,
        topic: str,
        handler: Callable[[SystemEvent], Awaitable[None]],
        consumer_group: str = "default"
    ) -> None:
        """Subscribe to events from Kafka topic."""
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self.producer._bootstrap_servers,
            group_id=consumer_group,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        await consumer.start()
        
        try:
            async for message in consumer:
                try:
                    event_data = message.value
                    event = SystemEvent(**event_data)
                    await handler(event)
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
                    # Send to dead letter queue
                    await self._send_to_dead_letter_queue(message, e)
        finally:
            await consumer.stop()
```

### 6.4 API Versioning Strategy

#### Semantic Versioning Implementation
```python
class APIVersionManager:
    """API versioning with backward compatibility."""
    
    def __init__(self):
        self.version_handlers = {
            "v1": APIv1Handler(),
            "v2": APIv2Handler(),
            "v3": APIv3Handler()
        }
        self.compatibility_matrix = self._load_compatibility_matrix()
    
    def get_handler(self, version: str) -> APIHandler:
        """Get appropriate API handler for version."""
        if version not in self.version_handlers:
            # Default to latest compatible version
            compatible_version = self._find_compatible_version(version)
            if not compatible_version:
                raise UnsupportedAPIVersionError(f"API version {version} not supported")
            version = compatible_version
        
        return self.version_handlers[version]
    
    def _find_compatible_version(self, requested_version: str) -> Optional[str]:
        """Find compatible version for unsupported version."""
        # Implement semantic versioning compatibility logic
        major, minor, patch = self._parse_version(requested_version)
        
        # Look for compatible versions (same major version)
        for available_version in self.version_handlers.keys():
            av_major, av_minor, av_patch = self._parse_version(available_version)
            if major == av_major and minor <= av_minor:
                return available_version
        
        return None
```

---

## 7. Testing Strategy

### 7.1 Unit Test Requirements

#### Test Coverage Requirements by Component
```python
# Testing Configuration
COVERAGE_REQUIREMENTS = {
    "core_services": 95,      # Streaming, Vision, Analytics
    "api_layer": 90,          # API Gateway, Routers
    "ml_pipeline": 95,        # Inference, Model Management
    "security": 98,           # Authentication, Authorization
    "data_layer": 85,         # Database, Cache
    "integration": 80         # External integrations
}

# Test Structure
class TestStreamingService:
    """Comprehensive unit tests for Streaming Service."""
    
    @pytest.fixture
    async def streaming_service(self):
        mock_redis = AsyncMock(spec=Redis)
        mock_config = StreamingConfig(quality_threshold=0.8)
        return StreamingDataProcessor(mock_redis, mock_config)
    
    async def test_frame_processing_success(self, streaming_service):
        """Test successful frame processing."""
        # Arrange
        mock_frame = Mock(spec=VideoFrame)
        mock_frame.quality_score = 0.9
        
        # Act
        result = await streaming_service.process_frame(mock_frame)
        
        # Assert
        assert result.success is True
        assert result.quality_score == 0.9
        streaming_service.frame_queue.enqueue.assert_called_once()
    
    async def test_frame_processing_low_quality(self, streaming_service):
        """Test frame processing with low quality."""
        # Arrange
        mock_frame = Mock(spec=VideoFrame)
        mock_frame.quality_score = 0.5
        
        # Act
        result = await streaming_service.process_frame(mock_frame)
        
        # Assert
        assert result.success is False
        assert result.reason == "Quality below threshold"
        streaming_service.frame_queue.enqueue.assert_not_called()
    
    @pytest.mark.parametrize("frame_count,expected_batches", [
        (1, 1), (10, 2), (50, 10), (100, 20)
    ])
    async def test_batch_processing(self, streaming_service, frame_count, expected_batches):
        """Test batch processing with various frame counts."""
        frames = [Mock(spec=VideoFrame) for _ in range(frame_count)]
        
        result = await streaming_service.process_batch(frames)
        
        assert len(result.batches) == expected_batches
```

#### Property-Based Testing
```python
from hypothesis import given, strategies as st

class TestVisionEngineProperties:
    """Property-based tests for Vision Engine."""
    
    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        image_width=st.integers(min_value=224, max_value=1920),
        image_height=st.integers(min_value=224, max_value=1080)
    )
    async def test_inference_output_shape(self, batch_size, image_width, image_height):
        """Test that inference output has correct shape regardless of input."""
        # Create mock input batch
        input_batch = torch.randn(batch_size, 3, image_height, image_width)
        
        # Run inference
        vision_engine = VisionEngine()
        output = await vision_engine.process_batch(input_batch)
        
        # Verify output properties
        assert len(output.detections) <= batch_size * 100  # Max detections per batch
        for detection in output.detections:
            assert 0 <= detection.confidence <= 1
            assert detection.bounding_box.is_valid()
    
    @given(confidence_threshold=st.floats(min_value=0.1, max_value=0.9))
    async def test_confidence_filtering(self, confidence_threshold):
        """Test confidence filtering works correctly."""
        vision_engine = VisionEngine(confidence_threshold=confidence_threshold)
        
        # Mock detections with various confidence scores
        mock_detections = [
            Detection(confidence=confidence_threshold + 0.1),
            Detection(confidence=confidence_threshold - 0.1),
            Detection(confidence=confidence_threshold + 0.05)
        ]
        
        filtered = vision_engine.filter_by_confidence(mock_detections)
        
        # All filtered detections should be above threshold
        assert all(d.confidence >= confidence_threshold for d in filtered)
```

### 7.2 Integration Test Scenarios

#### Service Integration Tests
```python
class TestServiceIntegration:
    """Integration tests for service-to-service communication."""
    
    @pytest.fixture
    async def integrated_services(self):
        """Set up integrated test environment."""
        # Start test containers
        redis_container = await start_redis_container()
        postgres_container = await start_postgres_container()
        
        # Initialize services
        streaming_service = StreamingService(redis_container.url)
        vision_engine = VisionEngine()
        analytics_service = AnalyticsService(postgres_container.url)
        
        await streaming_service.start()
        await analytics_service.start()
        
        yield {
            'streaming': streaming_service,
            'vision': vision_engine,
            'analytics': analytics_service
        }
        
        # Cleanup
        await streaming_service.stop()
        await analytics_service.stop()
        await redis_container.stop()
        await postgres_container.stop()
    
    async def test_complete_frame_processing_pipeline(self, integrated_services):
        """Test complete pipeline from stream to analytics."""
        streaming = integrated_services['streaming']
        vision = integrated_services['vision']
        analytics = integrated_services['analytics']
        
        # Inject test frame
        test_frame = create_test_frame_with_vehicles(vehicle_count=3)
        
        # Process through pipeline
        processed_frame = await streaming.process_frame(test_frame)
        assert processed_frame.success
        
        detections = await vision.process_frame(processed_frame.frame)
        assert len(detections) == 3
        
        analytics_result = await analytics.process_detections(detections)
        assert analytics_result.vehicle_count == 3
        assert analytics_result.traffic_flow > 0
    
    async def test_error_propagation_and_recovery(self, integrated_services):
        """Test error handling across services."""
        streaming = integrated_services['streaming']
        analytics = integrated_services['analytics']
        
        # Simulate analytics service failure
        analytics.simulate_failure(duration=5)
        
        # Process frames during failure
        test_frames = [create_test_frame() for _ in range(10)]
        
        results = []
        for frame in test_frames:
            result = await streaming.process_frame(frame)
            results.append(result)
        
        # Verify graceful degradation
        failed_results = [r for r in results if not r.success]
        assert len(failed_results) > 0  # Some should fail
        assert len(failed_results) < len(results)  # Not all should fail
        
        # Verify recovery after service recovery
        await asyncio.sleep(6)  # Wait for recovery
        
        recovery_frame = create_test_frame()
        recovery_result = await streaming.process_frame(recovery_frame)
        assert recovery_result.success
```

#### Database Integration Tests
```python
class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    @pytest.fixture
    async def db_session(self):
        """Provide test database session."""
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        async_session = async_sessionmaker(engine)
        async with async_session() as session:
            yield session
    
    async def test_camera_crud_operations(self, db_session):
        """Test complete CRUD operations for cameras."""
        camera_service = CameraService(db_session)
        
        # Create
        camera_data = CameraCreateRequest(
            camera_id="test_cam_001",
            name="Test Camera 1",
            stream_url="rtsp://test.example.com/stream1",
            location={"lat": 40.7128, "lng": -74.0060}
        )
        
        created_camera = await camera_service.create_camera(camera_data)
        assert created_camera.camera_id == "test_cam_001"
        
        # Read
        retrieved_camera = await camera_service.get_camera("test_cam_001")
        assert retrieved_camera.name == "Test Camera 1"
        
        # Update
        update_data = CameraUpdateRequest(name="Updated Test Camera")
        updated_camera = await camera_service.update_camera("test_cam_001", update_data)
        assert updated_camera.name == "Updated Test Camera"
        
        # Delete
        await camera_service.delete_camera("test_cam_001")
        
        with pytest.raises(CameraNotFoundError):
            await camera_service.get_camera("test_cam_001")
    
    async def test_transaction_rollback_on_error(self, db_session):
        """Test transaction rollback on database errors."""
        camera_service = CameraService(db_session)
        
        # Start transaction
        async with db_session.begin():
            # Create camera
            camera_data = CameraCreateRequest(
                camera_id="test_cam_002",
                name="Test Camera 2",
                stream_url="rtsp://test.example.com/stream2"
            )
            
            await camera_service.create_camera(camera_data)
            
            # Simulate error that should rollback transaction
            raise DatabaseError("Simulated database error")
        
        # Verify rollback - camera should not exist
        with pytest.raises(CameraNotFoundError):
            await camera_service.get_camera("test_cam_002")
```

### 7.3 Performance Benchmarks

#### Load Testing Framework
```python
class PerformanceBenchmarks:
    """Performance benchmarking for critical system components."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.load_generator = LoadGenerator()
    
    async def benchmark_inference_latency(
        self,
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        iterations: int = 100
    ) -> BenchmarkResults:
        """Benchmark inference latency across different batch sizes."""
        vision_engine = VisionEngine()
        results = {}
        
        for batch_size in batch_sizes:
            latencies = []
            
            for _ in range(iterations):
                # Generate test batch
                test_batch = generate_test_image_batch(batch_size)
                
                # Measure inference time
                start_time = time.perf_counter()
                detections = await vision_engine.process_batch(test_batch)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
            
            results[batch_size] = LatencyMetrics(
                mean=statistics.mean(latencies),
                median=statistics.median(latencies),
                p95=numpy.percentile(latencies, 95),
                p99=numpy.percentile(latencies, 99),
                min=min(latencies),
                max=max(latencies)
            )
        
        return BenchmarkResults(
            test_name="inference_latency",
            results=results,
            target_p95_ms=100  # Target: <100ms p95 latency
        )
    
    async def benchmark_api_throughput(
        self,
        concurrent_users: List[int] = [10, 50, 100, 500, 1000],
        duration_seconds: int = 60
    ) -> BenchmarkResults:
        """Benchmark API throughput under various loads."""
        api_client = TestAPIClient()
        results = {}
        
        for users in concurrent_users:
            # Run load test
            load_test_result = await self.load_generator.run_load_test(
                target_url="http://api-gateway:8000/cameras",
                concurrent_users=users,
                duration=duration_seconds,
                ramp_up_time=10
            )
            
            results[users] = ThroughputMetrics(
                requests_per_second=load_test_result.rps,
                response_time_p95=load_test_result.response_time_p95,
                error_rate=load_test_result.error_rate,
                successful_requests=load_test_result.successful_requests
            )
        
        return BenchmarkResults(
            test_name="api_throughput",
            results=results,
            target_rps=1000  # Target: >1000 RPS
        )
```

#### Memory and Resource Benchmarks
```python
class ResourceBenchmarks:
    """Resource utilization benchmarking."""
    
    async def benchmark_memory_usage(self, test_duration: int = 300) -> MemoryBenchmark:
        """Benchmark memory usage under typical load."""
        # Start resource monitoring
        resource_monitor = ResourceMonitor()
        await resource_monitor.start()
        
        # Simulate typical workload
        workload_simulator = WorkloadSimulator()
        await workload_simulator.run_typical_workload(duration=test_duration)
        
        # Collect metrics
        memory_metrics = await resource_monitor.get_memory_metrics()
        
        return MemoryBenchmark(
            peak_memory_mb=memory_metrics.peak_mb,
            average_memory_mb=memory_metrics.average_mb,
            memory_growth_rate=memory_metrics.growth_rate_mb_per_hour,
            gc_frequency=memory_metrics.gc_collections_per_minute
        )
    
    async def benchmark_gpu_utilization(self) -> GPUBenchmark:
        """Benchmark GPU utilization efficiency."""
        gpu_monitor = GPUMonitor()
        vision_engine = VisionEngine()
        
        # Run inference workload
        test_batches = [generate_test_image_batch(16) for _ in range(100)]
        
        await gpu_monitor.start()
        
        for batch in test_batches:
            await vision_engine.process_batch(batch)
        
        gpu_metrics = await gpu_monitor.get_metrics()
        
        return GPUBenchmark(
            average_utilization=gpu_metrics.average_utilization,
            peak_utilization=gpu_metrics.peak_utilization,
            memory_utilization=gpu_metrics.memory_utilization,
            target_utilization=85  # Target: >85% GPU utilization
        )
```

### 7.4 Security Testing

#### Security Test Suite
```python
class SecurityTests:
    """Comprehensive security testing suite."""
    
    async def test_authentication_security(self):
        """Test authentication security measures."""
        auth_client = TestAuthClient()
        
        # Test 1: Password brute force protection
        with pytest.raises(TooManyAttemptsError):
            for _ in range(6):  # Should block after 5 attempts
                await auth_client.login("user", "wrong_password")
        
        # Test 2: JWT token validation
        invalid_token = "invalid.jwt.token"
        with pytest.raises(InvalidTokenError):
            await auth_client.access_protected_resource(invalid_token)
        
        # Test 3: Session hijacking protection
        valid_token = await auth_client.login("user", "correct_password")
        
        # Simulate request from different IP
        with pytest.raises(SuspiciousActivityError):
            await auth_client.access_protected_resource(
                valid_token, 
                source_ip="different.ip.address"
            )
    
    async def test_input_validation_security(self):
        """Test input validation against injection attacks."""
        api_client = TestAPIClient()
        
        # Test SQL injection attempts
        sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'/**/OR/**/1=1/**/--"
        ]
        
        for payload in sql_injection_payloads:
            response = await api_client.get_camera(camera_id=payload)
            assert response.status_code != 200  # Should be rejected
            assert "error" in response.json()
        
        # Test XSS prevention
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ]
        
        for payload in xss_payloads:
            response = await api_client.create_camera({
                "name": payload,
                "stream_url": "rtsp://test.com"
            })
            # Should sanitize input
            created_camera = response.json()
            assert "<script>" not in created_camera["name"]
    
    async def test_authorization_enforcement(self):
        """Test role-based access control enforcement."""
        # Create users with different roles
        admin_token = await self.create_test_user("admin", ["admin"])
        operator_token = await self.create_test_user("operator", ["operator"])
        viewer_token = await self.create_test_user("viewer", ["viewer"])
        
        api_client = TestAPIClient()
        
        # Test admin access (should succeed)
        response = await api_client.delete_camera("test_cam", token=admin_token)
        assert response.status_code == 204
        
        # Test operator access to admin endpoint (should fail)
        with pytest.raises(ForbiddenError):
            await api_client.delete_camera("test_cam_2", token=operator_token)
        
        # Test viewer access to write endpoint (should fail)
        with pytest.raises(ForbiddenError):
            await api_client.create_camera(
                {"name": "new_cam", "stream_url": "rtsp://test.com"}, 
                token=viewer_token
            )
```

---

## 8. Deployment Strategy

### 8.1 Container Build Process

#### Multi-Stage Docker Builds
```dockerfile
# Production Dockerfile for Vision Engine
FROM nvidia/cuda:12.0-devel-ubuntu22.04 as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    build-essential \
    cmake \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy requirements and install dependencies
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --group ml --group gpu

# Production stage
FROM nvidia/cuda:12.0-runtime-ubuntu22.04 as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    libopencv-core4.5d \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
WORKDIR /app
COPY src/ /app/src/
COPY config/ /app/config/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')"

# Start the service
CMD ["python", "-m", "its_camera_ai.services.vision_engine"]
```

#### Optimized Build Pipeline
```yaml
# .github/workflows/build-and-deploy.yml
name: Build and Deploy ITS Camera AI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: its-camera-ai

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [api-gateway, vision-engine, streaming-service, analytics-service]
    
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/${{ matrix.service }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./docker/${{ matrix.service }}/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64
```

### 8.2 Kubernetes Manifests

#### Production Deployment Configuration
```yaml
# k8s/production/vision-engine-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vision-engine
  namespace: its-camera-ai
  labels:
    app: vision-engine
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: vision-engine
  template:
    metadata:
      labels:
        app: vision-engine
        version: v1
    spec:
      serviceAccountName: vision-engine
      containers:
      - name: vision-engine
        image: ghcr.io/its-camera-ai/vision-engine:latest
        ports:
        - containerPort: 8001
          name: http
        - containerPort: 9001
          name: grpc
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secrets
              key: url
        - name: GPU_MEMORY_LIMIT
          value: "8Gi"
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
            nvidia.com/gpu: 1
          limits:
            cpu: "4"
            memory: "8Gi"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
          readOnly: true
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
      - name: config
        configMap:
          name: vision-engine-config
      nodeSelector:
        accelerator: nvidia-gpu
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Equal"
        value: "present"
        effect: "NoSchedule"
```

#### Horizontal Pod Autoscaler
```yaml
# k8s/production/vision-engine-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vision-engine-hpa
  namespace: its-camera-ai
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vision-engine
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: inference_queue_length
      target:
        type: AverageValue
        averageValue: "5"
  - type: Pods
    pods:
      metric:
        name: gpu_utilization_percent
      target:
        type: AverageValue
        averageValue: "85"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### 8.3 CI/CD Pipeline Stages

#### Pipeline Configuration
```yaml
# .gitlab-ci.yml or similar
stages:
  - test
  - security-scan
  - build
  - deploy-staging
  - integration-test
  - deploy-production

variables:
  DOCKER_DRIVER: overlay2
  KUBERNETES_NAMESPACE: its-camera-ai

# Testing Stage
test:unit:
  stage: test
  image: python:3.12
  services:
    - redis:7-alpine
    - postgres:15-alpine
  script:
    - uv sync --group dev
    - pytest --cov=src/its_camera_ai --cov-report=xml --cov-fail-under=90
    - mypy src/
    - ruff check src/ tests/
    - bandit -r src/
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - coverage.xml

# Security Scanning
security:sast:
  stage: security-scan
  image: securecodewarrior/gitlab-sast:latest
  script:
    - /run-sast-analyzer.sh
  artifacts:
    reports:
      sast: gl-sast-report.json

security:dependency-scan:
  stage: security-scan
  image: python:3.12
  script:
    - uv sync
    - safety check
    - pip-audit
  allow_failure: false

# Build Stage
build:containers:
  stage: build
  image: docker:24-dind
  services:
    - docker:24-dind
  parallel:
    matrix:
      - SERVICE: [api-gateway, vision-engine, streaming-service, analytics-service]
  script:
    - docker build -f docker/$SERVICE/Dockerfile -t $CI_REGISTRY_IMAGE/$SERVICE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE/$SERVICE:$CI_COMMIT_SHA
  rules:
    - if: $CI_COMMIT_BRANCH == "main" || $CI_COMMIT_BRANCH == "develop"

# Staging Deployment
deploy:staging:
  stage: deploy-staging
  image: bitnami/kubectl:latest
  environment:
    name: staging
    url: https://staging-api.its-camera-ai.com
  script:
    - kubectl apply -f k8s/staging/ -n its-camera-ai-staging
    - kubectl set image deployment/vision-engine vision-engine=$CI_REGISTRY_IMAGE/vision-engine:$CI_COMMIT_SHA -n its-camera-ai-staging
    - kubectl rollout status deployment/vision-engine -n its-camera-ai-staging --timeout=300s
  rules:
    - if: $CI_COMMIT_BRANCH == "develop"

# Integration Testing in Staging
integration-test:
  stage: integration-test
  image: python:3.12
  needs:
    - deploy:staging
  script:
    - uv sync --group test
    - pytest tests/integration/ --base-url=https://staging-api.its-camera-ai.com
  rules:
    - if: $CI_COMMIT_BRANCH == "develop"

# Production Deployment
deploy:production:
  stage: deploy-production
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: https://api.its-camera-ai.com
  script:
    - kubectl apply -f k8s/production/ -n its-camera-ai
    - kubectl set image deployment/vision-engine vision-engine=$CI_REGISTRY_IMAGE/vision-engine:$CI_COMMIT_SHA -n its-camera-ai
    - kubectl rollout status deployment/vision-engine -n its-camera-ai --timeout=600s
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
  when: manual
```

### 8.4 Environment Promotion

#### Environment-Specific Configurations
```yaml
# k8s/environments/staging/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: its-camera-ai-staging

resources:
- ../../base

patchesStrategicMerge:
- deployment-patch.yaml
- service-patch.yaml

configMapGenerator:
- name: app-config
  literals:
  - LOG_LEVEL=DEBUG
  - ENABLE_DEBUG_ENDPOINTS=true
  - GPU_MEMORY_LIMIT=4Gi
  - INFERENCE_BATCH_SIZE=8

images:
- name: vision-engine
  newTag: staging-latest

replicas:
- name: vision-engine
  count: 2
```

```yaml
# k8s/environments/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: its-camera-ai

resources:
- ../../base
- hpa.yaml
- network-policies.yaml
- pod-disruption-budget.yaml

patchesStrategicMerge:
- deployment-patch.yaml
- security-patch.yaml

configMapGenerator:
- name: app-config
  literals:
  - LOG_LEVEL=INFO
  - ENABLE_DEBUG_ENDPOINTS=false
  - GPU_MEMORY_LIMIT=8Gi
  - INFERENCE_BATCH_SIZE=16
  - PROMETHEUS_METRICS_ENABLED=true

images:
- name: vision-engine
  newTag: production-v1.2.3

replicas:
- name: vision-engine
  count: 5
```

### 8.5 Rollback Procedures

#### Automated Rollback Strategy
```python
# scripts/rollback_service.py
#!/usr/bin/env python3
"""
Automated rollback script for ITS Camera AI services.
"""

import asyncio
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class ServiceRollback:
    """Automated service rollback with health validation."""
    
    def __init__(self, namespace: str = "its-camera-ai"):
        self.namespace = namespace
        self.kubectl_cmd = ["kubectl", "-n", namespace]
    
    async def rollback_service(
        self,
        service_name: str,
        target_revision: Optional[str] = None,
        validation_timeout: int = 300
    ) -> bool:
        """Rollback service to previous or specific revision."""
        try:
            # Get current deployment status
            current_status = await self._get_deployment_status(service_name)
            print(f"Current status for {service_name}: {current_status}")
            
            # Perform rollback
            if target_revision:
                rollback_cmd = self.kubectl_cmd + [
                    "rollout", "undo", f"deployment/{service_name}",
                    f"--to-revision={target_revision}"
                ]
            else:
                rollback_cmd = self.kubectl_cmd + [
                    "rollout", "undo", f"deployment/{service_name}"
                ]
            
            print(f"Executing rollback: {' '.join(rollback_cmd)}")
            subprocess.run(rollback_cmd, check=True)
            
            # Wait for rollback to complete
            await self._wait_for_rollout(service_name, validation_timeout)
            
            # Validate service health
            health_check_passed = await self._validate_service_health(service_name)
            
            if health_check_passed:
                print(f"✅ Rollback successful for {service_name}")
                await self._notify_rollback_success(service_name, target_revision)
                return True
            else:
                print(f"❌ Rollback health check failed for {service_name}")
                await self._notify_rollback_failure(service_name, target_revision)
                return False
                
        except Exception as e:
            print(f"❌ Rollback failed for {service_name}: {e}")
            await self._notify_rollback_failure(service_name, target_revision, str(e))
            return False
    
    async def _get_deployment_status(self, service_name: str) -> Dict:
        """Get current deployment status."""
        cmd = self.kubectl_cmd + [
            "get", f"deployment/{service_name}",
            "-o", "jsonpath={.status}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"ready_replicas": result.stdout}
    
    async def _wait_for_rollout(self, service_name: str, timeout: int):
        """Wait for deployment rollout to complete."""
        cmd = self.kubectl_cmd + [
            "rollout", "status", f"deployment/{service_name}",
            f"--timeout={timeout}s"
        ]
        
        subprocess.run(cmd, check=True)
    
    async def _validate_service_health(self, service_name: str) -> bool:
        """Validate service health after rollback."""
        # Check pod readiness
        pods_cmd = self.kubectl_cmd + [
            "get", "pods", "-l", f"app={service_name}",
            "-o", "jsonpath={.items[*].status.conditions[?(@.type=='Ready')].status}"
        ]
        
        result = subprocess.run(pods_cmd, capture_output=True, text=True, check=True)
        ready_statuses = result.stdout.split()
        
        if not all(status == "True" for status in ready_statuses):
            return False
        
        # Check service endpoints
        # Implement service-specific health checks
        return await self._check_service_endpoints(service_name)
    
    async def _check_service_endpoints(self, service_name: str) -> bool:
        """Check service-specific health endpoints."""
        health_checks = {
            "vision-engine": "http://vision-engine:8001/health",
            "api-gateway": "http://api-gateway:8000/health",
            "streaming-service": "http://streaming-service:8002/health",
            "analytics-service": "http://analytics-service:8003/health"
        }
        
        if service_name not in health_checks:
            return True  # No specific health check available
        
        # Perform HTTP health check
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(health_checks[service_name], timeout=10) as response:
                    return response.status == 200
        except Exception:
            return False

# Blue-Green Deployment Strategy
class BlueGreenDeployment:
    """Blue-green deployment with automatic rollback."""
    
    def __init__(self, namespace: str = "its-camera-ai"):
        self.namespace = namespace
        self.rollback_manager = ServiceRollback(namespace)
    
    async def deploy_green(
        self,
        service_name: str,
        new_image: str,
        validation_tests: List[str]
    ) -> bool:
        """Deploy to green environment and validate."""
        try:
            # Create green deployment
            await self._create_green_deployment(service_name, new_image)
            
            # Run validation tests
            validation_passed = await self._run_validation_tests(service_name, validation_tests)
            
            if validation_passed:
                # Switch traffic to green
                await self._switch_traffic_to_green(service_name)
                
                # Clean up blue deployment after successful switch
                await asyncio.sleep(300)  # 5-minute grace period
                await self._cleanup_blue_deployment(service_name)
                
                return True
            else:
                # Rollback to blue
                await self._cleanup_green_deployment(service_name)
                return False
                
        except Exception as e:
            print(f"Blue-green deployment failed: {e}")
            await self._cleanup_green_deployment(service_name)
            return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rollback ITS Camera AI services")
    parser.add_argument("service", help="Service name to rollback")
    parser.add_argument("--revision", help="Target revision (optional)")
    parser.add_argument("--namespace", default="its-camera-ai", help="Kubernetes namespace")
    
    args = parser.parse_args()
    
    rollback_manager = ServiceRollback(args.namespace)
    
    success = asyncio.run(rollback_manager.rollback_service(
        args.service, 
        args.revision
    ))
    
    exit(0 if success else 1)
```

---

## Conclusion

This comprehensive implementation plan provides a structured approach to building the ITS Camera AI system based on the architectural diagrams and design documents. The plan prioritizes critical components, defines clear interfaces, and provides actionable guidance for each phase of development.

### Key Success Factors

1. **Phased Approach**: The 6-phase implementation plan ensures steady progress while managing complexity
2. **Clear Interfaces**: Well-defined APIs and contracts enable parallel development
3. **Quality Gates**: Comprehensive testing strategy ensures production readiness
4. **Security First**: Security considerations integrated throughout the implementation
5. **Operational Excellence**: Complete monitoring and deployment strategies

### Next Steps

1. **Week 1**: Begin Phase 1 (Core Infrastructure) implementation
2. **Establish Teams**: Assign teams to specific components based on the matrix
3. **Set up CI/CD**: Implement the deployment pipeline early
4. **Create Development Environment**: Set up local development with containers
5. **Begin Implementation**: Start with P0 components as defined in the matrix

This plan serves as a living document that should be updated as implementation progresses and new requirements emerge. Regular reviews and adjustments will ensure successful delivery of the ITS Camera AI system.

**Total Estimated Timeline**: 14 weeks for complete implementation
**Team Requirements**: 12-15 developers across Backend, ML, DevOps, and Security teams
**Key Milestones**: End of each phase represents a deployable system increment