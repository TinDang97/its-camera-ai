# CLI Backend Integration

This document describes the comprehensive backend integration implemented for the ITS Camera AI CLI application. The integration provides seamless connectivity between the CLI and all backend services including APIs, databases, message queues, and monitoring systems.

## Architecture Overview

The CLI backend integration follows a modular architecture with the following key components:

```
┌────────────────────────────────────────────────────────────┐
│                    Service Orchestrator                    │
│         Coordinates complex multi-service operations      │
├────────────────────────────────────────────────────────────┤
│  API Client  │ Auth Manager │ Database Mgr │ Health Checker  │
│   HTTP/REST   │  JWT & RBAC   │ SQL Queries  │ Component Mon  │
├─────────────┬──────────────┬──────────────┬──────────────┤
│Queue Manager│Event Streamer│Service Disc │Metrics Collect│
│ Redis Queues │   SSE/WebSock  │ Auto-discover │ Prometheus Mon │
└─────────────┴──────────────┴──────────────┴──────────────┘
```

## Core Components

### 1. API Client (`api_client.py`)

**Purpose**: HTTP client for FastAPI backend communication

**Key Features**:
- Connection pooling and keep-alive
- Automatic retry with exponential backoff
- JWT token management
- Circuit breaker pattern
- Response caching
- Request/response logging

**Usage Example**:
```python
async with APIClient() as api_client:
    health = await api_client.get_health()
    cameras = await api_client.list_cameras()
    metrics = await api_client.get_metrics()
```

### 2. Database Manager (`database_manager.py`)

**Purpose**: Direct database operations for CLI commands

**Key Features**:
- AsyncPG connection pooling
- Transaction management with rollback
- Query execution with proper error handling
- Database statistics and maintenance
- Connection health checking
- Data cleanup operations

**Usage Example**:
```python
async with CLIDatabaseManager() as db_manager:
    stats = await db_manager.get_database_stats()
    connectivity = await db_manager.check_connectivity()
    deleted_count = await db_manager.cleanup_old_data("table_name", "timestamp_col", 30)
```

### 3. Authentication Manager (`auth_manager.py`)

**Purpose**: Authentication and authorization for CLI operations

**Key Features**:
- JWT token creation and validation
- User authentication with bcrypt
- Role-based access control (RBAC)
- Permission checking
- Session management
- Token refresh capabilities

**Usage Example**:
```python
async with CLIAuthManager() as auth_manager:
    login_result = await auth_manager.login("username", "password")
    has_permission = await auth_manager.check_permission("cameras:write")
    current_user = await auth_manager.get_current_user()
```

### 4. Service Discovery (`service_discovery.py`)

**Purpose**: Automatic discovery and health monitoring of backend services

**Key Features**:
- Redis-based service registration
- Automatic service discovery
- Continuous health monitoring
- Service endpoint management
- Load balancing support
- Circuit breaker integration

**Usage Example**:
```python
async with ServiceDiscovery() as discovery:
    discovered_count = await discovery.discover_services()
    await discovery.start_monitoring()
    service_status = await discovery.get_all_services_status()
```

### 5. Health Checker (`health_checker.py`)

**Purpose**: Comprehensive health monitoring system

**Key Features**:
- Service health monitoring
- Database connectivity checks
- Resource utilization monitoring
- Dependency validation
- Performance benchmarking
- Custom health check plugins

**Usage Example**:
```python
async with HealthChecker() as health_checker:
    system_health = await health_checker.check_all_components()
    benchmark_results = await health_checker.benchmark_component("api_service")
```

### 6. Queue Manager (`queue_manager.py`)

**Purpose**: Redis queue integration for background tasks

**Key Features**:
- Redis Streams and Lists support
- Background task submission
- Event publishing/subscribing
- Notification system
- Batch operations
- Queue health monitoring

**Usage Example**:
```python
async with CLIQueueManager() as queue_manager:
    task_id = await queue_manager.submit_task("process_video", {"video_id": 123})
    await queue_manager.publish_event("camera_added", {"camera_id": 456})
    notifications = await queue_manager.get_notifications()
```

### 7. Event Streamer (`event_streamer.py`)

**Purpose**: Real-time event streaming for monitoring

**Key Features**:
- Server-Sent Events (SSE) streaming
- WebSocket connections
- Event filtering and routing
- Connection management with reconnection
- Event buffering and replay

**Usage Example**:
```python
async with EventStreamer() as event_streamer:
    async for event in event_streamer.stream_system_events():
        print(f"Event: {event.event_type} - {event.data}")
```

### 8. Metrics Collector (`metrics_collector.py`)

**Purpose**: Comprehensive metrics collection and analysis

**Key Features**:
- Prometheus metrics scraping
- System metrics collection (psutil)
- Custom metric tracking
- Metric aggregation and analysis
- Historical data storage
- Dashboard metrics export

**Usage Example**:
```python
async with MetricsCollector() as metrics_collector:
    await metrics_collector.start_collection(interval=30)
    dashboard = metrics_collector.get_dashboard_metrics()
    exported_data = metrics_collector.export_metrics("json")
```

### 9. Service Orchestrator (`orchestrator.py`)

**Purpose**: High-level coordination of all backend services

**Key Features**:
- Coordinated service startup/shutdown
- Complex multi-service operations
- Dependency management
- Health monitoring coordination
- Event-driven workflows
- Service communication coordination

**Usage Example**:
```python
async with ServiceOrchestrator() as orchestrator:
    health_result = await orchestrator.full_system_health_check()
    monitoring_result = await orchestrator.start_monitoring_services()
    maintenance_result = await orchestrator.perform_system_maintenance(["cleanup_database"])
```

## Enhanced CLI Commands

The integration enhances existing CLI commands with real backend connectivity:

### Service Status Command
```bash
# Show comprehensive service status
its-camera-ai service status --detailed

# Watch status in real-time
its-camera-ai service status --watch --interval 10
```

### Service Discovery
```bash
# Discover all backend services
its-camera-ai service discover

# Discover and start monitoring
its-camera-ai service discover --monitor
```

### Health Checking
```bash
# Comprehensive health check
its-camera-ai service health --verbose

# Benchmark specific service
its-camera-ai service health api --benchmark
```

### Service Orchestration
```bash
# Run comprehensive health check
its-camera-ai service orchestrate health-check --detailed

# Start all monitoring services
its-camera-ai service orchestrate start-monitoring

# Perform system maintenance
its-camera-ai service orchestrate maintenance --detailed
```

### System Overview
```bash
# Show comprehensive system overview
its-camera-ai service overview
```

## Integration Benefits

### 1. **Unified Backend Access**
- Single interface to all backend services
- Consistent error handling and retry logic
- Connection pooling and resource management

### 2. **Real-time Monitoring**
- Live service status updates
- Real-time event streaming
- Continuous health monitoring

### 3. **Comprehensive Health Checking**
- Multi-component health validation
- Performance benchmarking
- Dependency verification

### 4. **Advanced Authentication**
- JWT-based authentication
- Role-based access control
- Permission validation

### 5. **Queue Integration**
- Background task management
- Event-driven architecture
- Notification system

### 6. **Metrics and Analytics**
- Comprehensive metrics collection
- Historical data analysis
- Performance monitoring

### 7. **Service Orchestration**
- Complex multi-service operations
- Coordinated maintenance tasks
- Dependency-aware operations

## Configuration

The integration uses the existing configuration system from `core/config.py`:

```python
# Database configuration
database:
  url: "postgresql+asyncpg://user:pass@localhost/its_camera_ai"
  pool_size: 10
  max_overflow: 20

# Redis configuration
redis:
  url: "redis://localhost:6379/0"
  max_connections: 20

# API configuration
api_host: "localhost"
api_port: 8080

# Security configuration
security:
  secret_key: "your-secret-key"
  algorithm: "HS256"
  access_token_expire_minutes: 30

# MinIO configuration
minio:
  endpoint: "localhost:9000"
  access_key: "minioadmin"
  secret_key: "minioadmin123"
```

## Error Handling

All components implement comprehensive error handling:

1. **Connection Errors**: Automatic retry with exponential backoff
2. **Authentication Errors**: Clear error messages and fallback options
3. **Service Unavailable**: Circuit breaker pattern with graceful degradation
4. **Configuration Errors**: Validation and helpful error messages
5. **Resource Errors**: Proper cleanup and resource management

## Performance Considerations

### Connection Pooling
- HTTP connections: Up to 100 concurrent connections
- Database connections: Configurable pool size (default: 10)
- Redis connections: Connection pooling with keep-alive

### Caching
- API responses cached for 5 minutes
- Service discovery results cached
- Metrics data with configurable TTL

### Async Operations
- All I/O operations are asynchronous
- Parallel execution where possible
- Proper resource cleanup

### Memory Management
- Bounded queues and buffers
- Automatic cleanup of old data
- Memory usage monitoring

## Testing

The integration includes comprehensive testing:

```bash
# Run all backend integration tests
pytest tests/cli/backend/

# Run specific component tests
pytest tests/cli/backend/test_api_client.py
pytest tests/cli/backend/test_health_checker.py

# Run with coverage
pytest --cov=src/its_camera_ai/cli/backend
```

## Example Usage

See `src/its_camera_ai/cli/backend/example_usage.py` for comprehensive usage examples demonstrating all components working together.

## Future Enhancements

1. **Enhanced Security**: Certificate-based authentication, encrypted connections
2. **Advanced Monitoring**: Custom dashboards, alert systems
3. **Load Balancing**: Intelligent service routing based on health and load
4. **Caching**: Distributed caching with Redis Cluster
5. **Observability**: Distributed tracing with OpenTelemetry

## Conclusion

The CLI backend integration provides a robust, scalable, and maintainable foundation for CLI operations. It ensures reliable communication with all backend services while providing advanced features like health monitoring, metrics collection, and service orchestration.

The modular architecture allows for easy extension and customization, while the comprehensive error handling and monitoring capabilities ensure reliable operation in production environments.
