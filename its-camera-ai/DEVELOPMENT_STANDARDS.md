# ITS Camera AI - Development Standards

This document outlines the development standards, coding conventions, and best practices for the ITS Camera AI project.

## Code Quality Standards

### Type Safety
- **100% type coverage** required for all public APIs
- Use type hints for all function signatures, class attributes, and return types
- Leverage Pydantic models for data validation and serialization
- Use `mypy` in strict mode for static type checking

```python
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

def process_camera_data(
    camera_id: str,
    frame_data: bytes,
    metadata: Optional[Dict[str, Any]] = None
) -> ProcessingResult:
    """Process camera frame data with type safety."""
    pass
```

### Code Formatting
- Use `black` for code formatting (line length: 88 characters)
- Use `isort` for import sorting with black-compatible profile
- Use `ruff` for linting and additional code quality checks

```bash
# Format code
black src/ tests/
isort src/ tests/
ruff check src/ tests/
```

### Documentation
- Use Google-style docstrings for all functions, classes, and modules
- Document all parameters, return values, and exceptions
- Include usage examples for complex functions

```python
def analyze_traffic_pattern(
    video_stream: VideoStream,
    model_config: ModelConfig,
    detection_threshold: float = 0.5
) -> TrafficAnalysis:
    """Analyze traffic patterns from video stream.
    
    Processes video frames to detect vehicles, count traffic flow,
    and identify congestion patterns using AI models.
    
    Args:
        video_stream: Input video stream from camera
        model_config: Configuration for ML models
        detection_threshold: Minimum confidence threshold for detections
    
    Returns:
        TrafficAnalysis: Analysis results with vehicle counts and patterns
    
    Raises:
        ProcessingError: If video processing fails
        ModelError: If ML model inference fails
    
    Example:
        >>> stream = VideoStream("rtsp://camera:554/stream")
        >>> config = ModelConfig(model_type="yolov8")
        >>> analysis = analyze_traffic_pattern(stream, config)
        >>> print(f"Vehicles detected: {analysis.vehicle_count}")
    """
    pass
```

## Architecture Principles

### Async-First Design
- Use `async`/`await` for all I/O-bound operations
- Leverage async context managers for resource management
- Use async database sessions and Redis connections

```python
async def process_camera_stream(camera_id: str) -> None:
    """Process camera stream asynchronously."""
    async with get_camera_stream(camera_id) as stream:
        async for frame in stream:
            result = await analyze_frame(frame)
            await store_result(result)
```

### Dependency Injection
- Use FastAPI's dependency injection system
- Create reusable dependency functions
- Maintain clear separation of concerns

```python
@router.post("/cameras/{camera_id}/analyze")
async def analyze_camera(
    camera_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    ml_service: MLService = Depends(get_ml_service),
) -> AnalysisResult:
    """Analyze camera feed with proper dependency injection."""
    pass
```

### Error Handling
- Use custom exception hierarchy
- Provide detailed error context
- Log errors with structured data

```python
try:
    result = await process_frame(frame_data)
except ProcessingError as e:
    logger.error(
        "Frame processing failed",
        camera_id=camera_id,
        frame_id=frame_id,
        error=str(e),
        exc_info=True,
    )
    raise InferenceError(
        "Failed to process camera frame",
        camera_id=camera_id,
        details={"frame_id": frame_id},
        cause=e,
    ) from e
```

## Testing Standards

### Test Coverage
- Maintain **minimum 90% test coverage**
- Write unit tests for all business logic
- Include integration tests for API endpoints
- Add performance tests for critical paths

### Test Structure
- Use `pytest` as the testing framework
- Organize tests by module in parallel directory structure
- Use descriptive test names that explain the scenario

```python
class TestTrafficAnalyzer:
    """Test traffic analysis functionality."""
    
    async def test_analyze_traffic_with_multiple_vehicles_returns_correct_count(self):
        """Test that traffic analyzer correctly counts multiple vehicles."""
        # Arrange
        mock_frames = create_mock_frames_with_vehicles(count=5)
        analyzer = TrafficAnalyzer()
        
        # Act
        result = await analyzer.analyze(mock_frames)
        
        # Assert
        assert result.vehicle_count == 5
        assert result.confidence > 0.8
```

### Test Fixtures
- Use pytest fixtures for common test data
- Create database fixtures with proper cleanup
- Mock external services consistently

```python
@pytest.fixture
async def sample_camera_data():
    """Create sample camera data for testing."""
    return {
        "name": "Test Camera",
        "location": "Test Location",
        "stream_url": "rtsp://test:554/stream",
        "is_active": True,
    }
```

## Performance Standards

### Response Time Targets
- API endpoints: < 100ms (95th percentile)
- ML inference: < 50ms per frame
- Database queries: < 10ms (simple), < 100ms (complex)
- Real-time streaming: < 200ms end-to-end latency

### Memory Management
- Use generators for large datasets
- Implement proper resource cleanup
- Monitor memory usage in production

```python
async def process_video_stream(stream_url: str) -> AsyncGenerator[AnalysisResult, None]:
    """Process video stream with memory-efficient generators."""
    async with VideoStream(stream_url) as stream:
        async for frame in stream:
            try:
                result = await analyze_frame(frame)
                yield result
            finally:
                # Ensure frame memory is released
                del frame
```

### Async Patterns
- Use connection pooling for databases and external services
- Implement proper async context managers
- Handle concurrent operations efficiently

```python
async def process_multiple_cameras(
    camera_ids: List[str]
) -> List[ProcessingResult]:
    """Process multiple cameras concurrently."""
    tasks = [
        process_camera_stream(camera_id)
        for camera_id in camera_ids
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle individual failures gracefully
    return [
        result for result in results
        if not isinstance(result, Exception)
    ]
```

## Security Standards

### Authentication & Authorization
- Use JWT tokens for API authentication
- Implement role-based access control (RBAC)
- Validate all input data with Pydantic

### Data Protection
- Never log sensitive information
- Use environment variables for secrets
- Implement proper rate limiting

### API Security
- Add security headers middleware
- Implement CORS properly
- Use HTTPS in production

```python
@router.post("/cameras/")
async def create_camera(
    camera_data: CameraCreate,
    current_user: User = Depends(require_roles("admin", "operator")),
    db: AsyncSession = Depends(get_db),
) -> Camera:
    """Create camera with proper authorization."""
    # Validate and sanitize input
    validated_data = camera_data.dict(exclude_unset=True)
    
    # Create camera with audit trail
    camera = await create_camera_with_audit(
        db, validated_data, created_by=current_user.id
    )
    
    return camera
```

## Development Workflow

### Git Workflow
- Use feature branches for all development
- Write descriptive commit messages
- Squash commits before merging to main
- Use semantic versioning for releases

### Code Review Process
1. All code must be reviewed before merging
2. Run all tests and linting checks
3. Verify documentation is updated
4. Check test coverage meets requirements

### CI/CD Pipeline
- Automated testing on all pull requests
- Type checking with mypy
- Security scanning with bandit
- Performance regression testing

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
  
  - repo: https://github.com/psf/black
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/isort
    hooks:
      - id: isort
  
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    hooks:
      - id: ruff
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy
```

## Production Deployment

### Environment Configuration
- Use environment-specific configuration files
- Validate configuration on startup
- Implement graceful degradation for missing services

### Monitoring & Observability
- Structured logging with correlation IDs
- Metrics collection with Prometheus
- Distributed tracing with OpenTelemetry
- Health check endpoints for monitoring

### Scalability
- Design for horizontal scaling
- Use async patterns for I/O operations
- Implement proper caching strategies
- Monitor resource usage and performance

## ML/AI Specific Standards

### Model Management
- Version control for model artifacts
- A/B testing for model deployments
- Model performance monitoring
- Automated model retraining pipelines

### Data Processing
- Validate input data quality
- Implement data preprocessing pipelines
- Handle missing or corrupted data gracefully
- Monitor data drift and model performance

```python
class ModelService:
    """Service for ML model operations."""
    
    async def predict(
        self,
        input_data: np.ndarray,
        model_version: str = "latest"
    ) -> PredictionResult:
        """Make prediction with proper error handling and monitoring."""
        try:
            # Validate input data
            validated_input = self.validate_input(input_data)
            
            # Load model
            model = await self.get_model(model_version)
            
            # Make prediction
            with timer("model_inference_duration"):
                prediction = await model.predict(validated_input)
            
            # Log metrics
            self.metrics.increment("predictions_total")
            
            return PredictionResult(
                prediction=prediction,
                confidence=prediction.confidence,
                model_version=model_version,
                timestamp=datetime.utcnow(),
            )
            
        except Exception as e:
            self.metrics.increment("prediction_errors_total")
            logger.error(
                "Prediction failed",
                model_version=model_version,
                input_shape=input_data.shape,
                error=str(e),
            )
            raise ModelError("Prediction failed", cause=e) from e
```

## Development Commands

```bash
# Setup development environment
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest --cov=its_camera_ai --cov-report=html

# Type checking
mypy src/its_camera_ai

# Code formatting
black src/ tests/
isort src/ tests/
ruff check src/ tests/ --fix

# Security scanning
bandit -r src/its_camera_ai
safety check

# Start development server
its-camera-ai serve --reload --log-level debug

# Run database migrations
its-camera-ai init-db --create-tables

# Health check
its-camera-ai health --component all
```

This comprehensive development standards document ensures code quality, maintainability, and production readiness for the ITS Camera AI system.
