# ITS Camera AI - Code Style & Conventions

## Python Version & Standards
- **Python**: 3.12+ required
- **Type Hints**: Mandatory for all functions and class attributes
- **Async/Await**: Use for all I/O operations
- **Docstrings**: Required for all public classes and functions

## Code Formatting Standards

### Black Configuration
- Line length: 88 characters
- Target version: Python 3.12
- Automatic formatting with `black src/ tests/`

### Import Sorting (isort)
- Profile: "black" compatibility
- Multi-line output: 3
- Trailing commas: enabled
- Force grid wrap: 0

### Linting (Ruff)
- Target version: Python 3.12
- Line length: 88 characters
- Enabled rules: E, W, F, I, B, C4, UP, ARG, SIM, TCH, S (security)
- Automatic fixes available with `ruff check --fix`

## Type Checking (MyPy)
- **Strict mode**: enabled
- **Disallow untyped**: All definitions must have type hints
- **Extra checks**: enabled for comprehensive type safety

## Naming Conventions

### Classes
- PascalCase: `class DataIngestionPipeline`
- Descriptive names reflecting purpose
- Use composition over inheritance

### Functions & Variables
- snake_case: `def process_camera_stream()`
- Descriptive names avoiding abbreviations
- Private methods: prefix with underscore `_process_internal()`

### Constants
- UPPER_SNAKE_CASE: `MAX_CONCURRENT_CAMERAS = 1000`
- Group related constants in dataclasses or enums

### Files & Modules
- snake_case: `production_pipeline.py`
- Clear module purpose from filename
- Avoid generic names like `utils.py`

## Project Structure Patterns

### TCA (Test-Driven Architecture)
- Main code in `src/its_camera_ai/`
- Clear separation of concerns
- Each component has single responsibility

### Module Organization
```
src/its_camera_ai/
├── core/          # Core utilities, config, logging
├── api/           # FastAPI routes and schemas  
├── ml/            # ML pipeline components
├── data/          # Data processing and streaming
├── services/      # Business logic services
├── models/        # Database models
├── storage/       # Model storage and registry
├── monitoring/    # System monitoring
└── proto/         # gRPC protocol definitions
```

## Security & Privacy Standards

### Security-First Approach
- All video data must be encrypted using `EncryptionManager`
- Apply privacy protections based on `SecurityLevel` classification
- Use `SecurityContext` for all authenticated operations
- Log security events using `SecurityAuditLogger`

### Authentication Patterns
- JWT-based authentication with refresh tokens
- Multi-factor authentication support
- Role-based access control (RBAC)
- Session management with Redis

## ML Model Development Standards

### Model Management
- Models managed through `ModelRegistry` class
- Semantic versioning for model releases
- Use `DeploymentStage` enum: development → staging → canary → production
- Production models require >90% accuracy and <100ms latency

### Performance Requirements
- **Inference latency**: <100ms target
- **Accuracy**: >90% for production deployment
- **Throughput**: 30,000 fps aggregate across all cameras
- **GPU utilization**: <80% target for auto-scaling

## Error Handling Patterns

### Exception Hierarchy
- Custom exceptions in `core/exceptions.py`
- Specific exception types for different error categories
- Proper error logging with structured logging

### Async Error Handling
```python
async def process_stream():
    try:
        # async operation
        pass
    except SpecificException as e:
        logger.error("Specific error occurred", exc_info=True)
        raise
    except Exception as e:
        logger.error("Unexpected error", exc_info=True) 
        raise ProcessingError("Stream processing failed") from e
```

## Testing Standards

### Test Organization
- Unit tests: Test individual components in isolation
- Integration tests: Test service interactions
- ML tests: Model performance and accuracy validation
- GPU tests: Hardware-specific inference optimization

### Coverage Requirements
- **Minimum**: 90% code coverage enforced
- **Preferred**: 95%+ coverage for critical components
- Use `pytest --cov-fail-under=90`

### Test Markers
```python
@pytest.mark.ml          # ML model tests
@pytest.mark.gpu         # GPU-dependent tests  
@pytest.mark.integration # Integration tests
@pytest.mark.slow        # Slow tests (>1s)
@pytest.mark.benchmark   # Performance benchmarks
```

## Documentation Standards

### Docstring Format
```python
def process_camera_stream(
    camera_id: str,
    stream_config: StreamConfig
) -> AsyncGenerator[ProcessedFrame, None]:
    """Process real-time camera stream with YOLO11 inference.
    
    Args:
        camera_id: Unique camera identifier
        stream_config: Camera stream configuration
        
    Yields:
        ProcessedFrame: Frames with detected objects and metadata
        
    Raises:
        StreamProcessingError: When stream cannot be processed
        ModelLoadError: When ML model fails to load
    """
```

### API Documentation
- FastAPI automatic documentation with detailed schemas
- Include examples in Pydantic models
- Document authentication requirements
- Provide curl examples for key endpoints