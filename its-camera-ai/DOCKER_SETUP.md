# ITS Camera AI Docker Setup Guide

Platform Engineering: Comprehensive Docker Compose Setup for Microservices Deployment

## Overview

This repository includes a complete Docker-based development and deployment platform for the ITS Camera AI system. The setup follows platform engineering best practices with:

- **Multi-environment support** (dev, production, GPU, edge)
- **Service discovery and health checks**
- **Comprehensive monitoring and observability**
- **Zero-downtime deployments**
- **Resource optimization and scaling**
- **Security hardening and compliance**

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Development   │    │   Production    │    │   GPU-Enabled   │
│   Environment   │    │   Environment   │    │   Environment   │
│                 │    │                 │    │                 │
│ • Hot reload    │    │ • Zero-downtime │    │ • CUDA support  │
│ • Debug tools   │    │ • Load balancer │    │ • Triton server │
│ • Dev databases │    │ • SSL/TLS       │    │ • GPU monitoring│
│ • Test services │    │ • Monitoring    │    │ • Benchmarking  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Edge Deployment│
                    │                 │
                    │ • Lightweight   │
                    │ • Offline mode  │
                    │ • ARM64 support │
                    │ • Edge sync     │
                    └─────────────────┘
```

## Quick Start

### Prerequisites

- Docker 24.0+ with Docker Compose
- Git
- For GPU environments: NVIDIA Docker runtime
- For production: At least 8GB RAM, 20GB disk space

### 1. Development Environment

```bash
# Clone and setup
git clone <repository>
cd its-camera-ai

# Start development environment
./start-dev.sh

# With optional services
./start-dev.sh --jupyter --kafka
```

**Access Points:**
- Application: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Database Admin: http://localhost:8080
- Redis Commander: http://localhost:8081
- MinIO Console: http://localhost:9001
- MailHog: http://localhost:8025

### 2. Production Deployment

```bash
# Setup production environment
cp .env.example .env.prod
# Edit .env.prod with production values

# Deploy with monitoring
./start-prod.sh --monitoring

# Zero-downtime deployment
./start-prod.sh --monitoring
```

### 3. GPU-Accelerated Environment

```bash
# Prerequisites: NVIDIA drivers + nvidia-docker2
# Verify GPU: nvidia-smi

# Start GPU environment
./start-gpu.sh --triton --benchmark

# With Jupyter for ML development
./start-gpu.sh --jupyter --triton
```

### 4. Edge Deployment

```bash
# For resource-constrained devices
./start-edge.sh

# With local data sync
./start-edge.sh --sync
```

## Environment Files

### Core Environment (.env)
```bash
# Application Settings
ENVIRONMENT=development
APP_PORT=8000
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql+asyncpg://its_user:password@postgres:5432/its_camera_ai
REDIS_URL=redis://:password@redis:6379/0

# MinIO Storage
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=secure_password

# Security
SECRET_KEY=your-secure-secret-key
JWT_SECRET_KEY=your-jwt-secret-key
```

### Development (.env.dev)
```bash
# Development-specific settings
DEBUG=true
RELOAD=true
DEV_MODE=true

# Development ports
ADMINER_PORT=8080
REDIS_COMMANDER_PORT=8081
MAILHOG_UI_PORT=8025
```

### Production (.env.prod)
```bash
# Production settings
ENVIRONMENT=production
WORKERS=4
SSL_ENABLED=true

# Resource limits
APP_MEMORY_LIMIT=8G
POSTGRES_MEMORY_LIMIT=4G
REDIS_MEMORY_LIMIT=2G
```

### GPU (.env.gpu)
```bash
# GPU settings
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
GPU_MEMORY_FRACTION=0.8
TENSORRT_ENABLED=true
```

## Services Overview

### Core Application Services

#### Main Application
- **Image**: Custom (multi-stage Dockerfile)
- **Ports**: 8000 (API), 8001 (metrics)
- **Health**: `/health` endpoint
- **Scaling**: Horizontal scaling supported

#### Database Stack
- **PostgreSQL**: Primary database with optimization
- **Redis**: Caching and session storage
- **TimescaleDB**: Time-series metrics storage

#### Storage & Messaging
- **MinIO**: S3-compatible object storage
- **Apache Kafka**: Event streaming (optional)

### Monitoring & Observability

#### Core Monitoring
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Alertmanager**: Alert management

#### Logging & Tracing
- **Loki**: Log aggregation
- **Jaeger**: Distributed tracing
- **Fluent Bit**: Log collection (edge)

#### System Monitoring
- **Node Exporter**: System metrics
- **NVIDIA Exporter**: GPU metrics
- **MinIO Metrics**: Storage metrics

### Development Tools

- **Adminer**: Database management
- **Redis Commander**: Redis management
- **MailHog**: Email testing
- **Jupyter Lab**: ML development (GPU-enabled)

## Docker Compose Files

### Main Configuration
- `docker-compose.yml`: Base services and configuration
- `docker-compose.override.yml`: Local overrides

### Environment-Specific
- `docker-compose.dev.yml`: Development optimizations
- `docker-compose.prod.yml`: Production hardening
- `docker-compose.gpu.yml`: GPU acceleration
- `docker-compose.edge.yml`: Edge deployment

## Service Profiles

Use profiles to control which services start:

```bash
# Development with tools
docker compose --profile dev up

# Production stack
docker compose --profile prod up

# GPU-accelerated
docker compose --profile gpu up

# Edge deployment
docker compose --profile edge up

# Monitoring stack
docker compose --profile monitoring up

# ML development
docker compose --profile ml up
```

## Networking

### Development Network
- **Subnet**: 172.20.0.0/16
- **Bridge**: its-dev-bridge
- **DNS**: Automatic service discovery

### Production Network
- **Subnet**: 172.22.0.0/16
- **Security**: Network policies applied
- **Isolation**: Service segmentation

### Edge Network
- **Subnet**: 172.21.0.0/16
- **Optimization**: Reduced MTU for edge
- **Connectivity**: Offline-first design

## Volume Management

### Development Volumes
```yaml
volumes:
  dev_data:           # Application data
  dev_postgres_data:  # Database data
  dev_redis_data:     # Cache data
  dev_models:         # ML models
  dev_logs:           # Application logs
```

### Production Volumes
```yaml
volumes:
  prod_data:          # Persistent application data
  prod_postgres_data: # Database with backups
  prod_logs:          # Structured logs
  prod_nginx_logs:    # Web server logs
```

### GPU Volumes
```yaml
volumes:
  gpu_models:         # GPU-optimized models
  gpu_cache:          # Model cache
  gpu_tensorrt_cache: # TensorRT optimizations
  gpu_cuda_cache:     # CUDA kernel cache
```

## Performance Optimization

### Resource Limits

#### Development
```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '4.0'
    reservations:
      memory: 1G
      cpus: '1.0'
```

#### Production
```yaml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4.0'
    reservations:
      memory: 4G
      cpus: '2.0'
```

#### GPU
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Health Checks

All services include comprehensive health checks:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## Security Features

### Container Security
- Non-root users (uid:gid 1001:1001)
- Read-only root filesystems where possible
- Security scanning enabled
- Vulnerability management

### Network Security
- Service isolation
- Rate limiting (Nginx)
- CORS configuration
- Security headers

### Data Security
- Encrypted storage volumes
- Secure credential management
- Audit logging
- Backup encryption

## Monitoring & Alerting

### Metrics Collection
```yaml
# Prometheus scrape configs
scrape_configs:
  - job_name: 'its-camera-ai'
    targets: ['app:8001']
    scrape_interval: 10s
  
  - job_name: 'postgres'
    targets: ['postgres:5432']
    scrape_interval: 30s
```

### Alert Rules
- Service availability
- Resource utilization
- Error rates
- Performance degradation
- Security events

### Dashboards
- Application performance
- Infrastructure metrics
- ML model performance
- Business metrics

## Backup & Recovery

### Automated Backups
```bash
# Database backups
docker compose exec postgres pg_dump -U its_user its_camera_ai > backup.sql

# Volume backups
docker run --rm -v prod_data:/data -v $(pwd)/backups:/backup alpine tar czf /backup/data-$(date +%Y%m%d).tar.gz /data
```

### Recovery Procedures
```bash
# Database restore
docker compose exec -T postgres psql -U its_user its_camera_ai < backup.sql

# Volume restore
docker run --rm -v prod_data:/data -v $(pwd)/backups:/backup alpine tar xzf /backup/data-20240101.tar.gz -C /
```

## Scaling & Load Balancing

### Horizontal Scaling
```bash
# Scale application
docker compose up -d --scale app=3

# Load balancer automatically distributes
```

### Vertical Scaling
```yaml
# Adjust resource limits
deploy:
  resources:
    limits:
      memory: 16G
      cpus: '8.0'
```

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
docker compose logs -f <service_name>

# Check health
docker compose ps

# Restart service
docker compose restart <service_name>
```

#### Performance Issues
```bash
# Check resource usage
docker stats

# Monitor GPU usage
nvidia-smi -l 1

# Check application metrics
curl http://localhost:8001/metrics
```

#### Database Issues
```bash
# Connect to database
docker compose exec postgres psql -U its_user its_camera_ai

# Check database performance
docker compose exec postgres pg_stat_activity
```

#### Storage Issues
```bash
# Check disk usage
docker system df

# Clean unused resources
docker system prune -a

# Check volume usage
docker volume ls
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Set debug environment
export DEBUG=true
export LOG_LEVEL=DEBUG

# Start with verbose logging
docker compose --verbose up
```

### Performance Profiling

```bash
# Application profiling
docker compose exec app python -m cProfile -o profile.stats app.py

# GPU profiling
nvidia-smi dmon -s pucvmet -d 1
```

## Maintenance

### Regular Updates
```bash
# Pull latest images
docker compose pull

# Update with zero downtime (production)
./start-prod.sh
```

### Health Monitoring
```bash
# Service status
./health-check.sh

# Resource monitoring
./monitor-resources.sh

# Performance benchmarks
./run-benchmarks.sh
```

### Cleanup
```bash
# Basic cleanup
./cleanup.sh

# Complete cleanup
./cleanup.sh --all

# Preserve data
./cleanup.sh --keep-volumes
```

## Best Practices

### Development
1. Use development profiles for hot reload
2. Enable debug logging
3. Use development tools (Adminer, Redis Commander)
4. Regular testing with fresh data

### Production
1. Use production profiles with resource limits
2. Enable monitoring and alerting
3. Regular backups
4. Security scanning
5. Zero-downtime deployments

### GPU
1. Monitor GPU utilization
2. Optimize batch sizes
3. Use mixed precision training
4. Profile inference performance

### Edge
1. Minimize resource usage
2. Enable offline capabilities
3. Regular sync with cloud
4. Monitor connectivity

## Integration

### CI/CD Pipeline
```yaml
# Example GitHub Actions
name: Deploy
on: [push]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Production
        run: |
          ./start-prod.sh --monitoring
```

### Infrastructure as Code
```hcl
# Terraform integration
resource "docker_compose" "its_camera_ai" {
  compose_file = "docker-compose.yml:docker-compose.prod.yml"
  project_name = "its-camera-ai"
}
```

## Support

### Documentation
- [API Documentation](http://localhost:8000/docs)
- [Architecture Guide](ARCHITECTURE.md)
- [Deployment Guide](DEPLOYMENT.md)

### Monitoring
- [Grafana Dashboards](http://localhost:3000)
- [Prometheus Metrics](http://localhost:9090)
- [Application Logs](http://localhost:3100)

### Community
- GitHub Issues
- Development Chat
- Stack Overflow Tags

---

**Platform Engineering Excellence**: This Docker setup provides a production-ready, scalable, and maintainable platform for the ITS Camera AI system with comprehensive observability, security, and performance optimization.