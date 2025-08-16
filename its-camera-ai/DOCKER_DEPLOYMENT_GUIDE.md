# Docker Deployment Guide for ITS Camera AI

This guide provides comprehensive instructions for deploying the ITS Camera AI system using Docker containers in development, staging, and production environments.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Development Deployment](#development-deployment)
- [GPU-Enabled Deployment](#gpu-enabled-deployment)
- [Production Deployment](#production-deployment)
- [Monitoring and Observability](#monitoring-and-observability)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)

## 🚀 Quick Start

### Development Environment (CPU-only)
```bash
# 1. Clone and setup
git clone <repository-url>
cd its-camera-ai

# 2. Configure environment
cp .env.example .env
# Edit .env with your configuration

# 3. Start development stack
docker-compose up -d

# 4. Verify deployment
curl http://localhost:8000/health
```

### Development with GPU Support
```bash
# Start GPU-enabled development environment
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### Production Deployment
```bash
# 1. Configure production environment
cp .env.example .env.prod
# Edit .env.prod with production secrets

# 2. Deploy production stack
docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d
```

## 📋 Prerequisites

### System Requirements

**Development Environment:**
- Docker Engine 20.10+
- Docker Compose 2.0+
- 8GB RAM minimum, 16GB recommended
- 50GB available disk space

**GPU-Enabled Environment:**
- NVIDIA Docker runtime
- NVIDIA GPU with CUDA 12.1+ support
- NVIDIA drivers 525.60.13+
- 16GB RAM minimum, 32GB recommended

**Production Environment:**
- Docker Engine 20.10+ with BuildKit
- Docker Compose 2.0+
- 32GB RAM minimum, 64GB recommended
- 500GB SSD storage minimum
- Multiple GPU nodes for ML inference

### Software Dependencies

**Required:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

**For GPU Support:**
```bash
# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Optional Tools:**
```bash
# For security scanning
brew install aquasecurity/trivy/trivy

# For image analysis
brew install dive

# For monitoring
brew install prometheus grafana
```

## 🔧 Environment Configuration

### 1. Copy and Configure Environment File

```bash
cp .env.example .env
```

### 2. Essential Configuration

**Database Settings:**
```bash
# PostgreSQL configuration
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_REPLICATION_PASSWORD=your_replication_password

# Connection settings
DATABASE_URL=postgresql+asyncpg://its_user:${POSTGRES_PASSWORD}@postgres:5432/its_camera_ai
```

**Security Settings:**
```bash
# Generate secure keys
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET_KEY=$(openssl rand -hex 32)
ENCRYPTION_KEY=$(openssl rand -hex 32)
```

**Object Storage:**
```bash
# MinIO credentials
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key_min_32_chars
```

**Monitoring:**
```bash
# Grafana configuration
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your_secure_grafana_password
```

### 3. Environment-Specific Files

Create separate environment files for different deployments:

```bash
# Development
cp .env.example .env.dev
# Edit with development-specific settings

# Staging
cp .env.example .env.staging
# Edit with staging-specific settings

# Production
cp .env.example .env.prod
# Edit with production-specific settings
```

## 🛠️ Development Deployment

### Basic Development Stack

**1. Start Core Services:**
```bash
docker-compose up -d postgres redis kafka minio
```

**2. Run Database Migrations:**
```bash
docker-compose exec api alembic upgrade head
```

**3. Start Application Services:**
```bash
docker-compose up -d api worker-analytics worker-aggregation
```

**4. Start Monitoring (Optional):**
```bash
docker-compose up -d prometheus grafana
```

### Development with Hot Reload

The development configuration includes hot-reload for rapid development:

```bash
# Start with hot-reload enabled
docker-compose up -d

# View logs
docker-compose logs -f api

# Execute commands in running containers
docker-compose exec api python -m its_camera_ai.cli.main database migrate
docker-compose exec api python -m its_camera_ai.cli.main ml deploy --model-path ./models/yolo11n.pt
```

### Available Services

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | FastAPI application |
| API Docs | http://localhost:8000/docs | Interactive API documentation |
| Grafana | http://localhost:3000 | Monitoring dashboards |
| Prometheus | http://localhost:9090 | Metrics collection |
| MinIO Console | http://localhost:9001 | Object storage management |
| Redis Insight | http://localhost:8001 | Redis debugging |
| Kafka UI | http://localhost:8080 | Kafka management |

## 🎮 GPU-Enabled Deployment

### Prerequisites for GPU Deployment

**1. Verify GPU Support:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

**2. Configure GPU Access:**
```bash
# Add GPU support to Docker daemon
sudo tee /etc/docker/daemon.json <<EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

sudo systemctl restart docker
```

### GPU Development Environment

```bash
# Start GPU-enabled development stack
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Verify GPU access in ML service
docker-compose exec ml-inference python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage
docker-compose exec dcgm-exporter nvidia-smi -l 1
```

### GPU Service Configuration

The GPU configuration includes:

- **ML Inference Service**: CUDA-enabled YOLO11 inference
- **Triton Server**: High-performance model serving (optional)
- **GPU Analytics Worker**: ML post-processing with GPU acceleration
- **DCGM Exporter**: GPU metrics collection

### GPU Performance Monitoring

```bash
# View GPU metrics in Grafana
open http://localhost:3000/d/gpu-monitoring

# Check GPU utilization
curl http://localhost:9400/metrics | grep dcgm_gpu_utilization

# Monitor inference performance
docker-compose logs -f ml-inference | grep "inference_time"
```

## 🏭 Production Deployment

### Production Architecture

The production deployment includes:

- **Load Balancer**: Nginx with SSL termination
- **API Cluster**: Primary/secondary FastAPI instances
- **ML Inference Cluster**: GPU-enabled inference servers
- **Database Cluster**: PostgreSQL with replication
- **Cache Cluster**: Redis master-replica setup
- **Message Broker**: Kafka 3-node cluster
- **Object Storage**: MinIO 4-node cluster
- **Monitoring Stack**: Prometheus, Grafana, AlertManager

### Production Deployment Steps

**1. Prepare Production Environment:**
```bash
# Create production directory
mkdir -p /opt/its-camera-ai
cd /opt/its-camera-ai

# Clone repository
git clone <repository-url> .

# Configure production environment
cp .env.example .env.prod
```

**2. Configure Production Secrets:**
```bash
# Generate secure passwords
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
MINIO_SECRET_KEY=$(openssl rand -base64 48)

# Update .env.prod with generated passwords
vim .env.prod
```

**3. Setup SSL Certificates:**
```bash
# Create SSL directory
mkdir -p infrastructure/nginx/ssl

# Copy SSL certificates
cp your-domain.crt infrastructure/nginx/ssl/its-camera-ai.crt
cp your-domain.key infrastructure/nginx/ssl/its-camera-ai.key

# Set proper permissions
chmod 600 infrastructure/nginx/ssl/its-camera-ai.key
```

**4. Deploy Production Stack:**
```bash
# Deploy infrastructure services first
docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d postgres-primary redis-primary kafka-1 minio-1

# Wait for services to be healthy
docker-compose -f docker-compose.prod.yml ps

# Deploy application services
docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d

# Verify deployment
curl -k https://your-domain.com/health
```

### Production Health Checks

```bash
# Check all services status
docker-compose -f docker-compose.prod.yml ps

# Verify database replication
docker-compose -f docker-compose.prod.yml exec postgres-primary psql -U its_user -c "SELECT * FROM pg_stat_replication;"

# Check Redis cluster status
docker-compose -f docker-compose.prod.yml exec redis-primary redis-cli INFO replication

# Test load balancer
curl -I https://your-domain.com/health

# Check GPU inference
grpcurl -plaintext ml.your-domain.com:50051 list
```

### Production Scaling

**Horizontal Scaling:**
```bash
# Scale API services
docker-compose -f docker-compose.prod.yml up -d --scale api-primary=3

# Scale workers
docker-compose -f docker-compose.prod.yml up -d --scale worker-analytics-cluster=5

# Scale ML inference
docker-compose -f docker-compose.prod.yml up -d --scale ml-inference-primary=2
```

**Resource Limits:**
```yaml
# Production resource configuration
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      cpus: '2.0'
      memory: 4G
```

## 📊 Monitoring and Observability

### Metrics Collection

**Prometheus Targets:**
- Application metrics (FastAPI, workers)
- Infrastructure metrics (PostgreSQL, Redis, Kafka)
- GPU metrics (DCGM exporter)
- Container metrics (cAdvisor)
- System metrics (Node exporter)

**Key Metrics:**
```promql
# API response time
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))

# ML inference latency
histogram_quantile(0.99, sum(rate(ml_inference_duration_seconds_bucket[5m])) by (le))

# Database connections
postgresql_connections{state="active"}

# GPU utilization
dcgm_gpu_utilization
```

### Grafana Dashboards

Access dashboards at `https://monitoring.your-domain.com/grafana`:

1. **ITS Camera AI Overview**: System health and performance
2. **ML Pipeline Performance**: Inference metrics and GPU usage
3. **Business Analytics**: Traffic analysis and KPIs
4. **Infrastructure Monitoring**: Database, cache, and message broker metrics
5. **Security Dashboard**: Authentication, authorization, and security events

### Alerting Rules

**Critical Alerts:**
- API response time > 1s
- ML inference latency > 100ms
- Database connection failures
- GPU memory usage > 90%
- Disk space < 10%

**Warning Alerts:**
- High CPU usage > 80%
- Memory usage > 85%
- Queue depth > 1000 messages
- Cache hit ratio < 70%

### Log Aggregation

**Structured Logging:**
```bash
# View application logs
docker-compose logs -f api | jq '.'

# Filter by log level
docker-compose logs api 2>&1 | grep '"level":"ERROR"'

# Monitor specific events
docker-compose logs -f ml-inference | grep inference_complete
```

## 🔧 Troubleshooting

### Common Issues

**1. Container Startup Failures:**
```bash
# Check container status
docker-compose ps

# View container logs
docker-compose logs <service-name>

# Check resource usage
docker stats

# Inspect container configuration
docker inspect <container-name>
```

**2. Database Connection Issues:**
```bash
# Test database connectivity
docker-compose exec api pg_isready -h postgres -p 5432 -U its_user

# Check database logs
docker-compose logs postgres

# Verify connection pool
docker-compose exec api python -c "from src.its_camera_ai.core.config import get_settings; print(get_settings().database_url)"
```

**3. GPU-Related Problems:**
```bash
# Verify GPU access
docker-compose exec ml-inference nvidia-smi

# Check CUDA availability
docker-compose exec ml-inference python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# Monitor GPU memory
docker-compose exec ml-inference nvidia-smi -l 1
```

**4. Performance Issues:**
```bash
# Check system resources
docker system df
docker system events

# Monitor container performance
docker stats --no-stream

# Profile application
docker-compose exec api python -m cProfile -o profile.stats -m its_camera_ai.main
```

### Debugging Tools

**Container Debugging:**
```bash
# Enter container shell
docker-compose exec api bash

# Run health checks manually
docker-compose exec api curl -f http://localhost:8000/health

# Check environment variables
docker-compose exec api env | grep DATABASE

# Test service connectivity
docker-compose exec api nc -zv postgres 5432
```

**Network Debugging:**
```bash
# List Docker networks
docker network ls

# Inspect network configuration
docker network inspect its-camera-ai_its-network

# Test inter-container connectivity
docker-compose exec api ping postgres
```

### Performance Profiling

**Application Profiling:**
```bash
# Enable profiling in development
export PROFILING_ENABLED=true
docker-compose up -d

# Generate performance report
docker-compose exec api python scripts/generate_performance_report.py

# Memory profiling
docker-compose exec api python -m memory_profiler src/its_camera_ai/main.py
```

**Database Profiling:**
```bash
# Enable query logging
docker-compose exec postgres psql -U its_user -c "ALTER SYSTEM SET log_statement = 'all';"

# Analyze slow queries
docker-compose exec postgres psql -U its_user -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

## ⚡ Performance Optimization

### Application Optimization

**FastAPI Configuration:**
```python
# Production settings
WORKERS=4
WORKER_CLASS=uvicorn.workers.UvicornWorker
WORKER_CONNECTIONS=1000
MAX_REQUESTS=1000
KEEP_ALIVE=2
```

**Database Optimization:**
```bash
# PostgreSQL tuning
POSTGRES_SHARED_BUFFERS=2GB
POSTGRES_EFFECTIVE_CACHE_SIZE=6GB
POSTGRES_WORK_MEM=64MB
POSTGRES_MAINTENANCE_WORK_MEM=512MB
POSTGRES_MAX_CONNECTIONS=1000
```

**Redis Optimization:**
```bash
# Redis configuration
REDIS_MAXMEMORY=2gb
REDIS_MAXMEMORY_POLICY=allkeys-lru
REDIS_SAVE_FREQUENCY="900 1 300 10 60 10000"
```

### ML Inference Optimization

**GPU Optimization:**
```bash
# CUDA settings
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ML_MEMORY_FRACTION=0.9
ML_PRECISION=fp16
ML_TENSORRT_ENABLED=true
```

**Batch Processing:**
```bash
# Inference batching
ML_BATCH_SIZE=16
ML_MAX_BATCH_DELAY=10
ML_WORKERS=4
```

### Container Optimization

**Image Optimization:**
```bash
# Multi-stage build optimization
docker build --target production --cache-from its-camera-ai:cache .

# Layer caching
docker buildx build --cache-from type=registry,ref=ghcr.io/your-org/its-camera-ai:cache .
```

**Resource Limits:**
```yaml
# Production resource configuration
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Network Optimization

**Nginx Configuration:**
```nginx
# Performance tuning
worker_processes auto;
worker_connections 1024;
keepalive_timeout 65;
client_max_body_size 50m;

# Compression
gzip on;
gzip_comp_level 6;
gzip_types text/plain application/json;

# Caching
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m;
```

## 📚 Additional Resources

### Documentation
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Prometheus Monitoring](https://prometheus.io/docs/guides/multi-target-exporter/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)
- [NVIDIA Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Build and Deployment Scripts
- `docker/docker-build.sh` - Comprehensive build script
- `scripts/deploy-production.sh` - Production deployment automation
- `scripts/deploy-monitoring.sh` - Monitoring stack deployment
- `scripts/quickstart.sh` - Quick development setup

### Support
- GitHub Issues: Create an issue for bugs or feature requests
- Documentation: Check the `/docs` directory for detailed guides
- Monitoring: Use Grafana dashboards for real-time system health

---

This guide provides a comprehensive foundation for deploying the ITS Camera AI system using Docker. For production deployments, ensure you follow security best practices and regularly update your containers and dependencies.