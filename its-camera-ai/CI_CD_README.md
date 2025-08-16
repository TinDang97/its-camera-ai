# 🚀 ITS Camera AI - CI/CD Pipeline Documentation

This document provides comprehensive guidance for the CI/CD pipeline implementation for Tasks 5.4.1-5.4.4, including Docker-based workflows, automated testing, image management, and deployment automation.

## 📋 Table of Contents

- [Overview](#overview)
- [Task 5.4.1: Docker CI/CD Workflow](#task-541-docker-cicd-workflow)
- [Task 5.4.2: Quality Gates & Testing](#task-542-quality-gates--testing)
- [Task 5.4.3: Image Registry & Versioning](#task-543-image-registry--versioning)
- [Task 5.4.4: Automated Deployment](#task-544-automated-deployment)
- [Getting Started](#getting-started)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The ITS Camera AI CI/CD pipeline provides a comprehensive, production-ready solution with:

- **Multi-stage Docker builds** with optimized caching
- **Comprehensive testing** including unit, integration, E2E, and security tests
- **Automated security scanning** with vulnerability assessments
- **Multi-architecture support** (linux/amd64, linux/arm64)
- **Blue-green deployments** with rollback mechanisms
- **Environment-specific configurations** for development, staging, and production
- **Monitoring and observability** integration

## 🐳 Task 5.4.1: Docker CI/CD Workflow

### Features

- **Matrix builds** for different environments and architectures
- **Multi-stage Dockerfiles** optimized for each environment
- **Layer caching** using GitHub Actions cache and registry cache
- **Security scanning** integrated into the build process
- **Container structure testing** for validation

### Workflow Files

```
.github/workflows/
├── docker-cicd.yml          # Main Docker CI/CD pipeline
├── ci.yml                   # Enhanced CI pipeline  
├── deployment.yml           # Automated deployment workflow
└── release.yml              # Release management workflow
```

### Docker Targets

The pipeline supports multiple Docker targets:

- **development**: Hot-reload enabled, debug tools included
- **production**: Optimized for production, minimal attack surface
- **gpu-production**: GPU-accelerated inference
- **edge**: Lightweight for edge deployment
- **testing**: Full test suite included

### Example Workflow Trigger

```yaml
on:
  push:
    branches: [main, develop, feature/*, hotfix/*]
    tags: ['v*']
  pull_request:
    branches: [main, develop]
```

## 🧪 Task 5.4.2: Quality Gates & Testing

### Testing Strategy

The pipeline implements comprehensive testing with strict quality gates:

1. **Unit Tests** (90% coverage requirement)
2. **Integration Tests** (80% coverage requirement)
3. **End-to-End Tests** (Puppeteer-based)
4. **Security Scans** (Bandit, Safety, pip-audit)
5. **Performance Tests** (Sub-100ms latency requirement)
6. **Load Tests** (K6-based)

### Test Execution

Tests run in isolated Docker environments:

```bash
# Run all test suites
docker-compose -f docker/docker-compose.test.yml up --abort-on-container-exit

# Run specific test suite
docker-compose -f docker/docker-compose.test.yml up unit-tests
docker-compose -f docker/docker-compose.test.yml up integration-tests
docker-compose -f docker/docker-compose.test.yml up security-tests
```

### Quality Metrics

- **Code Coverage**: Minimum 90% for unit tests, 80% for integration
- **Security**: Zero critical/high vulnerabilities allowed
- **Performance**: API response time < 200ms
- **Type Safety**: MyPy validation required

### Puppeteer E2E Testing

E2E tests run across multiple browsers and viewports:

```javascript
// Example E2E test configuration
const config = {
  browsers: ['chromium', 'firefox'],
  viewports: ['desktop', 'tablet', 'mobile'],
  baseURL: 'http://localhost:3000',
  timeout: 60000
};
```

## 📦 Task 5.4.3: Image Registry & Versioning

### Registry Strategy

Images are published to GitHub Container Registry with semantic versioning:

```
ghcr.io/username/its-camera-ai:latest
ghcr.io/username/its-camera-ai:v1.2.3
ghcr.io/username/its-camera-ai:v1.2.3-production
ghcr.io/username/its-camera-ai:v1.2.3-gpu
ghcr.io/username/its-camera-ai:v1.2.3-edge
```

### Image Promotion Pipeline

Images are promoted through environments:

```
development → staging → production → release
```

### Security Scanning

Multiple scanners validate image security:

- **Trivy**: Comprehensive vulnerability scanning
- **Grype**: Additional vulnerability analysis  
- **Docker Scout**: Docker-native security scanning

### Multi-Architecture Support

Images are built for multiple platforms:

- `linux/amd64` (x86_64)
- `linux/arm64` (ARM64/Apple Silicon)

## 🚀 Task 5.4.4: Automated Deployment

### Deployment Strategies

#### Rolling Deployment
- **Use case**: Development and staging environments
- **Zero downtime**: Gradual service replacement
- **Rollback**: Automatic rollback on health check failure

#### Blue-Green Deployment  
- **Use case**: Production environment
- **Zero downtime**: Parallel environment deployment
- **Instant rollback**: Switch traffic back to previous environment

#### Canary Deployment
- **Use case**: High-risk production changes
- **Risk mitigation**: Limited traffic exposure
- **Progressive rollout**: Monitor metrics before full deployment

### Deployment Script

Use the deployment script for manual deployments:

```bash
# Development deployment
./scripts/deploy.sh -e development

# Staging with specific version
./scripts/deploy.sh -e staging -v v1.2.3 -s rolling

# Production blue-green deployment
./scripts/deploy.sh -e production -v v1.2.3 -s blue-green

# Rollback production
./scripts/deploy.sh -e production --rollback
```

### Health Checks

Comprehensive health validation:

1. **Container Health**: Docker health checks
2. **API Health**: HTTP endpoint validation
3. **Database Connectivity**: Connection testing
4. **External Services**: Dependency validation
5. **Performance**: Response time validation

### Environment Configuration

Each environment has specific configuration:

- **Development**: Debug enabled, hot reload
- **Staging**: Production-like with monitoring
- **Production**: Optimized, security-hardened

## 🚀 Getting Started

### Prerequisites

1. **Docker & Docker Compose**: Latest stable versions
2. **GitHub Actions**: Enabled in repository
3. **Container Registry**: GitHub Container Registry access
4. **Environment Secrets**: Configured in GitHub repository

### Setup Steps

1. **Clone Repository**:
   ```bash
   git clone https://github.com/username/its-camera-ai.git
   cd its-camera-ai
   ```

2. **Configure Environment Variables**:
   ```bash
   # Copy example configurations
   cp .env.staging.example .env.staging
   cp .env.production.example .env.production
   
   # Update with your values
   nano .env.staging
   nano .env.production
   ```

3. **Configure GitHub Secrets**:
   ```
   STAGING_DATABASE_URL
   STAGING_REDIS_URL  
   STAGING_SECRET_KEY
   PRODUCTION_DATABASE_URL
   PRODUCTION_REDIS_URL
   PRODUCTION_SECRET_KEY
   SLACK_WEBHOOK_URL (optional)
   ```

4. **Test Local Development**:
   ```bash
   docker-compose up -d
   curl http://localhost:8000/health
   ```

## 📝 Usage Examples

### Development Workflow

```bash
# Start development environment
docker-compose -f docker/docker-compose.dev.yml up -d

# Run tests locally
docker-compose -f docker/docker-compose.test.yml up unit-tests

# Deploy to staging
git push origin develop  # Triggers staging deployment
```

### Production Deployment

```bash
# Create release tag
git tag v1.2.3
git push origin v1.2.3  # Triggers release workflow

# Manual production deployment
./scripts/deploy.sh -e production -v v1.2.3 -s blue-green
```

### CI/CD Pipeline Triggers

| Trigger | Action | Environment |
|---------|--------|-------------|
| Push to `develop` | CI tests + staging deployment | Staging |
| Push to `main` | CI tests + production deployment | Production |
| Tag `v*` | Full release pipeline | All |
| Pull Request | CI tests only | None |

### Testing Commands

```bash
# Run all tests
docker-compose -f docker/docker-compose.test.yml up

# Individual test suites
docker-compose -f docker/docker-compose.test.yml up unit-tests
docker-compose -f docker/docker-compose.test.yml up integration-tests  
docker-compose -f docker/docker-compose.test.yml up security-tests
docker-compose -f docker/docker-compose.test.yml up performance-tests

# Web E2E tests
cd web && yarn test:e2e:docker
```

## ⚙️ Configuration

### GitHub Actions Configuration

Key workflow files and their purposes:

- **docker-cicd.yml**: Main Docker CI/CD pipeline
- **deployment.yml**: Automated deployment workflows
- **release.yml**: Release management and tagging

### Environment Variables

#### Required Secrets
```
# Database
STAGING_DATABASE_URL
PRODUCTION_DATABASE_URL

# Cache
STAGING_REDIS_URL  
PRODUCTION_REDIS_URL

# Security
STAGING_SECRET_KEY
PRODUCTION_SECRET_KEY

# Monitoring (optional)
SLACK_WEBHOOK_URL
SENTRY_DSN
```

#### Pipeline Configuration
```
DOCKER_REGISTRY=ghcr.io
IMAGE_NAME=its-camera-ai
COVERAGE_THRESHOLD=90
SECURITY_SCAN_FAIL_ON=high
```

### Docker Compose Profiles

Use profiles to run specific service groups:

```bash
# Development services
docker-compose --profile dev up -d

# GPU-enabled services  
docker-compose --profile gpu up -d

# Monitoring stack
docker-compose --profile monitoring up -d

# Testing infrastructure
docker-compose --profile test up -d
```

## 🔧 Troubleshooting

### Common Issues

#### Build Failures

```bash
# Clear Docker cache
docker builder prune -a

# Rebuild without cache
docker-compose build --no-cache

# Check build logs
docker-compose logs app
```

#### Test Failures

```bash
# Check test logs
docker-compose -f docker/docker-compose.test.yml logs unit-tests

# Run tests with verbose output
docker-compose -f docker/docker-compose.test.yml run unit-tests pytest -v

# Debug failing tests
docker-compose -f docker/docker-compose.test.yml run unit-tests pytest -v -s --pdb
```

#### Deployment Issues

```bash
# Check deployment status
./scripts/deploy.sh -e staging --dry-run

# View service health
docker-compose ps
docker-compose logs app

# Rollback deployment
./scripts/deploy.sh -e staging --rollback
```

#### Network Issues

```bash
# Check network connectivity
docker network ls
docker network inspect its-network

# Test service communication
docker-compose exec app curl http://postgres:5432
docker-compose exec app curl http://redis:6379
```

### Performance Optimization

#### Build Performance
- Use multi-stage builds
- Optimize layer caching
- Use .dockerignore
- Minimize context size

#### Test Performance  
- Parallelize test execution
- Use test database fixtures
- Cache dependencies
- Optimize test data

#### Deployment Performance
- Use health checks
- Implement graceful shutdowns
- Configure resource limits
- Monitor deployment metrics

### Security Best Practices

#### Image Security
- Use minimal base images
- Scan for vulnerabilities
- Use non-root users
- Sign images

#### Secret Management
- Use environment variables
- Rotate secrets regularly
- Limit secret access
- Audit secret usage

#### Network Security
- Use private networks
- Implement TLS/SSL
- Configure firewalls
- Monitor network traffic

## 📊 Monitoring and Observability

### Metrics Collection

The pipeline collects comprehensive metrics:

- **Build Metrics**: Build time, success rate
- **Test Metrics**: Coverage, test duration
- **Deployment Metrics**: Deployment frequency, lead time
- **Application Metrics**: Performance, errors, usage

### Alerting

Configure alerts for:
- Build failures
- Test failures  
- Security vulnerabilities
- Deployment failures
- Performance degradation

### Dashboards

Monitor using:
- GitHub Actions dashboard
- Grafana dashboards
- Application metrics
- Infrastructure metrics

## 🤝 Contributing

### Workflow Modifications

1. Create feature branch
2. Modify workflow files
3. Test changes thoroughly
4. Create pull request
5. Get approval from platform team

### Adding New Tests

1. Create test files in appropriate directory
2. Update test configuration
3. Verify test execution
4. Update documentation

### Environment Changes

1. Update environment configuration
2. Test in development
3. Deploy to staging
4. Validate in production

---

For additional support, consult the [main project documentation](README.md) or contact the platform engineering team.