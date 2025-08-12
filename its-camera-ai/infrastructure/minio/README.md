# MinIO Object Storage for ITS Camera AI

This directory contains comprehensive MinIO deployment configurations for the ITS Camera AI system, providing high-performance, scalable object storage for camera streams, ML models, analytics data, and backups.

## Overview

MinIO is deployed in distributed mode with:
- **High Availability**: Multi-node cluster with automatic failover
- **Scalability**: Horizontal Pod Autoscaler based on CPU, memory, and custom metrics
- **Security**: Zero-trust networking, encryption at rest and in transit
- **Monitoring**: Comprehensive Prometheus metrics and Grafana dashboards
- **Compliance**: Audit logging and data governance features

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ITS Camera AI System                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Camera      │  │ ML Models   │  │ Analytics   │             │
│  │ Streams     │  │ Storage     │  │ Data        │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                     MinIO Distributed Cluster                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Node 1    │  │   Node 2    │  │   Node N    │             │
│  │ 4x 500GB    │  │ 4x 500GB    │  │ 4x 500GB    │             │
│  │ Drives      │  │ Drives      │  │ Drives      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│                    Kubernetes Infrastructure                     │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Helm 3.x (for Helm deployment)
- Terraform 1.x (for Terraform deployment)
- Storage class with fast SSDs (recommended)

### Deploy with Script

```bash
# Deploy to development environment
./deploy-minio.sh -e development

# Deploy to production with monitoring
./deploy-minio.sh -e production -m terraform

# Dry run deployment
./deploy-minio.sh -e staging --dry-run

# Deploy without tests
./deploy-minio.sh -e development --skip-tests
```

### Manual Deployment

#### Using Helm

```bash
cd infrastructure/helm/its-camera-ai
helm dependency update
helm install its-camera-ai . --namespace its-camera-ai --create-namespace \
  --set minio.enabled=true \
  --set minio.statefulset.replicaCount=4 \
  --set minio.persistence.size=500Gi
```

#### Using Terraform

```bash
cd infrastructure/terraform
terraform init
terraform workspace select production
terraform plan -var-file="environments/production.tfvars"
terraform apply -var-file="environments/production.tfvars"
```

#### Using kubectl

```bash
kubectl apply -f infrastructure/kubernetes/minio-statefulset.yaml
kubectl apply -f infrastructure/kubernetes/minio-hpa.yaml
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MINIO_ROOT_USER` | MinIO admin username | `its_camera_ai_admin` |
| `MINIO_ROOT_PASSWORD` | MinIO admin password | Generated |
| `MINIO_STORAGE_CLASS_STANDARD` | Default storage class | `EC:2` |
| `MINIO_API_REQUESTS_MAX` | Max concurrent requests | `10000` |
| `MINIO_AUDIT_WEBHOOK_ENABLE_target1` | Enable audit logging | `on` |

### Storage Classes

- **Development**: `standard` (100Gi per drive)
- **Staging**: `fast-ssd` (250Gi per drive)
- **Production**: `fast-ssd` (500Gi per drive)

### Replica Configuration

| Environment | Nodes | Drives/Node | Total Storage |
|-------------|-------|-------------|---------------|
| Development | 4 | 2 | 800Gi |
| Staging | 6 | 3 | 4.5Ti |
| Production | 8 | 4 | 16Ti |

## Bucket Structure

### Default Buckets

- **camera-streams**: Real-time camera video data
  - Lifecycle: 30 days retention
  - Policy: Private access
  - Versioning: Disabled

- **ml-models**: Machine learning models and artifacts
  - Lifecycle: Permanent retention
  - Policy: Read-only for applications
  - Versioning: Enabled

- **analytics**: Processed analytics and reports
  - Lifecycle: 90 days retention
  - Policy: Private access
  - Versioning: Disabled

- **backups**: System and data backups
  - Lifecycle: 30 days retention
  - Policy: Private access
  - Versioning: Enabled

- **temp**: Temporary processing files
  - Lifecycle: 7 days retention
  - Policy: Public read
  - Versioning: Disabled

### Access Patterns

```bash
# Camera stream processor service
BUCKET: camera-streams
OPERATIONS: PUT, GET, DELETE
ACCESS: Write video frames, read for processing

# ML inference service  
BUCKET: ml-models
OPERATIONS: GET
ACCESS: Read model files for inference

# Analytics service
BUCKET: analytics, camera-streams
OPERATIONS: GET, PUT
ACCESS: Read raw data, write processed results

# Backup service
BUCKET: backups
OPERATIONS: PUT, GET
ACCESS: Write system backups, read for restore
```

## Security

### Authentication

- **Root User**: Administrative access with strong password
- **Service Accounts**: Dedicated accounts per service with minimal permissions
- **IAM Policies**: Fine-grained access control per bucket and operation

### Network Security

- **Network Policies**: Restrict traffic between pods
- **TLS Encryption**: HTTPS/TLS for all API communication
- **VPC Isolation**: Private subnets for storage traffic

### Audit Logging

All access is logged to:
- Application audit webhook
- CloudTrail (for AWS deployments)
- Kubernetes audit logs

## Monitoring

### Metrics

Key metrics monitored:
- **Cluster Health**: Node availability, quorum status
- **Performance**: Request latency, throughput, error rates
- **Storage**: Disk usage, capacity planning
- **Security**: Failed authentication attempts, policy violations

### Alerts

Critical alerts configured for:
- Node failures
- Disk space thresholds (85% warning, 95% critical)
- High error rates (>5%)
- Request latency (>1s)
- Quorum loss

### Dashboards

Grafana dashboards available for:
- MinIO cluster overview
- Performance metrics
- Storage utilization
- Error tracking

## Backup and Recovery

### Automatic Backups

- **Schedule**: Daily at 3 AM
- **Retention**: 30 days (configurable)
- **Storage**: AWS S3 with versioning
- **Verification**: Backup integrity checks

### Disaster Recovery

- **RTO**: 15 minutes (Recovery Time Objective)
- **RPO**: 1 hour (Recovery Point Objective)
- **Multi-Region**: Cross-region replication for production
- **Failover**: Automatic DNS failover

## Performance Optimization

### Hardware Recommendations

- **CPU**: 4+ cores per node
- **Memory**: 8GB+ per node
- **Storage**: NVMe SSDs for best performance
- **Network**: 10Gbps+ for large deployments

### Tuning Parameters

```yaml
# High-performance configuration
MINIO_STORAGE_CLASS_STANDARD: "EC:2"
MINIO_CACHE_QUOTA: "80"
MINIO_API_REQUESTS_MAX: "10000"
MINIO_API_REQUESTS_DEADLINE: "10s"
```

### Scaling Strategies

1. **Vertical Scaling**: Increase node resources
2. **Horizontal Scaling**: Add more nodes (4-node increments)
3. **Storage Scaling**: Add drives per node
4. **Network Optimization**: Dedicated storage network

## Troubleshooting

### Common Issues

#### Pod Startup Failures

```bash
# Check pod logs
kubectl logs -n its-camera-ai minio-0

# Check storage class
kubectl get storageclass

# Verify PVC binding
kubectl get pvc -n its-camera-ai
```

#### Connectivity Issues

```bash
# Test internal connectivity
kubectl run test-pod --image=curlimages/curl --rm -i --restart=Never -- \
  curl -f http://minio-service.its-camera-ai.svc.cluster.local:9000/minio/health/live

# Check service endpoints
kubectl get endpoints -n its-camera-ai minio-service
```

#### Performance Issues

```bash
# Check resource usage
kubectl top pods -n its-camera-ai -l app.kubernetes.io/name=minio

# Monitor request patterns
kubectl port-forward -n its-camera-ai svc/minio-service 9000:9000
# Access http://localhost:9000 for MinIO Console
```

### Support

For issues and support:
1. Check application logs
2. Review Kubernetes events
3. Consult MinIO documentation
4. Contact ITS Camera AI support team

## API Reference

### MinIO Client Configuration

```bash
# Configure mc client
mc alias set its-minio http://minio-service.its-camera-ai.svc.cluster.local:9000 ACCESS_KEY SECRET_KEY

# List buckets
mc ls its-minio

# Upload file
mc cp /path/to/file its-minio/bucket-name/

# Download file
mc cp its-minio/bucket-name/file /path/to/destination
```

### S3 Compatible API

```python
import boto3

# Configure S3 client for MinIO
s3_client = boto3.client(
    's3',
    endpoint_url='http://minio-service.its-camera-ai.svc.cluster.local:9000',
    aws_access_key_id='ACCESS_KEY',
    aws_secret_access_key='SECRET_KEY'
)

# Upload file
s3_client.upload_file('/path/to/file', 'bucket-name', 'object-key')

# Download file
s3_client.download_file('bucket-name', 'object-key', '/path/to/destination')
```

## Development

### Local Testing

```bash
# Start local MinIO with Docker Compose
docker-compose up -d minio minio-init

# Access MinIO Console
open http://localhost:9001

# Run integration tests
kubectl apply -f infrastructure/testing/minio-integration-tests.yaml
```

### Contributing

1. Follow existing configuration patterns
2. Update environment-specific variables
3. Add appropriate monitoring and alerts
4. Test thoroughly in development environment
5. Update documentation

## License

This configuration is part of the ITS Camera AI project and follows the project's licensing terms.