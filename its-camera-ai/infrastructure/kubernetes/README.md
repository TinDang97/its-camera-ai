# ITS Camera AI - Kubernetes Infrastructure

Comprehensive Kubernetes infrastructure for the ITS Camera AI traffic monitoring system, designed to support 1000+ concurrent camera streams with sub-100ms ML inference latency.

## Architecture Overview

The infrastructure implements a **production-grade, zero-trust microservices architecture** with:

- **FastAPI-based API** with Redis caching and PostgreSQL persistence
- **YOLO11 ML inference** with TensorRT optimization on GPU nodes
- **Real-time stream processing** for 1000+ concurrent camera feeds
- **Event-driven architecture** with Kafka message queues
- **Auto-scaling** based on CPU, memory, GPU utilization, and custom metrics
- **Zero-trust security** with network policies, RBAC, and encryption
- **High availability** with multi-zone deployment and pod disruption budgets

## File Structure

```
infrastructure/kubernetes/
├── namespaces.yaml           # Multi-environment namespace configuration
├── configmaps.yaml           # Application configuration
├── secrets.yaml              # Sensitive data and credentials
├── deployments.yaml          # Main application deployments
├── statefulsets.yaml         # PostgreSQL and Redis with persistence
├── services.yaml             # Network services and load balancers
├── ingress.yaml              # TLS termination and routing
├── network-policies.yaml     # Zero-trust network security
├── rbac.yaml                 # Role-based access control
├── storage.yaml              # Persistent storage and backup
├── monitoring.yaml           # Prometheus monitoring and alerting
├── hpa.yaml                  # Horizontal Pod Autoscalers
├── resource-quotas.yaml      # Resource management and quotas
├── auto-scaling.yaml         # Advanced scaling configurations
├── cluster-architecture.yaml # Cluster-level configuration
└── README.md                 # This documentation
```

## Quick Start

### Prerequisites

- Kubernetes cluster (1.25+) with GPU support
- kubectl configured with cluster access
- Helm 3.x installed
- NGINX Ingress Controller
- Prometheus Operator
- cert-manager for TLS certificates
- KEDA for advanced autoscaling

### Deployment Steps

1. **Create namespaces:**
   ```bash
   kubectl apply -f namespaces.yaml
   ```

2. **Setup secrets (replace placeholders with actual values):**
   ```bash
   # Edit secrets.yaml with actual base64-encoded values
   kubectl apply -f secrets.yaml
   ```

3. **Deploy storage layer:**
   ```bash
   kubectl apply -f storage.yaml
   kubectl apply -f statefulsets.yaml
   ```

4. **Deploy application services:**
   ```bash
   kubectl apply -f configmaps.yaml
   kubectl apply -f rbac.yaml
   kubectl apply -f deployments.yaml
   kubectl apply -f services.yaml
   ```

5. **Configure networking:**
   ```bash
   kubectl apply -f network-policies.yaml
   kubectl apply -f ingress.yaml
   ```

6. **Enable monitoring and autoscaling:**
   ```bash
   kubectl apply -f monitoring.yaml
   kubectl apply -f hpa.yaml
   kubectl apply -f resource-quotas.yaml
   ```

7. **Verify deployment:**
   ```bash
   kubectl get pods -n its-camera-ai
   kubectl get services -n its-camera-ai
   kubectl get ingress -n its-camera-ai
   ```

## Component Details

### Core Services

#### API Service
- **Replicas:** 3-20 (auto-scaled)
- **Resources:** 500m CPU, 512Mi RAM (requests)
- **Ports:** 8000 (HTTP), 9090 (metrics)
- **Health checks:** `/health`, `/ready`

#### ML Inference Service
- **Replicas:** 6-50 (auto-scaled)
- **Resources:** 2 CPU, 4Gi RAM, 1 GPU (requests)
- **Node selector:** GPU nodes with NVIDIA T4
- **Latency target:** <100ms P95

#### Stream Processor
- **Replicas:** 10-100 (auto-scaled)
- **Resources:** 1 CPU, 2Gi RAM (requests)
- **Capacity:** 10 cameras per pod
- **Protocols:** RTSP, WebRTC, HTTP streaming

#### Event Processor
- **Replicas:** 5-30 (auto-scaled)
- **Resources:** 500m CPU, 1Gi RAM (requests)
- **Queue:** Kafka-based event processing

### Data Layer

#### PostgreSQL
- **Configuration:** Primary with streaming replication
- **Storage:** 500Gi fast SSD
- **Resources:** 1-4 CPU, 2-8Gi RAM
- **Backup:** Daily automated backups to S3

#### Redis
- **Configuration:** Master-replica with Sentinel
- **Storage:** 100Gi (master), 50Gi (replica)
- **Resources:** 500m-2 CPU, 1-4Gi RAM
- **Use cases:** Caching, session storage, queue management

### Scaling Configuration

#### Horizontal Pod Autoscaler (HPA)
- **API:** 2-20 replicas based on CPU (70%), memory (80%)
- **ML Inference:** 6-50 replicas based on GPU utilization, queue depth
- **Stream Processor:** 10-100 replicas based on active streams
- **Event Processor:** 5-30 replicas based on Kafka lag

#### Vertical Pod Autoscaler (VPA)
- **PostgreSQL:** Auto-sizing with 500m-4 CPU, 2-16Gi RAM
- **Update mode:** Auto with rolling updates

#### Cluster Autoscaler
- **Node pools:** ML inference (6-20), General (9-30), Database (3-9)
- **Scale-down delay:** 10 minutes
- **Utilization threshold:** 50%

### Security Implementation

#### Zero-Trust Network Policies
- **Default deny:** All traffic blocked by default
- **Micro-segmentation:** Service-to-service communication rules
- **Ingress control:** Only necessary external access
- **Egress control:** Limited external communication

#### RBAC Configuration
- **Service accounts:** Per-service with minimal permissions
- **Roles:** Fine-grained access control
- **AWS IAM integration:** Service account role bindings

#### Secret Management
- **External Secrets Operator:** AWS Secrets Manager integration
- **Encryption:** All secrets encrypted at rest
- **Rotation:** Automated secret rotation

### Monitoring and Observability

#### Prometheus Monitoring
- **ServiceMonitors:** API, ML inference, database, cache
- **PodMonitors:** Application-level metrics
- **Custom metrics:** Business logic and performance

#### Grafana Dashboards
- **System overview:** High-level health and performance
- **ML inference:** Model performance and GPU utilization
- **Traffic analytics:** Camera streams and detection rates

#### Alerting Rules
- **Performance:** Latency, error rates, throughput
- **Infrastructure:** Resource usage, pod health
- **Business logic:** Camera streams, model accuracy

### Storage Strategy

#### Storage Classes
- **fast-ssd:** High-performance GP3 with 16K IOPS
- **standard-ssd:** General-purpose GP3
- **gpu-local-nvme:** Local NVMe for GPU nodes
- **archive-storage:** Throughput-optimized for backups

#### Backup Strategy
- **PostgreSQL:** Daily pg_dump + volume snapshots
- **Redis:** Daily RDB dumps + volume snapshots
- **Retention:** 7 days local, 30 days S3, 1 year archive

## Performance Targets

### Latency Requirements
- **API response time:** <500ms P95
- **ML inference:** <100ms P95
- **Stream processing:** <50ms frame processing
- **Database queries:** <100ms P95

### Throughput Targets
- **Concurrent cameras:** 1000+
- **Inference requests:** 10,000+ per second
- **API requests:** 5,000+ per second
- **Event processing:** 1,000+ events per second

### Availability Requirements
- **Uptime SLA:** 99.9%
- **Recovery time:** <5 minutes
- **Data loss tolerance:** <1 minute

## Resource Planning

### Production Environment
- **CPU:** 100 cores (requests), 200 cores (limits)
- **Memory:** 400Gi (requests), 800Gi (limits)
- **GPU:** 30 NVIDIA T4 GPUs
- **Storage:** 10Ti persistent storage
- **Network:** 10Gbps bandwidth

### Node Pool Sizing
- **ML Inference:** 6-20 nodes (g4dn.2xlarge)
- **General workloads:** 9-30 nodes (c5.4xlarge)
- **Database:** 3-9 nodes (r5.2xlarge)
- **Edge gateway:** 3-12 nodes (m5.xlarge)

## Operational Procedures

### Deployment
```bash
# Zero-downtime deployment
kubectl rollout restart deployment/api-deployment -n its-camera-ai
kubectl rollout status deployment/api-deployment -n its-camera-ai

# ML model updates
kubectl patch deployment ml-inference-deployment -n its-camera-ai \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"ml-inference","image":"its-camera-ai/ml-inference:v1.1.0"}]}}}}'
```

### Scaling
```bash
# Manual scaling
kubectl scale deployment api-deployment --replicas=10 -n its-camera-ai

# Check HPA status
kubectl get hpa -n its-camera-ai

# View scaling events
kubectl describe hpa api-hpa -n its-camera-ai
```

### Monitoring
```bash
# Check pod resources
kubectl top pods -n its-camera-ai

# View logs
kubectl logs -f deployment/api-deployment -n its-camera-ai

# Check service endpoints
kubectl get endpoints -n its-camera-ai
```

### Troubleshooting
```bash
# Debug network policies
kubectl describe networkpolicy -n its-camera-ai

# Check resource quotas
kubectl describe resourcequota -n its-camera-ai

# Verify RBAC permissions
kubectl auth can-i --list --as=system:serviceaccount:its-camera-ai:api
```

## Security Considerations

### Network Security
- All pods follow zero-trust network policies
- Inter-service communication over mTLS (Istio)
- External traffic through authenticated ingress
- No direct pod-to-pod communication

### Data Protection
- All data encrypted at rest and in transit
- Video streams anonymized for privacy
- GDPR/CCPA compliance built-in
- Audit logging for all operations

### Access Control
- Multi-factor authentication required
- Role-based access with least privilege
- Service account tokens auto-rotated
- API rate limiting and DDoS protection

## Disaster Recovery

### Backup Strategy
- **RTO:** 5 minutes
- **RPO:** 1 minute
- **Multi-region:** Active-passive setup
- **Automated failover:** DNS-based

### Recovery Procedures
1. **Database recovery:** Point-in-time restore from S3 backups
2. **Application recovery:** Rolling deployment from container registry
3. **Configuration recovery:** GitOps-based infrastructure as code
4. **Network recovery:** Automated DNS failover

## Cost Optimization

### Resource Efficiency
- **Spot instances:** Non-critical workloads
- **Vertical scaling:** Right-sizing based on usage
- **Scheduled scaling:** Time-based traffic patterns
- **Storage tiering:** Hot, warm, cold data classification

### Cost Monitoring
- **Resource quotas:** Prevent runaway costs
- **Usage alerts:** Threshold-based notifications
- **Cost allocation:** Per-service cost tracking
- **Optimization recommendations:** Automated suggestions

## Compliance and Governance

### Standards Compliance
- **SOC 2 Type II:** Security and availability controls
- **ISO 27001:** Information security management
- **GDPR/CCPA:** Data privacy and protection
- **NIST Cybersecurity Framework:** Risk management

### Audit Requirements
- **Access logging:** All user and system access
- **Change tracking:** Infrastructure and configuration changes
- **Data lineage:** Video data processing and storage
- **Compliance reporting:** Automated compliance checks

## Support and Maintenance

### Regular Maintenance
- **Security patches:** Monthly Kubernetes and OS updates
- **Certificate renewal:** Automated via cert-manager
- **Backup verification:** Weekly restore testing
- **Performance tuning:** Quarterly resource optimization

### Support Contacts
- **Platform team:** platform-team@its-camera-ai.com
- **Security team:** security-team@its-camera-ai.com
- **Operations team:** ops-team@its-camera-ai.com
- **Emergency escalation:** +1-800-ITS-HELP

---

*This infrastructure is designed for production deployment of the ITS Camera AI traffic monitoring system with enterprise-grade security, reliability, and scalability.*
