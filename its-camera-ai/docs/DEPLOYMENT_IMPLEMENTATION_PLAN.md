# ITS Camera AI - Deployment & Infrastructure Implementation Plan

**Document Version:** 1.0  
**Date:** August 12, 2025  
**Author:** Platform Engineering Team  
**Complements:** IMPLEMENTATION_PLAN.md

---

## Executive Summary

This document provides a comprehensive deployment and infrastructure implementation plan for the ITS Camera AI system, focusing on platform engineering best practices, self-service capabilities, and operational excellence. It complements the component implementation plan by defining the infrastructure foundation, deployment strategies, and operational procedures required for production readiness.

## Table of Contents

1. [Infrastructure Architecture Overview](#1-infrastructure-architecture-overview)
2. [Infrastructure as Code Implementation](#2-infrastructure-as-code-implementation)
3. [Platform Services Setup](#3-platform-services-setup)
4. [Database Infrastructure](#4-database-infrastructure)
5. [Monitoring Stack Implementation](#5-monitoring-stack-implementation)
6. [CI/CD Pipeline Implementation](#6-cicd-pipeline-implementation)
7. [Edge Deployment Strategy](#7-edge-deployment-strategy)
8. [Scaling and Performance](#8-scaling-and-performance)
9. [Security Implementation](#9-security-implementation)
10. [Disaster Recovery](#10-disaster-recovery)
11. [Operations Runbook](#11-operations-runbook)
12. [Implementation Timeline](#12-implementation-timeline)

---

## 1. Infrastructure Architecture Overview

### 1.1 Deployment Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                        Cloud Control Plane                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Production    │  │     Staging     │  │   Development   │ │
│  │   Cluster       │  │    Cluster      │  │    Cluster      │ │
│  │                 │  │                 │  │                 │ │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │ │
│  │ │ GPU Nodes   │ │  │ │ GPU Nodes   │ │  │ │ CPU Nodes   │ │ │
│  │ │ (inference) │ │  │ │ (testing)   │ │  │ │ (dev work)  │ │ │
│  │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Edge Nodes                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Traffic Hub A  │  │  Traffic Hub B  │  │  Traffic Hub N  │ │
│  │   (K3s/MicroK8s)│  │   (K3s/MicroK8s)│  │   (K3s/MicroK8s)│ │
│  │                 │  │                 │  │                 │ │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │ │
│  │ │ Edge        │ │  │ │ Edge        │ │  │ │ Edge        │ │ │
│  │ │ Inference   │ │  │ │ Inference   │ │  │ │ Inference   │ │ │
│  │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack Matrix

| Layer | Technology | Purpose | High Availability |
|-------|------------|---------|-------------------|
| **Container Runtime** | containerd | Container execution | Multi-node redundancy |
| **Orchestration** | Kubernetes 1.31+ | Workload management | 3+ control plane nodes |
| **Edge Orchestration** | K3s 1.31+ | Edge deployment | Single node + failover |
| **Service Mesh** | Istio 1.23+ | Traffic management | Multi-zone deployment |
| **API Gateway** | Kong 3.8+ | External traffic | Load balanced |
| **Storage** | Longhorn/Rook-Ceph | Persistent storage | 3+ replicas |
| **Networking** | Cilium | CNI with eBPF | Multi-zone |
| **Load Balancer** | MetalLB/Cloud LB | Traffic distribution | Active-Active |

### 1.3 Platform Services Overview

| Service Category | Components | Purpose |
|------------------|------------|---------|
| **Infrastructure** | Terraform, Crossplane, ArgoCD | Infrastructure provisioning & GitOps |
| **Platform APIs** | Backstage, Platform APIs | Developer self-service portal |
| **Observability** | Prometheus, Grafana, Loki, Jaeger | Monitoring and observability |
| **Security** | Vault, cert-manager, Falco | Secrets, certificates, security |
| **Data** | PostgreSQL, Redis, TimescaleDB, Kafka | Data persistence and streaming |
| **ML Infrastructure** | Kubeflow, MLflow, ONNX Runtime | ML pipeline and model serving |

---

## 2. Infrastructure as Code Implementation

### 2.1 Terraform Module Structure

```
infrastructure/
├── modules/
│   ├── kubernetes-cluster/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── versions.tf
│   ├── database/
│   │   ├── postgresql/
│   │   ├── redis/
│   │   ├── timescaledb/
│   │   └── kafka/
│   ├── monitoring/
│   │   ├── prometheus/
│   │   ├── grafana/
│   │   └── loki/
│   ├── security/
│   │   ├── vault/
│   │   ├── cert-manager/
│   │   └── network-policies/
│   └── ml-infrastructure/
│       ├── gpu-nodes/
│       ├── model-registry/
│       └── inference-services/
├── environments/
│   ├── development/
│   │   ├── main.tf
│   │   ├── terraform.tfvars
│   │   └── backend.tf
│   ├── staging/
│   │   ├── main.tf
│   │   ├── terraform.tfvars
│   │   └── backend.tf
│   └── production/
│       ├── main.tf
│       ├── terraform.tfvars
│       └── backend.tf
└── scripts/
    ├── deploy.sh
    ├── destroy.sh
    └── validate.sh
```

### 2.2 Kubernetes Manifests Organization

```
k8s/
├── base/
│   ├── namespaces/
│   ├── rbac/
│   ├── network-policies/
│   └── priority-classes/
├── platform/
│   ├── istio/
│   ├── cert-manager/
│   ├── vault/
│   ├── argocd/
│   └── monitoring/
├── applications/
│   ├── camera-service/
│   │   ├── base/
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   ├── configmap.yaml
│   │   │   └── kustomization.yaml
│   │   └── overlays/
│   │       ├── development/
│   │       ├── staging/
│   │       └── production/
│   ├── analytics-service/
│   ├── streaming-service/
│   └── vision-engine/
└── data/
    ├── postgresql/
    ├── redis/
    ├── timescaledb/
    └── kafka/
```

### 2.3 Helm Charts Structure

```
charts/
├── its-camera-ai/
│   ├── Chart.yaml
│   ├── values.yaml
│   ├── values-dev.yaml
│   ├── values-staging.yaml
│   ├── values-prod.yaml
│   ├── templates/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── ingress.yaml
│   │   ├── configmap.yaml
│   │   ├── secret.yaml
│   │   ├── hpa.yaml
│   │   ├── pdb.yaml
│   │   └── serviceaccount.yaml
│   └── charts/
│       ├── postgresql/
│       ├── redis/
│       └── kafka/
├── monitoring/
│   ├── prometheus-operator/
│   ├── grafana/
│   └── loki-stack/
└── security/
    ├── vault/
    └── cert-manager/
```

### 2.4 GitOps Workflow Implementation

#### ArgoCD Application Structure

```yaml
# argocd/applications/its-camera-ai-prod.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: its-camera-ai-production
  namespace: argocd
  labels:
    environment: production
    component: its-camera-ai
spec:
  project: its-camera-ai
  source:
    repoURL: https://github.com/its-team/its-camera-ai-infra
    targetRevision: HEAD
    path: k8s/applications/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: its-camera-ai-prod
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

#### Environment Configuration Strategy

```yaml
# environments/production/config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: environment-config
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  DEBUG: "false"
  DATABASE_POOL_SIZE: "20"
  REDIS_MAX_CONNECTIONS: "100"
  ML_BATCH_SIZE: "32"
  INFERENCE_TIMEOUT: "100ms"
  GPU_MEMORY_LIMIT: "8Gi"
  AUTO_SCALING_TARGET_CPU: "70"
  AUTO_SCALING_TARGET_MEMORY: "80"
```

---

## 3. Platform Services Setup

### 3.1 Service Mesh Implementation (Istio)

#### Installation and Configuration

```bash
# Install Istio
curl -L https://istio.io/downloadIstio | sh -
istioctl install --set values.defaultRevision=default

# Enable automatic sidecar injection
kubectl label namespace its-camera-ai-prod istio-injection=enabled
```

#### Traffic Management Configuration

```yaml
# istio/virtual-service.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: its-camera-ai-vs
  namespace: its-camera-ai-prod
spec:
  hosts:
  - its-camera-ai.example.com
  gateways:
  - its-camera-ai-gateway
  http:
  - match:
    - uri:
        prefix: "/api/v1/camera"
    route:
    - destination:
        host: camera-service
        port:
          number: 8080
    timeout: 100ms
    retries:
      attempts: 3
      perTryTimeout: 30ms
  - match:
    - uri:
        prefix: "/api/v1/analytics"
    route:
    - destination:
        host: analytics-service
        port:
          number: 8080
    timeout: 5s
```

### 3.2 API Gateway Configuration (Kong)

```yaml
# kong/kong-ingress.yaml
apiVersion: configuration.konghq.com/v1
kind: KongIngress
metadata:
  name: its-camera-ai-ingress
proxy:
  connect_timeout: 10000
  read_timeout: 60000
  write_timeout: 60000
route:
  strip_path: false
  preserve_host: true
upstream:
  algorithm: round-robin
  healthchecks:
    active:
      healthy:
        interval: 10
        successes: 3
      unhealthy:
        interval: 10
        http_failures: 3
```

### 3.3 Certificate Management (cert-manager)

```yaml
# cert-manager/cluster-issuer.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: devops@its-camera-ai.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: kong
```

### 3.4 Secret Management (Vault)

```yaml
# vault/vault-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vault-config
data:
  vault.hcl: |
    ui = true
    
    storage "postgresql" {
      connection_url = "postgres://vault:password@postgresql:5432/vault?sslmode=disable"
      table = "vault_kv_store"
      max_parallel = 128
    }
    
    listener "tcp" {
      address = "0.0.0.0:8200"
      tls_disable = false
      tls_cert_file = "/vault/tls/server.crt"
      tls_key_file = "/vault/tls/server.key"
    }
    
    seal "gcpckms" {
      project = "its-camera-ai"
      region = "global"
      key_ring = "vault-seal"
      crypto_key = "vault-key"
    }
```

---

## 4. Database Infrastructure

### 4.1 PostgreSQL Cluster Setup

```yaml
# postgresql/postgresql-cluster.yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: postgresql-cluster
  namespace: its-camera-ai-prod
spec:
  instances: 3
  primaryUpdateStrategy: unsupervised
  
  postgresql:
    parameters:
      max_connections: "400"
      shared_buffers: "256MB"
      effective_cache_size: "1GB"
      maintenance_work_mem: "64MB"
      checkpoint_completion_target: "0.9"
      wal_buffers: "16MB"
      default_statistics_target: "100"
      random_page_cost: "1.1"
      effective_io_concurrency: "200"
      work_mem: "4MB"
      min_wal_size: "1GB"
      max_wal_size: "4GB"
      
  resources:
    requests:
      memory: "2Gi"
      cpu: "1"
    limits:
      memory: "4Gi"
      cpu: "2"
      
  storage:
    size: 100Gi
    storageClass: fast-ssd
    
  monitoring:
    enabled: true
    
  backup:
    retentionPolicy: "30d"
    barmanObjectStore:
      destinationPath: "s3://its-camera-ai-backups/postgresql"
      s3Credentials:
        accessKeyId:
          name: postgresql-backup-credentials
          key: ACCESS_KEY_ID
        secretAccessKey:
          name: postgresql-backup-credentials
          key: SECRET_ACCESS_KEY
      wal:
        retention: "7d"
      data:
        retention: "30d"
```

### 4.2 TimescaleDB Configuration

```yaml
# timescaledb/timescaledb-deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: timescaledb
  namespace: its-camera-ai-prod
spec:
  serviceName: timescaledb
  replicas: 3
  selector:
    matchLabels:
      app: timescaledb
  template:
    metadata:
      labels:
        app: timescaledb
    spec:
      containers:
      - name: timescaledb
        image: timescale/timescaledb-ha:pg16
        env:
        - name: POSTGRES_DB
          value: metrics
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: timescaledb-credentials
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: timescaledb-credentials
              key: password
        - name: TIMESCALEDB_TELEMETRY
          value: "off"
        ports:
        - containerPort: 5432
        resources:
          requests:
            cpu: "500m"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
        - name: config
          mountPath: /etc/postgresql/postgresql.conf
          subPath: postgresql.conf
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 200Gi
```

### 4.3 Redis Cluster Configuration

```yaml
# redis/redis-cluster.yaml
apiVersion: redis.redis.opstreelabs.in/v1beta1
kind: RedisCluster
metadata:
  name: redis-cluster
  namespace: its-camera-ai-prod
spec:
  clusterSize: 6
  clusterVersion: v7
  persistenceEnabled: true
  redisExporter:
    enabled: true
    image: quay.io/opstree/redis-exporter:latest
  storage:
    volumeClaimTemplate:
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 50Gi
        storageClassName: fast-ssd
  resources:
    requests:
      cpu: "100m"
      memory: "512Mi"
    limits:
      cpu: "1"
      memory: "2Gi"
```

### 4.4 Kafka Cluster Setup

```yaml
# kafka/kafka-cluster.yaml
apiVersion: kafka.strimzi.io/v1beta2
kind: Kafka
metadata:
  name: kafka-cluster
  namespace: its-camera-ai-prod
spec:
  kafka:
    version: 3.7.0
    replicas: 3
    listeners:
      - name: plain
        port: 9092
        type: internal
        tls: false
      - name: tls
        port: 9093
        type: internal
        tls: true
    config:
      offsets.topic.replication.factor: 3
      transaction.state.log.replication.factor: 3
      transaction.state.log.min.isr: 2
      default.replication.factor: 3
      min.insync.replicas: 2
      inter.broker.protocol.version: "3.7"
    storage:
      type: persistent-claim
      size: 100Gi
      class: fast-ssd
    resources:
      requests:
        cpu: "500m"
        memory: "2Gi"
      limits:
        cpu: "2"
        memory: "4Gi"
  zookeeper:
    replicas: 3
    storage:
      type: persistent-claim
      size: 20Gi
      class: fast-ssd
    resources:
      requests:
        cpu: "200m"
        memory: "1Gi"
      limits:
        cpu: "1"
        memory: "2Gi"
  entityOperator:
    topicOperator: {}
    userOperator: {}
```

### 4.5 MinIO Object Storage

```yaml
# minio/minio-tenant.yaml
apiVersion: minio.min.io/v2
kind: Tenant
metadata:
  name: minio-tenant
  namespace: its-camera-ai-prod
spec:
  image: quay.io/minio/minio:latest
  pools:
  - servers: 4
    name: pool-0
    volumesPerServer: 4
    volumeClaimTemplate:
      metadata:
        name: data
      spec:
        accessModes:
        - ReadWriteOnce
        resources:
          requests:
            storage: 500Gi
        storageClassName: fast-ssd
    resources:
      requests:
        cpu: "500m"
        memory: "2Gi"
      limits:
        cpu: "2"
        memory: "4Gi"
  requestAutoCert: false
  certConfig:
    commonName: "minio-tenant"
    organizationName: ["ITS Camera AI"]
    dnsNames:
    - "minio-tenant.its-camera-ai-prod.svc.cluster.local"
```

---

## 5. Monitoring Stack Implementation

### 5.1 Prometheus Operator Setup

```yaml
# monitoring/prometheus-operator.yaml
apiVersion: monitoring.coreos.com/v1
kind: Prometheus
metadata:
  name: prometheus-main
  namespace: monitoring
spec:
  serviceAccountName: prometheus
  serviceMonitorSelector:
    matchLabels:
      app: its-camera-ai
  ruleSelector:
    matchLabels:
      app: its-camera-ai
  resources:
    requests:
      memory: 4Gi
      cpu: "2"
    limits:
      memory: 8Gi
      cpu: "4"
  retention: 15d
  storage:
    volumeClaimTemplate:
      spec:
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 200Gi
  additionalScrapeConfigs:
    name: additional-scrape-configs
    key: prometheus-additional.yaml
```

### 5.2 Grafana Dashboard Configuration

```yaml
# monitoring/grafana-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards-its-camera-ai
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  its-camera-ai-overview.json: |
    {
      "dashboard": {
        "id": null,
        "title": "ITS Camera AI - System Overview",
        "tags": ["its-camera-ai"],
        "style": "dark",
        "timezone": "browser",
        "panels": [
          {
            "title": "Inference Latency (p99)",
            "type": "stat",
            "targets": [
              {
                "expr": "histogram_quantile(0.99, rate(inference_duration_seconds_bucket[5m]))",
                "legendFormat": "p99 Latency"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "unit": "s",
                "thresholds": {
                  "steps": [
                    {"color": "green", "value": null},
                    {"color": "yellow", "value": 0.08},
                    {"color": "red", "value": 0.1}
                  ]
                }
              }
            }
          },
          {
            "title": "GPU Utilization",
            "type": "graph",
            "targets": [
              {
                "expr": "DCGM_FI_DEV_GPU_UTIL",
                "legendFormat": "GPU {{gpu}}"
              }
            ]
          },
          {
            "title": "Camera Stream Status",
            "type": "table",
            "targets": [
              {
                "expr": "camera_stream_status",
                "legendFormat": "{{camera_id}}"
              }
            ]
          }
        ]
      }
    }
```

### 5.3 Loki Log Aggregation

```yaml
# monitoring/loki-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: loki-config
  namespace: monitoring
data:
  loki.yaml: |
    auth_enabled: false
    
    server:
      http_listen_port: 3100
      grpc_listen_port: 9096
    
    common:
      path_prefix: /loki
      storage:
        filesystem:
          chunks_directory: /loki/chunks
          rules_directory: /loki/rules
      replication_factor: 1
      ring:
        kvstore:
          store: inmemory
    
    query_scheduler:
      max_outstanding_requests_per_tenant: 4096
    
    schema_config:
      configs:
        - from: 2020-10-24
          store: boltdb-shipper
          object_store: filesystem
          schema: v11
          index:
            prefix: index_
            period: 24h
    
    analytics:
      reporting_enabled: false
    
    limits_config:
      retention_period: 168h  # 7 days
      max_query_parallelism: 32
      max_query_series: 10000
```

### 5.4 Jaeger Distributed Tracing

```yaml
# monitoring/jaeger-deployment.yaml
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: jaeger-its-camera-ai
  namespace: monitoring
spec:
  strategy: production
  storage:
    type: elasticsearch
    elasticsearch:
      nodeCount: 3
      storage:
        size: 100Gi
        storageClassName: fast-ssd
      resources:
        requests:
          cpu: "500m"
          memory: "2Gi"
        limits:
          cpu: "2"
          memory: "4Gi"
  collector:
    replicas: 3
    resources:
      requests:
        cpu: "200m"
        memory: "512Mi"
      limits:
        cpu: "1"
        memory: "1Gi"
  query:
    replicas: 2
    resources:
      requests:
        cpu: "200m"
        memory: "512Mi"
      limits:
        cpu: "1"
        memory: "1Gi"
```

### 5.5 SLI/SLO Definitions

```yaml
# monitoring/slo-definitions.yaml
apiVersion: sloth.slok.dev/v1
kind: PrometheusServiceLevel
metadata:
  name: its-camera-ai-slos
  namespace: monitoring
spec:
  service: "its-camera-ai"
  labels:
    team: "its-camera-ai"
  slos:
  - name: "inference-availability"
    objective: 99.9
    description: "99.9% of inference requests should be successful"
    sli:
      events:
        error_query: sum(rate(inference_requests_total{code=~"5.."}[5m]))
        total_query: sum(rate(inference_requests_total[5m]))
    alerting:
      name: InferenceAvailability
      labels:
        severity: critical
      annotations:
        summary: "Inference service availability is below SLO"
        
  - name: "inference-latency"
    objective: 95.0
    description: "95% of inference requests should complete within 100ms"
    sli:
      events:
        error_query: sum(rate(inference_duration_seconds_bucket{le="0.1"}[5m]))
        total_query: sum(rate(inference_duration_seconds_count[5m]))
    alerting:
      name: InferenceLatency
      labels:
        severity: warning
      annotations:
        summary: "Inference latency is above SLO threshold"
```

---

## 6. CI/CD Pipeline Implementation

### 6.1 GitHub Actions Workflow Structure

```yaml
# .github/workflows/ci-cd-main.yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  code-quality:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.12'
    
    - name: Install uv
      uses: astral-sh/setup-uv@v3
      with:
        version: "latest"
    
    - name: Install dependencies
      run: |
        uv sync --group dev
    
    - name: Run linting
      run: |
        uv run ruff check src/ tests/
        uv run black --check src/ tests/
        uv run mypy src/
    
    - name: Run security scan
      run: |
        uv run bandit -r src/
        uv run safety check
    
    - name: Run tests
      run: |
        uv run pytest --cov=src/its_camera_ai \
          --cov-report=xml \
          --cov-report=html \
          --cov-fail-under=90 \
          -m "not slow and not gpu"
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v4
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  build-and-test:
    needs: code-quality
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [camera-service, analytics-service, streaming-service, vision-engine]
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v6
      with:
        context: .
        file: ./docker/Dockerfile.${{ matrix.service }}
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-${{ matrix.service }}:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        platforms: linux/amd64,linux/arm64
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-${{ matrix.service }}:${{ github.sha }}
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v3
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'

  integration-tests:
    needs: build-and-test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Kind cluster
      uses: helm/kind-action@v1.10.0
      with:
        cluster_name: its-camera-ai-test
        config: .github/kind-config.yaml
    
    - name: Install Helm
      uses: azure/setup-helm@v4
      with:
        version: '3.15.4'
    
    - name: Deploy to test cluster
      run: |
        helm upgrade --install its-camera-ai ./charts/its-camera-ai \
          --namespace its-camera-ai-test \
          --create-namespace \
          --values ./charts/its-camera-ai/values-test.yaml \
          --set image.tag=${{ github.sha }}
    
    - name: Run integration tests
      run: |
        kubectl wait --for=condition=ready pod \
          --selector=app.kubernetes.io/name=its-camera-ai \
          --timeout=300s \
          --namespace=its-camera-ai-test
        
        uv run pytest tests/integration/ \
          --k8s-namespace=its-camera-ai-test

  deploy-staging:
    needs: integration-tests
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: staging
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2
    
    - name: Update kubeconfig
      run: |
        aws eks update-kubeconfig --name its-camera-ai-staging
    
    - name: Deploy to staging
      run: |
        kubectl set image deployment/camera-service \
          camera-service=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-camera-service:${{ github.sha }} \
          --namespace=its-camera-ai-staging
        
        kubectl rollout status deployment/camera-service \
          --namespace=its-camera-ai-staging \
          --timeout=300s
    
    - name: Run smoke tests
      run: |
        uv run pytest tests/smoke/ \
          --base-url=https://staging.its-camera-ai.com

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2
    
    - name: Update kubeconfig
      run: |
        aws eks update-kubeconfig --name its-camera-ai-production
    
    - name: Canary deployment
      run: |
        # Deploy to canary environment first
        kubectl set image deployment/camera-service-canary \
          camera-service=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-camera-service:${{ github.sha }} \
          --namespace=its-camera-ai-prod
        
        kubectl rollout status deployment/camera-service-canary \
          --namespace=its-camera-ai-prod \
          --timeout=300s
    
    - name: Validate canary deployment
      run: |
        # Run canary validation tests
        uv run pytest tests/canary/ \
          --base-url=https://canary.its-camera-ai.com
        
        # Check metrics for 5 minutes
        sleep 300
        
        # Validate error rates and latency
        python scripts/validate-canary-metrics.py
    
    - name: Promote to production
      run: |
        kubectl set image deployment/camera-service \
          camera-service=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-camera-service:${{ github.sha }} \
          --namespace=its-camera-ai-prod
        
        kubectl rollout status deployment/camera-service \
          --namespace=its-camera-ai-prod \
          --timeout=600s
```

### 6.2 Progressive Deployment Strategies

#### Blue-Green Deployment Configuration

```yaml
# deployments/blue-green-deployment.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: camera-service-rollout
  namespace: its-camera-ai-prod
spec:
  replicas: 10
  strategy:
    blueGreen:
      activeService: camera-service-active
      previewService: camera-service-preview
      autoPromotionEnabled: false
      scaleDownDelaySeconds: 30
      prePromotionAnalysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: camera-service-preview
      postPromotionAnalysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: camera-service-active
      promotionPolicy:
        timeoutSeconds: 300
  selector:
    matchLabels:
      app: camera-service
  template:
    metadata:
      labels:
        app: camera-service
    spec:
      containers:
      - name: camera-service
        image: ghcr.io/its-team/its-camera-ai-camera-service:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: "200m"
            memory: "512Mi"
          limits:
            cpu: "1"
            memory: "2Gi"
```

#### Canary Analysis Template

```yaml
# deployments/canary-analysis.yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
  namespace: its-camera-ai-prod
spec:
  args:
  - name: service-name
  metrics:
  - name: success-rate
    interval: 2m
    count: 5
    successCondition: result[0] >= 0.95
    failureLimit: 2
    provider:
      prometheus:
        address: http://prometheus.monitoring.svc.cluster.local:9090
        query: |
          sum(rate(http_requests_total{service="{{args.service-name}}",status=~"2.."}[2m])) /
          sum(rate(http_requests_total{service="{{args.service-name}}"}[2m]))
  - name: avg-response-time
    interval: 2m
    count: 5
    successCondition: result[0] <= 0.1
    failureLimit: 2
    provider:
      prometheus:
        address: http://prometheus.monitoring.svc.cluster.local:9090
        query: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket{service="{{args.service-name}}"}[2m])) by (le)
          )
```

### 6.3 Artifact Management

```yaml
# .github/workflows/artifact-management.yaml
name: Artifact Management

on:
  release:
    types: [published]

jobs:
  create-helm-release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure Git
      run: |
        git config user.name "$GITHUB_ACTOR"
        git config user.email "$GITHUB_ACTOR@users.noreply.github.com"
    
    - name: Install Helm
      uses: azure/setup-helm@v4
      with:
        version: '3.15.4'
    
    - name: Run chart-releaser
      uses: helm/chart-releaser-action@v1.6.0
      with:
        charts_dir: charts
      env:
        CR_TOKEN: "${{ secrets.GITHUB_TOKEN }}"
    
    - name: Update Helm repository index
      run: |
        helm repo index . --url https://its-team.github.io/its-camera-ai
        git add index.yaml
        git commit -m "Update Helm repository index"
        git push
```

---

## 7. Edge Deployment Strategy

### 7.1 K3s Edge Node Configuration

```bash
#!/bin/bash
# scripts/setup-edge-node.sh

set -euo pipefail

# Edge node configuration
EDGE_NODE_ID=${1:-"edge-node-001"}
CLOUD_ENDPOINT=${2:-"https://its-camera-ai.example.com"}
GPU_ENABLED=${3:-"false"}

echo "Setting up edge node: $EDGE_NODE_ID"

# Install K3s with minimal footprint
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server \
  --disable traefik \
  --disable servicelb \
  --disable local-storage \
  --node-name $EDGE_NODE_ID \
  --data-dir /opt/k3s-data \
  --kubelet-arg containerd=/run/k3s/containerd/containerd.sock" sh -

# Install NVIDIA GPU support if enabled
if [[ "$GPU_ENABLED" == "true" ]]; then
    echo "Installing NVIDIA GPU support..."
    
    # Install NVIDIA Container Runtime
    curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
      sudo apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-runtime
    
    # Configure K3s to use NVIDIA runtime
    sudo mkdir -p /var/lib/rancher/k3s/agent/etc/containerd/
    cat << EOF | sudo tee /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl
[plugins.cri.containerd.runtimes.runc]
  runtime_type = "io.containerd.runc.v2"

[plugins.cri.containerd.runtimes.nvidia]
  runtime_type = "io.containerd.runc.v2"

[plugins.cri.containerd.runtimes.nvidia.options]
  BinaryName = "/usr/bin/nvidia-container-runtime"
EOF
    
    sudo systemctl restart k3s
fi

# Install edge-specific applications
kubectl apply -f - << EOF
apiVersion: v1
kind: Namespace
metadata:
  name: its-camera-ai-edge
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: edge-agent
  namespace: its-camera-ai-edge
spec:
  replicas: 1
  selector:
    matchLabels:
      app: edge-agent
  template:
    metadata:
      labels:
        app: edge-agent
    spec:
      containers:
      - name: edge-agent
        image: ghcr.io/its-team/its-camera-ai-edge-agent:latest
        env:
        - name: EDGE_NODE_ID
          value: "$EDGE_NODE_ID"
        - name: CLOUD_ENDPOINT
          value: "$CLOUD_ENDPOINT"
        - name: GPU_ENABLED
          value: "$GPU_ENABLED"
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "500m"
            memory: "512Mi"
        volumeMounts:
        - name: edge-config
          mountPath: /etc/edge-config
      volumes:
      - name: edge-config
        configMap:
          name: edge-config
EOF

echo "Edge node setup complete: $EDGE_NODE_ID"
```

### 7.2 Edge-Cloud Synchronization

```yaml
# edge/sync-controller.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sync-controller
  namespace: its-camera-ai-edge
spec:
  replicas: 1
  selector:
    matchLabels:
      app: sync-controller
  template:
    metadata:
      labels:
        app: sync-controller
    spec:
      containers:
      - name: sync-controller
        image: ghcr.io/its-team/its-camera-ai-sync-controller:latest
        env:
        - name: SYNC_INTERVAL
          value: "30s"
        - name: CLOUD_API_ENDPOINT
          value: "https://api.its-camera-ai.com"
        - name: EDGE_NODE_ID
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        resources:
          requests:
            cpu: "50m"
            memory: "64Mi"
          limits:
            cpu: "200m"
            memory: "256Mi"
        volumeMounts:
        - name: sync-storage
          mountPath: /var/lib/sync
      volumes:
      - name: sync-storage
        emptyDir: {}
```

### 7.3 Offline Capabilities Configuration

```yaml
# edge/offline-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: offline-config
  namespace: its-camera-ai-edge
data:
  offline-mode.yaml: |
    offline:
      enabled: true
      buffer_size: "10Gi"
      retention_period: "72h"
      sync_on_reconnect: true
      
    local_storage:
      type: "sqlite"
      path: "/var/lib/edge-storage/cache.db"
      max_size: "5Gi"
      
    ml_models:
      cache_enabled: true
      models:
      - name: "yolo11n"
        version: "v1.0.0"
        local_path: "/var/lib/models/yolo11n.pt"
        required: true
      - name: "tracking-model"
        version: "v1.2.0"
        local_path: "/var/lib/models/tracking.pt"
        required: false
    
    health_checks:
      cloud_connectivity:
        endpoint: "https://api.its-camera-ai.com/health"
        timeout: "5s"
        interval: "30s"
      local_services:
        camera_processor:
          endpoint: "http://localhost:8080/health"
          timeout: "2s"
          interval: "10s"
```

### 7.4 OTA Update Mechanism

```yaml
# edge/ota-updater.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ota-updater-config
  namespace: its-camera-ai-edge
data:
  update-policy.yaml: |
    update_policy:
      auto_update: true
      maintenance_window:
        start: "02:00"
        end: "04:00"
        timezone: "UTC"
      rollback:
        enabled: true
        timeout: "5m"
        health_check_retries: 3
      
    update_channels:
      stable:
        check_interval: "24h"
        max_updates_per_day: 1
      beta:
        check_interval: "12h"
        max_updates_per_day: 2
    
    health_checks:
      post_update:
      - name: "camera-service"
        endpoint: "http://localhost:8080/health"
        timeout: "10s"
        retries: 5
      - name: "inference-engine"
        endpoint: "http://localhost:8081/health"
        timeout: "30s"
        retries: 3
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: ota-updater
  namespace: its-camera-ai-edge
spec:
  selector:
    matchLabels:
      app: ota-updater
  template:
    metadata:
      labels:
        app: ota-updater
    spec:
      hostNetwork: true
      hostPID: true
      containers:
      - name: ota-updater
        image: ghcr.io/its-team/its-camera-ai-ota-updater:latest
        securityContext:
          privileged: true
        env:
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        volumeMounts:
        - name: config
          mountPath: /etc/ota-updater
        - name: host-root
          mountPath: /host
          readOnly: true
        resources:
          requests:
            cpu: "50m"
            memory: "64Mi"
          limits:
            cpu: "200m"
            memory: "256Mi"
      volumes:
      - name: config
        configMap:
          name: ota-updater-config
      - name: host-root
        hostPath:
          path: /
```

---

## 8. Scaling and Performance

### 8.1 Horizontal Pod Autoscaler Configuration

```yaml
# scaling/hpa-camera-service.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: camera-service-hpa
  namespace: its-camera-ai-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: camera-service
  minReplicas: 3
  maxReplicas: 50
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
        averageValue: "10"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
      - type: Pods
        value: 5
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Min
```

### 8.2 Vertical Pod Autoscaler Setup

```yaml
# scaling/vpa-vision-engine.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: vision-engine-vpa
  namespace: its-camera-ai-prod
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vision-engine
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: vision-engine
      maxAllowed:
        cpu: "8"
        memory: "32Gi"
        nvidia.com/gpu: "2"
      minAllowed:
        cpu: "1"
        memory: "4Gi"
        nvidia.com/gpu: "1"
      controlledResources: ["cpu", "memory", "nvidia.com/gpu"]
```

### 8.3 Cluster Autoscaler Configuration

```yaml
# scaling/cluster-autoscaler.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      containers:
      - image: registry.k8s.io/autoscaling/cluster-autoscaler:v1.31.0
        name: cluster-autoscaler
        resources:
          limits:
            cpu: 100m
            memory: 300Mi
          requests:
            cpu: 100m
            memory: 300Mi
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/its-camera-ai-prod
        - --balance-similar-node-groups
        - --scale-down-enabled=true
        - --scale-down-delay-after-add=10m
        - --scale-down-unneeded-time=10m
        - --scale-down-utilization-threshold=0.5
        - --max-node-provision-time=15m
        env:
        - name: AWS_REGION
          value: us-west-2
```

### 8.4 GPU Node Pool Management

```yaml
# scaling/gpu-nodepool.yaml
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: gpu-nodepool
spec:
  template:
    metadata:
      labels:
        node-type: gpu
    spec:
      requirements:
      - key: kubernetes.io/arch
        operator: In
        values: ["amd64"]
      - key: karpenter.sh/capacity-type
        operator: In
        values: ["spot", "on-demand"]
      - key: node.kubernetes.io/instance-type
        operator: In
        values: ["p3.2xlarge", "p3.8xlarge", "g4dn.xlarge", "g4dn.2xlarge"]
      - key: nvidia.com/gpu
        operator: Exists
      nodeClassRef:
        apiVersion: karpenter.k8s.aws/v1beta1
        kind: EC2NodeClass
        name: gpu-nodeclass
      taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
  limits:
    cpu: 1000
    memory: 1000Gi
  disruption:
    consolidationPolicy: WhenUnderutilized
    consolidateAfter: 30s
    expireAfter: 30m
---
apiVersion: karpenter.k8s.aws/v1beta1
kind: EC2NodeClass
metadata:
  name: gpu-nodeclass
spec:
  amiFamily: AL2
  subnetSelectorTerms:
  - tags:
        karpenter.sh/discovery: "its-camera-ai-prod"
  securityGroupSelectorTerms:
  - tags:
        karpenter.sh/discovery: "its-camera-ai-prod"
  instanceStorePolicy: RAID0
  userData: |
    #!/bin/bash
    /etc/eks/bootstrap.sh its-camera-ai-prod
    yum install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=containerd
    systemctl restart containerd
    /opt/aws/bin/cfn-signal --exit-code $? --stack ${AWS::StackName} --resource NodeGroup --region ${AWS::Region}
```

### 8.5 Performance Tuning Guidelines

```yaml
# performance/performance-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: performance-tuning-config
  namespace: its-camera-ai-prod
data:
  performance-settings.yaml: |
    # GPU Configuration
    gpu:
      batch_size: 32
      memory_fraction: 0.9
      allow_growth: true
      mixed_precision: true
      
    # Inference Engine Settings
    inference:
      max_batch_wait_time: "5ms"
      max_batch_size: 64
      num_workers: 4
      prefetch_factor: 2
      
    # Networking Optimizations
    networking:
      tcp_keepalive: true
      tcp_nodelay: true
      connection_pool_size: 100
      timeout_connect: "5s"
      timeout_read: "30s"
      
    # Memory Management
    memory:
      jemalloc_enabled: true
      memory_pool_size: "2Gi"
      gc_threshold: 0.8
      
    # Threading Configuration
    threading:
      max_threads: 16
      thread_pool_size: 8
      async_workers: 4
      
    # Caching Strategy
    caching:
      redis_pool_size: 20
      cache_ttl: "5m"
      local_cache_size: "512Mi"
      
    # Database Connection Tuning
    database:
      pool_size: 20
      max_overflow: 30
      pool_timeout: 30
      pool_recycle: 3600
      pool_pre_ping: true
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: kernel-tuning
  namespace: its-camera-ai-prod
data:
  99-its-camera-ai.conf: |
    # Network performance tuning
    net.core.rmem_max = 134217728
    net.core.wmem_max = 134217728
    net.ipv4.tcp_rmem = 4096 87380 134217728
    net.ipv4.tcp_wmem = 4096 65536 134217728
    net.ipv4.tcp_window_scaling = 1
    net.ipv4.tcp_timestamps = 1
    net.ipv4.tcp_sack = 1
    net.core.netdev_max_backlog = 5000
    
    # Memory management
    vm.swappiness = 10
    vm.dirty_ratio = 15
    vm.dirty_background_ratio = 5
    
    # CPU scheduling
    kernel.sched_migration_cost_ns = 5000000
    kernel.sched_autogroup_enabled = 0
```

---

## 9. Security Implementation

### 9.1 Network Policies

```yaml
# security/network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: its-camera-ai-network-policy
  namespace: its-camera-ai-prod
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/part-of: its-camera-ai
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system
    - podSelector:
        matchLabels:
          app.kubernetes.io/part-of: its-camera-ai
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 9090
  egress:
  - to:
    - podSelector:
        matchLabels:
          app.kubernetes.io/part-of: its-camera-ai
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
    - protocol: TCP
      port: 9092  # Kafka
```

### 9.2 Pod Security Standards

```yaml
# security/pod-security-policy.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: its-camera-ai-prod
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: its-camera-ai-serviceaccount
  namespace: its-camera-ai-prod
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: its-camera-ai-role
  namespace: its-camera-ai-prod
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: its-camera-ai-rolebinding
  namespace: its-camera-ai-prod
subjects:
- kind: ServiceAccount
  name: its-camera-ai-serviceaccount
  namespace: its-camera-ai-prod
roleRef:
  kind: Role
  name: its-camera-ai-role
  apiGroup: rbac.authorization.k8s.io
```

### 9.3 Workload Identity Configuration

```yaml
# security/workload-identity.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: its-camera-ai-workload-identity
  namespace: its-camera-ai-prod
  annotations:
    iam.gke.io/gcp-service-account: its-camera-ai@project-id.iam.gserviceaccount.com
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/its-camera-ai-role
---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: its-camera-ai-authz
  namespace: its-camera-ai-prod
spec:
  selector:
    matchLabels:
      app: camera-service
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/its-camera-ai-prod/sa/its-camera-ai-workload-identity"]
    to:
    - operation:
        methods: ["GET", "POST"]
        paths: ["/api/v1/*"]
  - from:
    - source:
        namespaces: ["istio-system"]
    to:
    - operation:
        methods: ["GET"]
        paths: ["/health", "/metrics"]
```

### 9.4 Encryption Configuration

```yaml
# security/encryption-config.yaml
apiVersion: v1
kind: Secret
metadata:
  name: encryption-keys
  namespace: its-camera-ai-prod
type: Opaque
data:
  data-encryption-key: <base64-encoded-key>
  communication-encryption-key: <base64-encoded-key>
---
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: its-camera-ai-prod
spec:
  mtls:
    mode: STRICT
---
apiVersion: security.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: its-camera-ai-tls
  namespace: its-camera-ai-prod
spec:
  host: "*.its-camera-ai-prod.svc.cluster.local"
  trafficPolicy:
    tls:
      mode: ISTIO_MUTUAL
```

### 9.5 Compliance Scanning

```yaml
# security/compliance-scan.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: compliance-scanner
  namespace: its-camera-ai-prod
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: compliance-scanner
            image: aquasec/trivy:latest
            command:
            - /bin/sh
            - -c
            - |
              # Scan container images
              trivy image --format json --output /tmp/image-scan.json \
                ghcr.io/its-team/its-camera-ai-camera-service:latest
              
              # Scan Kubernetes configurations
              trivy k8s --format json --output /tmp/k8s-scan.json \
                cluster --namespace its-camera-ai-prod
              
              # Upload results to security dashboard
              curl -X POST \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $SECURITY_TOKEN" \
                -d @/tmp/image-scan.json \
                https://security-dashboard.its-camera-ai.com/api/v1/scans
                
              curl -X POST \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $SECURITY_TOKEN" \
                -d @/tmp/k8s-scan.json \
                https://security-dashboard.its-camera-ai.com/api/v1/k8s-scans
            env:
            - name: SECURITY_TOKEN
              valueFrom:
                secretKeyRef:
                  name: security-dashboard-token
                  key: token
            volumeMounts:
            - name: tmp
              mountPath: /tmp
          volumes:
          - name: tmp
            emptyDir: {}
          restartPolicy: OnFailure
```

---

## 10. Disaster Recovery

### 10.1 Backup Strategies

#### Database Backup Configuration

```yaml
# disaster-recovery/postgresql-backup.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgresql-backup
  namespace: its-camera-ai-prod
spec:
  schedule: "0 3 * * *"  # Daily at 3 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: postgresql-backup
            image: postgres:16
            command:
            - /bin/bash
            - -c
            - |
              export PGPASSWORD="$POSTGRES_PASSWORD"
              
              # Create backup
              BACKUP_FILE="postgresql-backup-$(date +%Y%m%d-%H%M%S).sql.gz"
              pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB | gzip > /tmp/$BACKUP_FILE
              
              # Upload to S3
              aws s3 cp /tmp/$BACKUP_FILE s3://its-camera-ai-backups/postgresql/
              
              # Cleanup old backups (keep 30 days)
              aws s3 ls s3://its-camera-ai-backups/postgresql/ | \
                while read -r line; do
                  createDate=$(echo $line | awk '{print $1" "$2}')
                  createDate=$(date -d "$createDate" +%s)
                  olderThan=$(date -d '30 days ago' +%s)
                  if [[ $createDate -lt $olderThan ]]; then
                    fileName=$(echo $line | awk '{print $4}')
                    aws s3 rm s3://its-camera-ai-backups/postgresql/$fileName
                  fi
                done
            env:
            - name: POSTGRES_HOST
              value: "postgresql-cluster-rw"
            - name: POSTGRES_USER
              valueFrom:
                secretKeyRef:
                  name: postgresql-credentials
                  key: username
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgresql-credentials
                  key: password
            - name: POSTGRES_DB
              value: "its_camera_ai"
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: backup-credentials
                  key: access-key-id
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: backup-credentials
                  key: secret-access-key
            - name: AWS_DEFAULT_REGION
              value: "us-west-2"
          restartPolicy: OnFailure
```

#### TimescaleDB Backup

```yaml
# disaster-recovery/timescaledb-backup.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: timescaledb-backup
  namespace: its-camera-ai-prod
spec:
  schedule: "30 3 * * *"  # Daily at 3:30 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: timescaledb-backup
            image: timescale/timescaledb:latest-pg16
            command:
            - /bin/bash
            - -c
            - |
              export PGPASSWORD="$POSTGRES_PASSWORD"
              
              # Create compressed backup with TimescaleDB extensions
              BACKUP_FILE="timescaledb-backup-$(date +%Y%m%d-%H%M%S).sql.gz"
              pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB \
                --format=custom --compress=9 \
                --exclude-table-data='*_log_*' \
                --exclude-table-data='*_temp_*' | gzip > /tmp/$BACKUP_FILE
              
              # Upload to S3
              aws s3 cp /tmp/$BACKUP_FILE s3://its-camera-ai-backups/timescaledb/
              
              # Create continuous aggregate backups
              psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -c \
                "SELECT create_hypertable_backup('s3://its-camera-ai-backups/timescaledb/continuous-aggregates/');"
            env:
            - name: POSTGRES_HOST
              value: "timescaledb"
            - name: POSTGRES_USER
              valueFrom:
                secretKeyRef:
                  name: timescaledb-credentials
                  key: username
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: timescaledb-credentials
                  key: password
            - name: POSTGRES_DB
              value: "metrics"
          restartPolicy: OnFailure
```

### 10.2 Multi-Region Deployment

```yaml
# disaster-recovery/multi-region-setup.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: its-camera-ai-dr-region
  namespace: argocd
spec:
  project: its-camera-ai
  source:
    repoURL: https://github.com/its-team/its-camera-ai-infra
    targetRevision: HEAD
    path: k8s/applications/overlays/disaster-recovery
  destination:
    server: https://eks-dr-cluster.us-east-1.amazonaws.com
    namespace: its-camera-ai-dr
  syncPolicy:
    automated:
      prune: false  # Manual sync for DR environment
      selfHeal: false
    syncOptions:
    - CreateNamespace=true
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: dr-configuration
  namespace: its-camera-ai-dr
data:
  dr-settings.yaml: |
    disaster_recovery:
      mode: "standby"  # active, standby, or failover
      primary_region: "us-west-2"
      dr_region: "us-east-1"
      
    replication:
      postgresql:
        streaming_replication: true
        sync_mode: "async"
        replication_lag_threshold: "10s"
      
      redis:
        master_slave_replication: true
        sentinel_enabled: true
        sentinel_quorum: 2
      
      timescaledb:
        continuous_aggregate_sync: true
        hypertable_replication: true
        sync_interval: "5m"
    
    failover:
      automatic: false
      health_check_interval: "30s"
      failover_timeout: "5m"
      rollback_enabled: true
      
    monitoring:
      replication_lag_alert_threshold: "30s"
      cross_region_latency_threshold: "100ms"
      data_consistency_check_interval: "1h"
```

### 10.3 Failover Procedures

```bash
#!/bin/bash
# scripts/disaster-recovery-failover.sh

set -euo pipefail

DR_REGION=${1:-"us-east-1"}
PRIMARY_REGION=${2:-"us-west-2"}
FAILOVER_TYPE=${3:-"manual"}  # manual or automatic

echo "Initiating disaster recovery failover..."
echo "DR Region: $DR_REGION"
echo "Primary Region: $PRIMARY_REGION"
echo "Failover Type: $FAILOVER_TYPE"

# Validate prerequisites
validate_failover_prerequisites() {
    echo "Validating failover prerequisites..."
    
    # Check DR cluster connectivity
    kubectl --context="eks-$DR_REGION" cluster-info || {
        echo "Error: Cannot connect to DR cluster in $DR_REGION"
        exit 1
    }
    
    # Check replication status
    REPLICATION_LAG=$(kubectl --context="eks-$DR_REGION" exec -n its-camera-ai-dr \
        postgresql-replica-0 -- psql -t -c "SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()));" | tr -d ' ')
    
    if [[ $(echo "$REPLICATION_LAG > 60" | bc -l) -eq 1 ]]; then
        echo "Warning: Replication lag is ${REPLICATION_LAG}s (>60s threshold)"
        if [[ "$FAILOVER_TYPE" == "automatic" ]]; then
            echo "Error: Automatic failover aborted due to high replication lag"
            exit 1
        fi
    fi
    
    echo "Prerequisites validated successfully"
}

# Promote DR databases to primary
promote_dr_databases() {
    echo "Promoting DR databases to primary..."
    
    # Promote PostgreSQL replica
    kubectl --context="eks-$DR_REGION" exec -n its-camera-ai-dr \
        postgresql-replica-0 -- su postgres -c "pg_promote"
    
    # Promote Redis replica
    kubectl --context="eks-$DR_REGION" exec -n its-camera-ai-dr \
        redis-replica-0 -- redis-cli REPLICAOF NO ONE
    
    # Update TimescaleDB configuration
    kubectl --context="eks-$DR_REGION" patch configmap -n its-camera-ai-dr \
        timescaledb-config --patch '{"data":{"postgresql.conf":"wal_level = replica\nmax_wal_senders = 10\nmax_replication_slots = 10"}}'
    
    echo "Database promotion completed"
}

# Update DNS to point to DR region
update_dns_records() {
    echo "Updating DNS records to point to DR region..."
    
    # Update Route53 records
    aws route53 change-resource-record-sets \
        --hosted-zone-id Z1234567890ABC \
        --change-batch '{
            "Changes": [{
                "Action": "UPSERT",
                "ResourceRecordSet": {
                    "Name": "its-camera-ai.example.com",
                    "Type": "CNAME",
                    "TTL": 300,
                    "ResourceRecords": [{"Value": "its-camera-ai-dr-'$DR_REGION'.example.com"}]
                }
            }]
        }'
    
    echo "DNS records updated"
}

# Scale up DR services
scale_dr_services() {
    echo "Scaling up DR services..."
    
    # Update ArgoCD to sync DR applications
    kubectl --context="eks-$DR_REGION" patch application -n argocd \
        its-camera-ai-dr --patch '{"spec":{"syncPolicy":{"automated":{"prune":true,"selfHeal":true}}}}'
    
    # Scale up deployments
    kubectl --context="eks-$DR_REGION" scale deployment -n its-camera-ai-dr \
        --replicas=3 camera-service analytics-service streaming-service
    
    # Scale up vision-engine with GPU nodes
    kubectl --context="eks-$DR_REGION" scale deployment -n its-camera-ai-dr \
        --replicas=2 vision-engine
    
    echo "DR services scaled up"
}

# Verify failover success
verify_failover() {
    echo "Verifying failover success..."
    
    # Wait for services to be ready
    kubectl --context="eks-$DR_REGION" wait --for=condition=available \
        deployment -n its-camera-ai-dr --all --timeout=300s
    
    # Run health checks
    for service in camera-service analytics-service streaming-service vision-engine; do
        HEALTH_URL="https://its-camera-ai.example.com/api/v1/$service/health"
        HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL")
        if [[ "$HTTP_STATUS" != "200" ]]; then
            echo "Error: $service health check failed (HTTP $HTTP_STATUS)"
            exit 1
        fi
        echo "$service health check passed"
    done
    
    # Verify data consistency
    python3 scripts/verify-data-consistency.py --region="$DR_REGION"
    
    echo "Failover verification completed successfully"
}

# Send notifications
send_notifications() {
    echo "Sending failover notifications..."
    
    # Send Slack notification
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"🚨 Disaster Recovery Failover Completed\n• DR Region: $DR_REGION\n• Primary Region: $PRIMARY_REGION\n• Failover Type: $FAILOVER_TYPE\n• Status: SUCCESS\"}" \
        "$SLACK_WEBHOOK_URL"
    
    # Send email notification
    aws ses send-email \
        --source "alerts@its-camera-ai.com" \
        --destination "ToAddresses=ops-team@its-camera-ai.com" \
        --message "Subject={Data='DR Failover Completed - $DR_REGION'},Body={Text={Data='Disaster recovery failover has been completed successfully. System is now running in $DR_REGION.'}}"
    
    echo "Notifications sent"
}

# Main execution
main() {
    validate_failover_prerequisites
    promote_dr_databases
    update_dns_records
    scale_dr_services
    verify_failover
    send_notifications
    
    echo "✅ Disaster recovery failover completed successfully"
    echo "🔄 System is now running in DR region: $DR_REGION"
    echo "📋 Next steps:"
    echo "   1. Monitor system performance and metrics"
    echo "   2. Investigate root cause of primary region failure"
    echo "   3. Plan failback procedures when primary region is restored"
}

# Execute main function
main "$@"
```

### 10.4 RTO/RPO Targets

```yaml
# disaster-recovery/rto-rpo-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rto-rpo-targets
  namespace: its-camera-ai-prod
data:
  targets.yaml: |
    service_level_objectives:
      disaster_recovery:
        rto_target: "15m"  # Recovery Time Objective
        rpo_target: "5m"   # Recovery Point Objective
        
    service_specific_targets:
      camera_service:
        rto: "10m"
        rpo: "2m"
        criticality: "high"
        
      analytics_service:
        rto: "15m"
        rpo: "5m"
        criticality: "high"
        
      streaming_service:
        rto: "5m"
        rpo: "1m"
        criticality: "critical"
        
      vision_engine:
        rto: "20m"
        rpo: "10m"
        criticality: "medium"
    
    backup_schedules:
      database:
        full_backup: "0 2 * * 0"  # Weekly full backup
        incremental_backup: "0 */4 * * *"  # Every 4 hours
        transaction_log_backup: "*/15 * * * *"  # Every 15 minutes
        
      application_data:
        backup_interval: "1h"
        retention_period: "30d"
        encryption: "AES-256"
        
      ml_models:
        backup_interval: "24h"
        retention_period: "90d"
        versioning: "enabled"
    
    monitoring:
      replication_lag_threshold: "30s"
      backup_failure_alert: "immediate"
      dr_health_check_interval: "5m"
      cross_region_latency_threshold: "100ms"
```

---

## 11. Operations Runbook

### 11.1 Deployment Procedures

#### Standard Deployment Process

```bash
#!/bin/bash
# scripts/deploy-production.sh

set -euo pipefail

VERSION=${1:-"latest"}
ENVIRONMENT=${2:-"production"}
DRY_RUN=${3:-"false"}

echo "🚀 Starting deployment to $ENVIRONMENT"
echo "Version: $VERSION"
echo "Dry run: $DRY_RUN"

# Pre-deployment checks
pre_deployment_checks() {
    echo "Running pre-deployment checks..."
    
    # Check cluster connectivity
    kubectl cluster-info || exit 1
    
    # Verify sufficient resources
    kubectl describe nodes | grep -A 5 "Allocated resources" || {
        echo "Warning: Could not verify node resources"
    }
    
    # Check ArgoCD sync status
    argocd app list --output json | jq -r '.[] | select(.metadata.name | contains("its-camera-ai")) | .status.sync.status' | grep -q "Synced" || {
        echo "Warning: Some applications are not in sync"
    }
    
    # Validate Helm charts
    helm template ./charts/its-camera-ai --values ./charts/its-camera-ai/values-$ENVIRONMENT.yaml > /dev/null || {
        echo "Error: Helm template validation failed"
        exit 1
    }
    
    echo "✅ Pre-deployment checks passed"
}

# Deploy application
deploy_application() {
    echo "Deploying application..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY RUN: Would deploy version $VERSION to $ENVIRONMENT"
        helm diff upgrade its-camera-ai ./charts/its-camera-ai \
            --namespace its-camera-ai-$ENVIRONMENT \
            --values ./charts/its-camera-ai/values-$ENVIRONMENT.yaml \
            --set image.tag=$VERSION
        return
    fi
    
    # Deploy using Helm
    helm upgrade --install its-camera-ai ./charts/its-camera-ai \
        --namespace its-camera-ai-$ENVIRONMENT \
        --create-namespace \
        --values ./charts/its-camera-ai/values-$ENVIRONMENT.yaml \
        --set image.tag=$VERSION \
        --wait \
        --timeout=10m
    
    echo "✅ Application deployed successfully"
}

# Post-deployment verification
post_deployment_verification() {
    echo "Running post-deployment verification..."
    
    # Wait for rollout to complete
    kubectl rollout status deployment --all \
        --namespace=its-camera-ai-$ENVIRONMENT \
        --timeout=600s
    
    # Run health checks
    for service in camera-service analytics-service streaming-service vision-engine; do
        kubectl wait --for=condition=ready pod \
            --selector=app=$service \
            --namespace=its-camera-ai-$ENVIRONMENT \
            --timeout=300s
        echo "✅ $service is ready"
    done
    
    # Run smoke tests
    pytest tests/smoke/ \
        --base-url=https://its-camera-ai-$ENVIRONMENT.example.com \
        --timeout=30
    
    echo "✅ Post-deployment verification completed"
}

# Rollback procedure
rollback_if_needed() {
    if [[ $? -ne 0 ]]; then
        echo "❌ Deployment failed, initiating rollback..."
        
        helm rollback its-camera-ai 0 \
            --namespace its-camera-ai-$ENVIRONMENT \
            --wait \
            --timeout=5m
        
        echo "✅ Rollback completed"
        exit 1
    fi
}

# Main execution
main() {
    pre_deployment_checks
    deploy_application
    post_deployment_verification || rollback_if_needed
    
    echo "🎉 Deployment to $ENVIRONMENT completed successfully!"
    echo "📊 Deployment metrics:"
    kubectl get pods -n its-camera-ai-$ENVIRONMENT -o wide
    kubectl top pods -n its-camera-ai-$ENVIRONMENT
}

# Execute main function with error handling
trap rollback_if_needed ERR
main "$@"
```

### 11.2 Rollback Strategies

```bash
#!/bin/bash
# scripts/rollback-deployment.sh

set -euo pipefail

ENVIRONMENT=${1:-"production"}
TARGET_REVISION=${2:-""}
REASON=${3:-"Manual rollback"}

echo "🔄 Starting rollback for environment: $ENVIRONMENT"
echo "Target revision: $TARGET_REVISION"
echo "Reason: $REASON"

# Quick rollback (automated)
quick_rollback() {
    echo "Performing quick rollback..."
    
    if [[ -z "$TARGET_REVISION" ]]; then
        # Rollback to previous revision
        helm rollback its-camera-ai 0 \
            --namespace its-camera-ai-$ENVIRONMENT \
            --wait \
            --timeout=5m
    else
        # Rollback to specific revision
        helm rollback its-camera-ai $TARGET_REVISION \
            --namespace its-camera-ai-$ENVIRONMENT \
            --wait \
            --timeout=5m
    fi
    
    echo "✅ Quick rollback completed"
}

# Database rollback (if needed)
database_rollback() {
    echo "Checking if database rollback is needed..."
    
    # Check for schema changes
    CURRENT_MIGRATION=$(kubectl exec -n its-camera-ai-$ENVIRONMENT \
        deployment/camera-service -- alembic current)
    
    TARGET_MIGRATION=$(kubectl exec -n its-camera-ai-$ENVIRONMENT \
        deployment/camera-service -- alembic history | grep $TARGET_REVISION | awk '{print $1}')
    
    if [[ "$CURRENT_MIGRATION" != "$TARGET_MIGRATION" ]]; then
        echo "Database rollback required..."
        kubectl exec -n its-camera-ai-$ENVIRONMENT \
            deployment/camera-service -- alembic downgrade $TARGET_MIGRATION
        echo "✅ Database rollback completed"
    else
        echo "No database rollback needed"
    fi
}

# Traffic management rollback
traffic_rollback() {
    echo "Managing traffic during rollback..."
    
    # Gradually shift traffic back
    for percentage in 25 50 75 100; do
        kubectl patch virtualservice its-camera-ai-vs \
            --namespace its-camera-ai-$ENVIRONMENT \
            --type merge \
            --patch "{\"spec\":{\"http\":[{\"match\":[{\"uri\":{\"prefix\":\"/\"}}],\"route\":[{\"destination\":{\"host\":\"camera-service\"},\"weight\":$percentage}]}]}}"
        
        echo "Traffic shifted: $percentage% to rolled back version"
        
        # Monitor metrics for 2 minutes
        sleep 120
        
        # Check error rates
        ERROR_RATE=$(curl -s "http://prometheus.monitoring.svc.cluster.local:9090/api/v1/query?query=rate(http_requests_total{status=~\"5..\"}[5m])/rate(http_requests_total[5m])" | jq -r '.data.result[0].value[1] // "0"')
        
        if (( $(echo "$ERROR_RATE > 0.05" | bc -l) )); then
            echo "❌ High error rate detected ($ERROR_RATE), aborting rollback"
            exit 1
        fi
    done
    
    echo "✅ Traffic rollback completed successfully"
}

# Verification and monitoring
verify_rollback() {
    echo "Verifying rollback success..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod \
        --selector=app.kubernetes.io/name=its-camera-ai \
        --namespace=its-camera-ai-$ENVIRONMENT \
        --timeout=300s
    
    # Run health checks
    pytest tests/smoke/ \
        --base-url=https://its-camera-ai-$ENVIRONMENT.example.com \
        --timeout=30
    
    # Monitor metrics for 10 minutes
    echo "Monitoring system metrics for 10 minutes..."
    for i in {1..10}; do
        CURRENT_TIME=$(date)
        ERROR_RATE=$(curl -s "http://prometheus.monitoring.svc.cluster.local:9090/api/v1/query?query=rate(http_requests_total{status=~\"5..\"}[1m])/rate(http_requests_total[1m])" | jq -r '.data.result[0].value[1] // "0"')
        RESPONSE_TIME=$(curl -s "http://prometheus.monitoring.svc.cluster.local:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[1m]))" | jq -r '.data.result[0].value[1] // "0"')
        
        echo "[$CURRENT_TIME] Error rate: $ERROR_RATE, P95 latency: ${RESPONSE_TIME}s"
        sleep 60
    done
    
    echo "✅ Rollback verification completed"
}

# Notification
send_rollback_notification() {
    echo "Sending rollback notifications..."
    
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"🔄 Rollback Completed\n• Environment: $ENVIRONMENT\n• Target Revision: $TARGET_REVISION\n• Reason: $REASON\n• Status: SUCCESS\"}" \
        "$SLACK_WEBHOOK_URL"
    
    echo "✅ Notifications sent"
}

# Main execution
main() {
    quick_rollback
    database_rollback
    traffic_rollback
    verify_rollback
    send_rollback_notification
    
    echo "🎉 Rollback completed successfully!"
}

main "$@"
```

### 11.3 Incident Response Playbooks

#### High Latency Incident

```yaml
# runbooks/high-latency-incident.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: high-latency-playbook
  namespace: its-camera-ai-prod
data:
  playbook.yaml: |
    incident_type: "High Latency"
    severity: "P1"
    
    detection:
      alerts:
      - "InferenceLatencyHigh"
      - "APIResponseTimeHigh"
      metrics:
      - "histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m])) > 0.1"
      - "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5"
    
    immediate_actions:
    - step: "Acknowledge alert and create incident ticket"
      owner: "On-call engineer"
      timeout: "2m"
      
    - step: "Check system dashboard for anomalies"
      command: "kubectl port-forward -n monitoring svc/grafana 3000:80"
      url: "http://localhost:3000/d/its-camera-ai-overview"
      
    - step: "Verify GPU utilization and memory usage"
      command: "kubectl top nodes --show-capacity | grep gpu"
      
    - step: "Check for resource constraints"
      commands:
      - "kubectl describe nodes | grep -A 10 'Allocated resources'"
      - "kubectl get hpa -n its-camera-ai-prod"
    
    investigation:
    - step: "Analyze inference queue length"
      query: "inference_queue_length"
      threshold: "> 50"
      
    - step: "Check database performance"
      commands:
      - "kubectl logs -n its-camera-ai-prod deployment/analytics-service | grep 'slow query'"
      - "kubectl exec -n its-camera-ai-prod postgresql-0 -- psql -c 'SELECT * FROM pg_stat_activity WHERE state = \"active\";'"
      
    - step: "Review recent deployments"
      command: "kubectl rollout history deployment -n its-camera-ai-prod"
    
    mitigation:
    - step: "Scale up inference workers"
      command: "kubectl scale deployment vision-engine --replicas=10 -n its-camera-ai-prod"
      
    - step: "Increase GPU node capacity"
      command: "kubectl annotate nodepool gpu-nodepool karpenter.sh/do-not-evict=true"
      
    - step: "Enable circuit breaker"
      command: "kubectl patch configmap api-config -p '{\"data\":{\"circuit_breaker\":\"enabled\"}}' -n its-camera-ai-prod"
    
    escalation:
    - condition: "Latency remains > 200ms after 15 minutes"
      action: "Page senior engineer"
      
    - condition: "Multiple service degradation"
      action: "Escalate to incident commander"
    
    resolution:
    - step: "Verify metrics return to normal"
      duration: "10m"
      
    - step: "Run smoke tests"
      command: "pytest tests/smoke/ --base-url=https://its-camera-ai.example.com"
      
    - step: "Create post-incident review ticket"
      template: "PIR-TEMPLATE-001"
```

#### Database Connection Issues

```yaml
# runbooks/database-connection-incident.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: database-connection-playbook
  namespace: its-camera-ai-prod
data:
  playbook.yaml: |
    incident_type: "Database Connection Issues"
    severity: "P0"
    
    detection:
      alerts:
      - "PostgreSQLDown"
      - "DatabaseConnectionPoolExhausted"
      - "DatabaseReplicationLag"
      metrics:
      - "postgresql_up == 0"
      - "postgresql_active_connections > postgresql_max_connections * 0.9"
    
    immediate_actions:
    - step: "Check database cluster status"
      commands:
      - "kubectl get postgresql-cluster -n its-camera-ai-prod"
      - "kubectl describe postgresql-cluster postgresql-cluster -n its-camera-ai-prod"
      
    - step: "Verify pod status"
      command: "kubectl get pods -l app=postgresql -n its-camera-ai-prod"
      
    - step: "Check logs for errors"
      command: "kubectl logs -n its-camera-ai-prod postgresql-cluster-1 --tail=100"
    
    investigation:
    - step: "Check connection pool status"
      command: "kubectl exec -n its-camera-ai-prod postgresql-cluster-1 -- psql -c 'SELECT * FROM pg_stat_activity;'"
      
    - step: "Verify disk space and I/O"
      commands:
      - "kubectl exec -n its-camera-ai-prod postgresql-cluster-1 -- df -h"
      - "kubectl exec -n its-camera-ai-prod postgresql-cluster-1 -- iostat -x 1 5"
      
    - step: "Check for long-running queries"
      command: "kubectl exec -n its-camera-ai-prod postgresql-cluster-1 -- psql -c 'SELECT pid, now() - pg_stat_activity.query_start AS duration, query FROM pg_stat_activity WHERE (now() - pg_stat_activity.query_start) > interval \"5 minutes\";'"
    
    mitigation:
    - step: "Kill long-running queries"
      command: "kubectl exec -n its-camera-ai-prod postgresql-cluster-1 -- psql -c 'SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE (now() - pg_stat_activity.query_start) > interval \"10 minutes\";'"
      
    - step: "Restart connection pooler"
      command: "kubectl rollout restart deployment pgbouncer -n its-camera-ai-prod"
      
    - step: "Failover to replica (if needed)"
      command: "kubectl exec -n its-camera-ai-prod postgresql-cluster-2 -- pg_promote"
    
    recovery:
    - step: "Monitor replication recovery"
      command: "kubectl exec -n its-camera-ai-prod postgresql-cluster-1 -- psql -c 'SELECT * FROM pg_stat_replication;'"
      
    - step: "Verify application connectivity"
      command: "kubectl logs -n its-camera-ai-prod deployment/camera-service | grep 'database connection'"
```

### 11.4 Maintenance Windows

```bash
#!/bin/bash
# scripts/maintenance-window.sh

set -euo pipefail

MAINTENANCE_TYPE=${1:-"standard"}  # standard, emergency, extended
START_TIME=${2:-"02:00"}
DURATION=${3:-"2h"}
ENVIRONMENT=${4:-"production"}

echo "🔧 Starting maintenance window"
echo "Type: $MAINTENANCE_TYPE"
echo "Start time: $START_TIME"
echo "Duration: $DURATION"
echo "Environment: $ENVIRONMENT"

# Pre-maintenance notifications
send_maintenance_start_notification() {
    echo "Sending maintenance start notifications..."
    
    # Update status page
    curl -X POST \
        -H "Authorization: Bearer $STATUS_PAGE_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"incident\": {
                \"name\": \"Scheduled Maintenance - $MAINTENANCE_TYPE\",
                \"status\": \"investigating\",
                \"message\": \"Scheduled maintenance is starting. Some services may be temporarily unavailable.\",
                \"component_ids\": [\"camera-service\", \"analytics-service\"]
            }
        }" \
        "https://api.statuspage.io/v1/pages/$STATUS_PAGE_ID/incidents"
    
    # Send team notifications
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"🔧 Maintenance Window Started\n• Type: $MAINTENANCE_TYPE\n• Duration: $DURATION\n• Environment: $ENVIRONMENT\"}" \
        "$SLACK_WEBHOOK_URL"
}

# Enable maintenance mode
enable_maintenance_mode() {
    echo "Enabling maintenance mode..."
    
    # Update ingress to show maintenance page
    kubectl patch ingress its-camera-ai-ingress \
        --namespace its-camera-ai-$ENVIRONMENT \
        --patch '{
            "metadata": {
                "annotations": {
                    "nginx.ingress.kubernetes.io/maintenance": "true",
                    "nginx.ingress.kubernetes.io/maintenance-page": "https://maintenance.its-camera-ai.com"
                }
            }
        }'
    
    # Reduce traffic by 90%
    kubectl patch virtualservice its-camera-ai-vs \
        --namespace its-camera-ai-$ENVIRONMENT \
        --type merge \
        --patch '{
            "spec": {
                "http": [{
                    "fault": {
                        "abort": {
                            "percentage": {"value": 90},
                            "httpStatus": 503
                        }
                    }
                }]
            }
        }'
    
    echo "✅ Maintenance mode enabled"
}

# Perform maintenance tasks
perform_maintenance_tasks() {
    echo "Performing maintenance tasks..."
    
    case $MAINTENANCE_TYPE in
        "standard")
            standard_maintenance
            ;;
        "security")
            security_maintenance
            ;;
        "database")
            database_maintenance
            ;;
        "emergency")
            emergency_maintenance
            ;;
    esac
    
    echo "✅ Maintenance tasks completed"
}

standard_maintenance() {
    # Update Kubernetes nodes
    kubectl drain node-1 --ignore-daemonsets --delete-emptydir-data
    # Perform node updates...
    kubectl uncordon node-1
    
    # Update application images
    kubectl set image deployment/camera-service \
        camera-service=ghcr.io/its-team/its-camera-ai-camera-service:v1.2.3 \
        --namespace its-camera-ai-$ENVIRONMENT
    
    # Clean up old resources
    kubectl delete pods --field-selector=status.phase==Succeeded \
        --namespace its-camera-ai-$ENVIRONMENT
}

security_maintenance() {
    # Update security patches
    kubectl patch daemonset security-agent \
        --patch '{"spec":{"template":{"spec":{"containers":[{"name":"security-agent","image":"security-agent:latest"}]}}}}'
    
    # Rotate secrets
    kubectl create secret generic database-credentials-new \
        --from-literal=username="$NEW_DB_USER" \
        --from-literal=password="$NEW_DB_PASSWORD" \
        --namespace its-camera-ai-$ENVIRONMENT
    
    kubectl patch deployment camera-service \
        --patch '{"spec":{"template":{"spec":{"containers":[{"name":"camera-service","env":[{"name":"DB_SECRET","valueFrom":{"secretKeyRef":{"name":"database-credentials-new","key":"password"}}}]}]}}}}'
    
    # Delete old secrets
    kubectl delete secret database-credentials \
        --namespace its-camera-ai-$ENVIRONMENT
}

database_maintenance() {
    # Backup database
    kubectl exec -n its-camera-ai-$ENVIRONMENT postgresql-0 -- \
        pg_dump -U postgres its_camera_ai > /tmp/pre-maintenance-backup.sql
    
    # Run database migrations
    kubectl exec -n its-camera-ai-$ENVIRONMENT deployment/camera-service -- \
        alembic upgrade head
    
    # Reindex database
    kubectl exec -n its-camera-ai-$ENVIRONMENT postgresql-0 -- \
        psql -U postgres -d its_camera_ai -c "REINDEX DATABASE its_camera_ai;"
    
    # Update statistics
    kubectl exec -n its-camera-ai-$ENVIRONMENT postgresql-0 -- \
        psql -U postgres -d its_camera_ai -c "ANALYZE;"
}

# Disable maintenance mode
disable_maintenance_mode() {
    echo "Disabling maintenance mode..."
    
    # Remove maintenance annotations
    kubectl patch ingress its-camera-ai-ingress \
        --namespace its-camera-ai-$ENVIRONMENT \
        --type=json \
        --patch='[
            {"op": "remove", "path": "/metadata/annotations/nginx.ingress.kubernetes.io~1maintenance"},
            {"op": "remove", "path": "/metadata/annotations/nginx.ingress.kubernetes.io~1maintenance-page"}
        ]'
    
    # Restore normal traffic
    kubectl patch virtualservice its-camera-ai-vs \
        --namespace its-camera-ai-$ENVIRONMENT \
        --type merge \
        --patch '{
            "spec": {
                "http": [{
                    "route": [{
                        "destination": {"host": "camera-service"},
                        "weight": 100
                    }]
                }]
            }
        }'
    
    echo "✅ Maintenance mode disabled"
}

# Post-maintenance verification
verify_system_health() {
    echo "Verifying system health..."
    
    # Wait for deployments to be ready
    kubectl wait --for=condition=available deployment --all \
        --namespace=its-camera-ai-$ENVIRONMENT \
        --timeout=300s
    
    # Run health checks
    pytest tests/smoke/ \
        --base-url=https://its-camera-ai-$ENVIRONMENT.example.com \
        --timeout=60
    
    # Monitor metrics for 10 minutes
    for i in {1..10}; do
        ERROR_RATE=$(curl -s "http://prometheus.monitoring.svc.cluster.local:9090/api/v1/query?query=rate(http_requests_total{status=~\"5..\"}[1m])/rate(http_requests_total[1m])" | jq -r '.data.result[0].value[1] // "0"')
        echo "Minute $i: Error rate $ERROR_RATE"
        sleep 60
    done
    
    echo "✅ System health verification completed"
}

# Send completion notification
send_maintenance_complete_notification() {
    echo "Sending maintenance completion notifications..."
    
    # Update status page
    curl -X POST \
        -H "Authorization: Bearer $STATUS_PAGE_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"incident\": {
                \"status\": \"resolved\",
                \"message\": \"Scheduled maintenance has been completed successfully. All services are operational.\"
            }
        }" \
        "https://api.statuspage.io/v1/pages/$STATUS_PAGE_ID/incidents/$INCIDENT_ID"
    
    # Send team notification
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"✅ Maintenance Window Completed\n• Type: $MAINTENANCE_TYPE\n• Duration: $DURATION\n• Status: SUCCESS\n• All systems operational\"}" \
        "$SLACK_WEBHOOK_URL"
}

# Main execution
main() {
    send_maintenance_start_notification
    enable_maintenance_mode
    perform_maintenance_tasks
    disable_maintenance_mode
    verify_system_health
    send_maintenance_complete_notification
    
    echo "🎉 Maintenance window completed successfully!"
}

main "$@"
```

### 11.5 Capacity Planning

```python
#!/usr/bin/env python3
# scripts/capacity-planning.py

import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class CapacityPlanner:
    def __init__(self, prometheus_url="http://prometheus.monitoring.svc.cluster.local:9090"):
        self.prometheus_url = prometheus_url
        self.metrics_cache = {}
    
    def query_prometheus(self, query, days_back=30):
        """Query Prometheus for historical data"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        params = {
            'query': query,
            'start': start_time.timestamp(),
            'end': end_time.timestamp(),
            'step': '1h'
        }
        
        response = requests.get(f"{self.prometheus_url}/api/v1/query_range", params=params)
        return response.json()
    
    def get_resource_utilization(self):
        """Get current resource utilization metrics"""
        metrics = {
            'cpu_utilization': 'avg(rate(container_cpu_usage_seconds_total[5m])) by (pod)',
            'memory_utilization': 'avg(container_memory_working_set_bytes) by (pod)',
            'gpu_utilization': 'avg(DCGM_FI_DEV_GPU_UTIL) by (gpu)',
            'inference_requests': 'sum(rate(inference_requests_total[1h]))',
            'storage_usage': 'avg(kubelet_volume_stats_used_bytes) by (persistentvolumeclaim)'
        }
        
        results = {}
        for metric_name, query in metrics.items():
            data = self.query_prometheus(query)
            results[metric_name] = self.process_metric_data(data)
        
        return results
    
    def process_metric_data(self, prometheus_data):
        """Process Prometheus response data"""
        if 'data' not in prometheus_data or 'result' not in prometheus_data['data']:
            return []
        
        processed_data = []
        for result in prometheus_data['data']['result']:
            if 'values' in result:
                for timestamp, value in result['values']:
                    processed_data.append({
                        'timestamp': datetime.fromtimestamp(float(timestamp)),
                        'value': float(value),
                        'labels': result.get('metric', {})
                    })
        
        return processed_data
    
    def predict_resource_needs(self, metric_data, days_ahead=30):
        """Predict future resource needs using linear regression"""
        if not metric_data:
            return None
        
        df = pd.DataFrame(metric_data)
        df['timestamp_numeric'] = df['timestamp'].astype(np.int64) // 10**9
        
        # Prepare data for regression
        X = df[['timestamp_numeric']].values
        y = df['value'].values
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future values
        future_timestamps = []
        current_time = datetime.now()
        for i in range(days_ahead):
            future_time = current_time + timedelta(days=i)
            future_timestamps.append(future_time.timestamp())
        
        future_X = np.array(future_timestamps).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        return {
            'predictions': predictions.tolist(),
            'timestamps': future_timestamps,
            'score': model.score(X, y)
        }
    
    def generate_scaling_recommendations(self):
        """Generate scaling recommendations based on predictions"""
        utilization_data = self.get_resource_utilization()
        recommendations = {}
        
        # CPU scaling recommendations
        cpu_prediction = self.predict_resource_needs(utilization_data['cpu_utilization'])
        if cpu_prediction and max(cpu_prediction['predictions']) > 0.8:
            recommendations['cpu'] = {
                'action': 'scale_up',
                'current_utilization': np.mean([d['value'] for d in utilization_data['cpu_utilization']]),
                'predicted_peak': max(cpu_prediction['predictions']),
                'recommended_replicas': int(np.ceil(max(cpu_prediction['predictions']) / 0.7)),
                'timeline': '7 days'
            }
        
        # Memory scaling recommendations
        memory_prediction = self.predict_resource_needs(utilization_data['memory_utilization'])
        if memory_prediction:
            avg_memory = np.mean([d['value'] for d in utilization_data['memory_utilization']])
            predicted_peak = max(memory_prediction['predictions'])
            
            if predicted_peak > avg_memory * 1.5:
                recommendations['memory'] = {
                    'action': 'increase_limits',
                    'current_average': avg_memory / (1024**3),  # GB
                    'predicted_peak': predicted_peak / (1024**3),  # GB
                    'recommended_limit': predicted_peak * 1.3 / (1024**3),  # GB with buffer
                    'timeline': '14 days'
                }
        
        # GPU scaling recommendations
        gpu_prediction = self.predict_resource_needs(utilization_data['gpu_utilization'])
        if gpu_prediction and max(gpu_prediction['predictions']) > 85:
            recommendations['gpu'] = {
                'action': 'add_gpu_nodes',
                'current_utilization': np.mean([d['value'] for d in utilization_data['gpu_utilization']]),
                'predicted_peak': max(gpu_prediction['predictions']),
                'recommended_nodes': 2,
                'timeline': '3 days'
            }
        
        # Storage scaling recommendations
        storage_prediction = self.predict_resource_needs(utilization_data['storage_usage'])
        if storage_prediction:
            current_storage = np.mean([d['value'] for d in utilization_data['storage_usage']])
            predicted_storage = max(storage_prediction['predictions'])
            
            if predicted_storage > current_storage * 0.8:
                recommendations['storage'] = {
                    'action': 'expand_volumes',
                    'current_usage': current_storage / (1024**3),  # GB
                    'predicted_usage': predicted_storage / (1024**3),  # GB
                    'recommended_size': predicted_storage * 2 / (1024**3),  # GB with buffer
                    'timeline': '21 days'
                }
        
        return recommendations
    
    def generate_cost_analysis(self, recommendations):
        """Generate cost analysis for scaling recommendations"""
        costs = {
            'cpu': {'cost_per_core_hour': 0.05, 'unit': 'core'},
            'memory': {'cost_per_gb_hour': 0.01, 'unit': 'GB'},
            'gpu': {'cost_per_gpu_hour': 2.50, 'unit': 'GPU'},
            'storage': {'cost_per_gb_month': 0.10, 'unit': 'GB'}
        }
        
        cost_analysis = {}
        total_monthly_cost = 0
        
        for resource, recommendation in recommendations.items():
            if resource in costs:
                cost_info = costs[resource]
                
                if resource == 'cpu':
                    monthly_cost = recommendation['recommended_replicas'] * cost_info['cost_per_core_hour'] * 24 * 30
                elif resource == 'memory':
                    monthly_cost = recommendation['recommended_limit'] * cost_info['cost_per_gb_hour'] * 24 * 30
                elif resource == 'gpu':
                    monthly_cost = recommendation['recommended_nodes'] * 4 * cost_info['cost_per_gpu_hour'] * 24 * 30  # 4 GPUs per node
                elif resource == 'storage':
                    monthly_cost = recommendation['recommended_size'] * cost_info['cost_per_gb_month']
                
                cost_analysis[resource] = {
                    'monthly_cost': monthly_cost,
                    'annual_cost': monthly_cost * 12,
                    'cost_per_unit': cost_info['cost_per_core_hour'] if resource != 'storage' else cost_info['cost_per_gb_month']
                }
                
                total_monthly_cost += monthly_cost
        
        cost_analysis['total'] = {
            'monthly_cost': total_monthly_cost,
            'annual_cost': total_monthly_cost * 12
        }
        
        return cost_analysis
    
    def generate_report(self):
        """Generate comprehensive capacity planning report"""
        print("🔍 ITS Camera AI - Capacity Planning Report")
        print("=" * 50)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Get current utilization
        utilization_data = self.get_resource_utilization()
        
        print("📊 Current Resource Utilization:")
        for metric_name, data in utilization_data.items():
            if data:
                avg_value = np.mean([d['value'] for d in data])
                if metric_name == 'memory_utilization':
                    print(f"  {metric_name}: {avg_value / (1024**3):.2f} GB")
                elif metric_name == 'storage_usage':
                    print(f"  {metric_name}: {avg_value / (1024**3):.2f} GB")
                else:
                    print(f"  {metric_name}: {avg_value:.2f}")
        print()
        
        # Generate recommendations
        recommendations = self.generate_scaling_recommendations()
        
        if recommendations:
            print("🚀 Scaling Recommendations:")
            for resource, rec in recommendations.items():
                print(f"  {resource.upper()}:")
                print(f"    Action: {rec['action']}")
                print(f"    Timeline: {rec['timeline']}")
                for key, value in rec.items():
                    if key not in ['action', 'timeline']:
                        if isinstance(value, float):
                            print(f"    {key}: {value:.2f}")
                        else:
                            print(f"    {key}: {value}")
                print()
        else:
            print("✅ No scaling recommendations needed at this time.")
            print()
        
        # Generate cost analysis
        if recommendations:
            cost_analysis = self.generate_cost_analysis(recommendations)
            
            print("💰 Cost Analysis:")
            for resource, cost_info in cost_analysis.items():
                if resource != 'total':
                    print(f"  {resource.upper()}:")
                    print(f"    Monthly: ${cost_info['monthly_cost']:.2f}")
                    print(f"    Annual: ${cost_info['annual_cost']:.2f}")
                    print()
            
            print(f"  TOTAL ADDITIONAL COSTS:")
            print(f"    Monthly: ${cost_analysis['total']['monthly_cost']:.2f}")
            print(f"    Annual: ${cost_analysis['total']['annual_cost']:.2f}")
            print()
        
        print("📋 Next Steps:")
        print("  1. Review recommendations with team")
        print("  2. Plan implementation timeline")
        print("  3. Update capacity planning schedule")
        print("  4. Monitor resource usage trends")

if __name__ == "__main__":
    planner = CapacityPlanner()
    planner.generate_report()
```

---

## 12. Implementation Timeline

### 12.1 Phase-based Rollout Plan

| Phase | Duration | Components | Success Criteria |
|-------|----------|------------|------------------|
| **Phase 1 - Foundation** | 3 weeks | Infrastructure setup, base platform services | All clusters operational, basic monitoring active |
| **Phase 2 - Data Layer** | 2 weeks | Database clusters, message queues, object storage | Data persistence layer stable, replication working |
| **Phase 3 - Core Services** | 4 weeks | Application deployment, service mesh, API gateway | Core functionality deployed, traffic routing active |
| **Phase 4 - Observability** | 2 weeks | Complete monitoring stack, alerting, dashboards | Full observability coverage, SLI/SLO tracking |
| **Phase 5 - Security** | 2 weeks | Security hardening, compliance scanning, audit logging | Security controls active, compliance validated |
| **Phase 6 - Edge Deployment** | 3 weeks | Edge nodes, synchronization, offline capabilities | Edge infrastructure operational, sync working |
| **Phase 7 - Scaling & Optimization** | 2 weeks | Autoscaling, performance tuning, capacity planning | Auto-scaling active, performance optimized |
| **Phase 8 - DR & Operations** | 2 weeks | Disaster recovery, runbooks, maintenance procedures | DR tested, operational procedures documented |

### 12.2 Success Metrics and Validation

#### Platform Engineering KPIs

| Metric | Target | Current | Timeline |
|--------|--------|---------|----------|
| **Self-Service Rate** | 95% | - | Week 12 |
| **Provisioning Time** | < 5 minutes | - | Week 8 |
| **Platform Uptime** | 99.9% | - | Week 16 |
| **API Response Time** | < 200ms | - | Week 10 |
| **Documentation Coverage** | 100% | - | Week 18 |
| **Developer Onboarding** | < 1 day | - | Week 14 |
| **Deployment Success Rate** | > 98% | - | Week 12 |
| **Mean Time to Recovery** | < 15 minutes | - | Week 20 |

#### Technical Performance Targets

| Component | Metric | Target | Validation Method |
|-----------|--------|--------|-------------------|
| **Inference Engine** | Latency (p95) | < 100ms | Load testing |
| **API Gateway** | Throughput | 10k RPS | Performance testing |
| **Database** | Query latency (p95) | < 50ms | Database benchmarks |
| **Storage** | IOPS | > 10k | Storage performance tests |
| **Network** | Latency between regions | < 100ms | Network monitoring |
| **GPU Utilization** | Efficiency | > 80% | GPU monitoring |
| **Cache Hit Rate** | Redis cache | > 95% | Cache analytics |
| **Message Queue** | Processing latency | < 10ms | Queue monitoring |

### 12.3 Risk Mitigation Timeline

| Week | Risk Area | Mitigation Activity | Owner |
|------|-----------|-------------------|--------|
| 1-2 | Infrastructure setup delays | Pre-provision base infrastructure, validate tooling | Platform Team |
| 3-4 | Database migration complexity | Implement comprehensive backup/restore procedures | Database Team |
| 5-6 | Service mesh complexity | Deploy Istio in staging first, validate traffic routing | Network Team |
| 7-8 | Performance bottlenecks | Conduct early load testing, identify optimization areas | Performance Team |
| 9-10 | Security compliance gaps | Execute security audit, implement missing controls | Security Team |
| 11-12 | Edge deployment challenges | Test edge synchronization in lab environment | Edge Team |
| 13-14 | Monitoring gaps | Validate all SLIs/SLOs, test alerting workflows | Monitoring Team |
| 15-16 | Disaster recovery validation | Execute full DR drill, validate RTO/RPO targets | Operations Team |

---

## Conclusion

This comprehensive deployment and infrastructure implementation plan provides a production-ready foundation for the ITS Camera AI system. The plan emphasizes:

- **Platform Engineering Excellence**: Self-service capabilities, developer experience optimization, and operational efficiency
- **Production Readiness**: High availability, disaster recovery, security hardening, and compliance
- **Scalability**: Auto-scaling, performance optimization, and capacity planning
- **Operational Excellence**: Comprehensive monitoring, incident response, and maintenance procedures

### Key Success Factors

1. **Incremental Implementation**: Phased rollout minimizes risk and allows for learning
2. **Automation First**: Everything is automated and codified for consistency
3. **Observability-Driven**: Comprehensive monitoring and alerting from day one
4. **Security by Design**: Zero-trust architecture and defense in depth
5. **Developer Experience**: Self-service portal and golden paths reduce friction

### Next Steps

1. **Week 1-2**: Begin infrastructure provisioning and team setup
2. **Week 3-4**: Deploy core platform services and establish GitOps workflows
3. **Week 5-8**: Implement database layer and core application services
4. **Week 9-12**: Add comprehensive monitoring and security controls
5. **Week 13-16**: Deploy edge infrastructure and test disaster recovery
6. **Week 17-20**: Performance optimization and operational hardening

The implementation maintains the project's critical requirements of sub-100ms inference latency, >90% test coverage, and zero-trust security while providing a scalable foundation for growth from edge deployments to cloud-scale operations.