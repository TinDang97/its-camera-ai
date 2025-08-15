# ITS Camera AI Analytics Service - Monitoring Setup Guide

## Overview

This guide provides complete setup instructions for production monitoring of the ITS Camera AI Analytics Service, including **Prometheus**, **Grafana**, **ELK Stack**, **Jaeger**, and **custom alerting** systems for achieving comprehensive observability.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Monitoring Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                   │
│  │  Analytics      │───▶│   Prometheus    │                   │
│  │  Service        │    │   (Metrics)     │                   │
│  │  (:9090/metrics)│    └─────────────────┘                   │
│  └─────────────────┘             │                            │
│           │                      ▼                            │
│           │              ┌─────────────────┐                  │
│           │              │    Grafana      │                  │
│           │              │  (Dashboards)   │                  │
│           │              └─────────────────┘                  │
│           │                      │                            │
│           ▼                      ▼                            │
│  ┌─────────────────┐    ┌─────────────────┐                  │
│  │  ELK Stack      │    │   AlertManager  │                  │
│  │  (Logs)         │    │   (Alerting)    │                  │
│  │                 │    └─────────────────┘                  │
│  └─────────────────┘             │                            │
│           │                      ▼                            │
│           ▼              ┌─────────────────┐                  │
│  ┌─────────────────┐    │   PagerDuty     │                  │
│  │     Jaeger      │    │   (On-call)     │                  │
│  │   (Tracing)     │    └─────────────────┘                  │
│  └─────────────────┘                                         │
│                                                               │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Infrastructure Requirements
- **Kubernetes cluster** (1.21+) with sufficient resources
- **Persistent storage** for time-series data (100GB+ recommended)
- **Network connectivity** between monitoring components
- **Resource allocation**: 8 CPU cores, 32GB RAM for monitoring stack

### Software Requirements
```bash
# Required tools
kubectl (1.21+)
helm (3.7+)
docker (20.10+)

# Verify cluster access
kubectl cluster-info
kubectl get nodes
```

## Prometheus Setup

### Installation with Helm
```bash
# Add Prometheus community Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Create monitoring namespace
kubectl create namespace monitoring

# Install Prometheus with custom values
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --values monitoring/prometheus-values.yaml
```

### Prometheus Configuration (`monitoring/prometheus-values.yaml`)
```yaml
# High-performance Prometheus configuration for ITS Analytics
prometheus:
  prometheusSpec:
    # Resource allocation for high-throughput metrics
    resources:
      requests:
        cpu: "2000m"
        memory: "8Gi"
      limits:
        cpu: "4000m"
        memory: "16Gi"
    
    # Storage configuration
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: fast-ssd
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 100Gi
    
    # Retention and performance settings
    retention: 30d
    retentionSize: 80GB
    walCompression: true
    
    # High-frequency scraping for real-time monitoring
    scrapeInterval: 10s
    evaluationInterval: 10s
    
    # Additional scrape configs for analytics service
    additionalScrapeConfigs:
    - job_name: 'its-analytics-service'
      kubernetes_sd_configs:
      - role: endpoints
        namespaces:
          names:
          - its-camera-ai
      relabel_configs:
      - source_labels: [__meta_kubernetes_service_name]
        action: keep
        regex: its-analytics-service
      - source_labels: [__meta_kubernetes_endpoint_port_name]
        action: keep
        regex: metrics
      scrape_interval: 5s  # High-frequency for latency monitoring
      metrics_path: /metrics
    
    - job_name: 'gpu-metrics'
      static_configs:
      - targets: ['gpu-exporter:9445']
      scrape_interval: 5s

    # Rule groups for alerting
    ruleFiles:
    - /etc/prometheus/rules/*.yml

# AlertManager configuration
alertmanager:
  config:
    global:
      smtp_smarthost: 'localhost:587'
      smtp_from: 'alerts@its-camera-ai.com'
    
    route:
      group_by: ['alertname', 'service']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 12h
      receiver: 'web.hook'
      
      routes:
      - match:
          severity: critical
        receiver: 'pagerduty-critical'
        group_wait: 0s
        repeat_interval: 5m
      
      - match:
          severity: warning
        receiver: 'slack-warnings'
        repeat_interval: 30m
    
    receivers:
    - name: 'web.hook'
      webhook_configs:
      - url: 'http://analytics-alertmanager-webhook:9093/webhook'
    
    - name: 'pagerduty-critical'
      pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: 'ITS Analytics Critical Alert: {{ .GroupLabels.alertname }}'
    
    - name: 'slack-warnings'
      slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#its-alerts'
        title: 'ITS Analytics Warning'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

# Grafana configuration
grafana:
  adminPassword: "admin123!"  # Change in production
  
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 1000m
      memory: 2Gi
  
  persistence:
    enabled: true
    storageClassName: fast-ssd
    size: 10Gi
  
  # Data sources configuration
  datasources:
    datasources.yaml:
      apiVersion: 1
      datasources:
      - name: Prometheus
        type: prometheus
        url: http://prometheus-server:80
        access: proxy
        isDefault: true
      - name: Jaeger
        type: jaeger
        url: http://jaeger-query:16686
        access: proxy
      - name: Elasticsearch
        type: elasticsearch
        url: http://elasticsearch:9200
        access: proxy
        database: logstash-*
        timeField: "@timestamp"
  
  # Dashboard providers
  dashboardProviders:
    dashboardproviders.yaml:
      apiVersion: 1
      providers:
      - name: 'its-analytics'
        orgId: 1
        folder: 'ITS Analytics'
        type: file
        disableDeletion: false
        updateIntervalSeconds: 10
        options:
          path: /var/lib/grafana/dashboards/its-analytics
```

### Custom Alert Rules (`monitoring/alert-rules.yaml`)
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-alert-rules
  namespace: monitoring
data:
  analytics-alerts.yml: |
    groups:
    - name: its-analytics.rules
      interval: 10s
      rules:
      
      # Critical Performance Alerts
      - alert: AnalyticsLatencyHigh
        expr: histogram_quantile(0.99, rate(analytics_request_duration_seconds_bucket[1m])) > 0.100
        for: 1m
        labels:
          severity: critical
          service: analytics
        annotations:
          summary: "Analytics latency exceeded 100ms SLA"
          description: "p99 latency is {{ $value }}s, exceeding 100ms SLA for {{ $labels.instance }}"
          runbook_url: "https://docs.its-camera-ai.com/runbook/latency-high"
      
      - alert: GPUUtilizationLow
        expr: avg(gpu_utilization_percent) < 50
        for: 2m
        labels:
          severity: warning
          service: gpu
        annotations:
          summary: "GPU utilization below optimal threshold"
          description: "Average GPU utilization is {{ $value }}%, check batch sizing and model optimization"
      
      - alert: AnalyticsErrorRateHigh
        expr: rate(analytics_errors_total[5m]) / rate(analytics_requests_total[5m]) > 0.01
        for: 30s
        labels:
          severity: critical
          service: analytics
        annotations:
          summary: "Analytics error rate exceeded 1%"
          description: "Error rate is {{ $value | humanizePercentage }}, immediate investigation required"
      
      # Queue Management Alerts
      - alert: QueueDepthHigh
        expr: analytics_queue_depth > 800
        for: 30s
        labels:
          severity: warning
          service: analytics
        annotations:
          summary: "Analytics queue depth is high"
          description: "Queue depth is {{ $value }} items, approaching capacity limit of 1000"
      
      - alert: QueueDepthCritical
        expr: analytics_queue_depth > 950
        for: 10s
        labels:
          severity: critical
          service: analytics
        annotations:
          summary: "Analytics queue near capacity"
          description: "Queue depth is {{ $value }} items, immediate scaling required"
      
      # Resource Utilization Alerts
      - alert: AnalyticsMemoryHigh
        expr: container_memory_usage_bytes{container="analytics-service"} / container_spec_memory_limit_bytes > 0.90
        for: 2m
        labels:
          severity: warning
          service: analytics
        annotations:
          summary: "Analytics service memory usage high"
          description: "Memory usage is {{ $value | humanizePercentage }}, potential memory leak"
      
      - alert: DatabaseConnectionPoolHigh
        expr: db_connection_pool_active / db_connection_pool_max > 0.80
        for: 1m
        labels:
          severity: warning
          service: database
        annotations:
          summary: "Database connection pool utilization high"
          description: "Connection pool is {{ $value | humanizePercentage }} utilized"
      
      # Model Performance Alerts
      - alert: ModelDriftDetected
        expr: ml_model_drift_score > 0.1
        for: 5m
        labels:
          severity: warning
          service: ml-pipeline
        annotations:
          summary: "ML model drift detected"
          description: "Drift score is {{ $value }}, model retraining may be required"
      
      - alert: ModelAccuracyLow
        expr: ml_model_accuracy < 0.95
        for: 10m
        labels:
          severity: critical
          service: ml-pipeline
        annotations:
          summary: "ML model accuracy below threshold"
          description: "Model accuracy is {{ $value }}, below 95% requirement"
      
      # Cache Performance Alerts
      - alert: CacheHitRateLow
        expr: cache_hit_rate < 0.70
        for: 5m
        labels:
          severity: warning
          service: cache
        annotations:
          summary: "Cache hit rate is low"
          description: "Hit rate is {{ $value | humanizePercentage }}, cache optimization needed"
      
      # Service Health Alerts
      - alert: AnalyticsServiceDown
        expr: up{job="its-analytics-service"} == 0
        for: 30s
        labels:
          severity: critical
          service: analytics
        annotations:
          summary: "Analytics service is down"
          description: "Service {{ $labels.instance }} is not responding"
      
      - alert: ThroughputLow
        expr: rate(analytics_frames_processed_total[5m]) < 30
        for: 2m
        labels:
          severity: warning
          service: analytics
        annotations:
          summary: "Analytics throughput below target"
          description: "Current throughput is {{ $value }} FPS, below 30 FPS target"
```

## Grafana Dashboard Configuration

### Main Analytics Dashboard (`monitoring/dashboards/analytics-overview.json`)
```json
{
  "dashboard": {
    "id": null,
    "title": "ITS Analytics Service - Overview",
    "tags": ["its-analytics", "performance", "monitoring"],
    "timezone": "browser",
    "refresh": "5s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "id": 1,
        "title": "Latency Percentiles",
        "type": "stat",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(analytics_request_duration_seconds_bucket[1m]))",
            "legendFormat": "p50",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.95, rate(analytics_request_duration_seconds_bucket[1m]))",
            "legendFormat": "p95",
            "refId": "B"
          },
          {
            "expr": "histogram_quantile(0.99, rate(analytics_request_duration_seconds_bucket[1m]))",
            "legendFormat": "p99",
            "refId": "C"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 0.080},
                {"color": "red", "value": 0.100}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(analytics_requests_total[1m])",
            "legendFormat": "Requests/sec",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "requests/sec",
            "min": 0
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "avg(gpu_utilization_percent)",
            "legendFormat": "GPU Utilization %",
            "refId": "A"
          },
          {
            "expr": "avg(gpu_memory_utilization_percent)",
            "legendFormat": "GPU Memory %",
            "refId": "B"
          }
        ],
        "yAxes": [
          {
            "label": "percentage",
            "min": 0,
            "max": 100
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "Queue Metrics",
        "type": "graph",
        "targets": [
          {
            "expr": "analytics_queue_depth",
            "legendFormat": "Queue Depth",
            "refId": "A"
          },
          {
            "expr": "rate(analytics_queue_processed_total[1m])",
            "legendFormat": "Processing Rate",
            "refId": "B"
          }
        ],
        "yAxes": [
          {
            "label": "items",
            "min": 0
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      },
      {
        "id": 5,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(analytics_errors_total[1m])",
            "legendFormat": "Errors/sec",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "errors/sec",
            "min": 0
          }
        ],
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16}
      }
    ]
  }
}
```

### Performance Dashboard (`monitoring/dashboards/performance-detailed.json`)
```json
{
  "dashboard": {
    "title": "ITS Analytics - Performance Analysis",
    "panels": [
      {
        "title": "Latency Heatmap",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(analytics_request_duration_seconds_bucket[1m])",
            "format": "heatmap",
            "legendFormat": "{{le}}"
          }
        ],
        "heatmap": {
          "yAxis": {
            "unit": "s",
            "decimals": 3
          }
        }
      },
      {
        "title": "Throughput by Camera",
        "type": "table",
        "targets": [
          {
            "expr": "rate(analytics_frames_processed_total[5m]) by (camera_id)",
            "format": "table",
            "legendFormat": "FPS"
          }
        ],
        "transform": [
          {
            "id": "organize",
            "options": {
              "excludeByName": {},
              "indexByName": {},
              "renameByName": {
                "camera_id": "Camera ID",
                "Value": "FPS"
              }
            }
          }
        ]
      }
    ]
  }
}
```

## ELK Stack Setup

### Elasticsearch Installation
```bash
# Add Elastic Helm repository
helm repo add elastic https://helm.elastic.co
helm repo update

# Install Elasticsearch
helm install elasticsearch elastic/elasticsearch \
  --namespace monitoring \
  --values monitoring/elasticsearch-values.yaml
```

### Elasticsearch Configuration (`monitoring/elasticsearch-values.yaml`)
```yaml
# Elasticsearch configuration for log storage
clusterName: "its-analytics-logs"
nodeGroup: "master"

# Replica configuration for high availability
replicas: 3
minimumMasterNodes: 2

# Resource allocation
resources:
  requests:
    cpu: "1000m"
    memory: "4Gi"
  limits:
    cpu: "2000m"
    memory: "8Gi"

# Storage configuration
volumeClaimTemplate:
  accessModes: ["ReadWriteOnce"]
  storageClassName: "fast-ssd"
  resources:
    requests:
      storage: 100Gi

# JVM heap settings
esJavaOpts: "-Xmx4g -Xms4g"

# Elasticsearch configuration
esConfig:
  elasticsearch.yml: |
    cluster.name: "its-analytics-logs"
    network.host: 0.0.0.0
    bootstrap.memory_lock: false
    discovery.zen.ping.unicast.hosts: elasticsearch-master-headless
    discovery.zen.minimum_master_nodes: 2
    
    # Index management
    action.auto_create_index: true
    
    # Performance settings
    thread_pool.write.queue_size: 1000
    thread_pool.search.queue_size: 1000
    
    # Log retention
    cluster.routing.allocation.disk.watermark.low: "85%"
    cluster.routing.allocation.disk.watermark.high: "90%"
```

### Logstash Configuration
```bash
# Install Logstash
helm install logstash elastic/logstash \
  --namespace monitoring \
  --values monitoring/logstash-values.yaml
```

### Logstash Pipeline (`monitoring/logstash-values.yaml`)
```yaml
# Logstash configuration for log processing
resources:
  requests:
    cpu: "500m"
    memory: "2Gi"
  limits:
    cpu: "1000m"
    memory: "4Gi"

# Logstash pipeline configuration
logstashPipeline:
  logstash.conf: |
    input {
      beats {
        port => 5044
      }
      tcp {
        port => 5000
        codec => json_lines
      }
    }
    
    filter {
      # Parse analytics service logs
      if [service] == "analytics" {
        grok {
          match => { 
            "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{DATA:logger} - %{GREEDYDATA:log_message}"
          }
        }
        
        # Extract performance metrics from logs
        if [log_message] =~ /latency/ {
          grok {
            match => { 
              "log_message" => ".*latency: (?<latency_ms>\d+\.?\d*)ms.*"
            }
          }
          mutate {
            convert => { "latency_ms" => "float" }
          }
        }
        
        # Parse error logs
        if [level] == "ERROR" {
          mutate {
            add_tag => [ "error" ]
          }
        }
      }
      
      # Add metadata
      mutate {
        add_field => { 
          "[@metadata][index_name]" => "its-analytics-%{+YYYY.MM.dd}"
        }
      }
    }
    
    output {
      elasticsearch {
        hosts => ["elasticsearch-master:9200"]
        index => "%{[@metadata][index_name]}"
        template_name => "its-analytics"
        template => "/usr/share/logstash/templates/analytics-template.json"
        template_overwrite => true
      }
      
      # Debug output
      if [level] == "ERROR" {
        stdout { 
          codec => rubydebug 
        }
      }
    }

# Index template
logstashConfig:
  analytics-template.json: |
    {
      "index_patterns": ["its-analytics-*"],
      "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 1,
        "refresh_interval": "5s"
      },
      "mappings": {
        "properties": {
          "@timestamp": { "type": "date" },
          "level": { "type": "keyword" },
          "logger": { "type": "keyword" },
          "service": { "type": "keyword" },
          "latency_ms": { "type": "float" },
          "camera_id": { "type": "keyword" },
          "error_type": { "type": "keyword" }
        }
      }
    }
```

### Kibana Setup
```bash
# Install Kibana
helm install kibana elastic/kibana \
  --namespace monitoring \
  --values monitoring/kibana-values.yaml
```

### Kibana Configuration (`monitoring/kibana-values.yaml`)
```yaml
# Kibana configuration for log visualization
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "1000m"
    memory: "2Gi"

# Service configuration
service:
  type: ClusterIP
  port: 5601

# Kibana settings
kibanaConfig:
  kibana.yml: |
    server.host: 0.0.0.0
    elasticsearch.hosts: ["http://elasticsearch-master:9200"]
    
    # Dashboard settings
    server.defaultRoute: "/app/dashboards"
    
    # Security settings
    xpack.security.enabled: false
    xpack.monitoring.enabled: false
```

## Jaeger Tracing Setup

### Jaeger Installation
```bash
# Install Jaeger operator
kubectl create namespace observability-system
kubectl apply -n observability-system -f https://github.com/jaegertracing/jaeger-operator/releases/download/v1.35.0/jaeger-operator.yaml

# Wait for operator to be ready
kubectl wait --for=condition=available --timeout=300s deployment/jaeger-operator -n observability-system

# Deploy Jaeger instance
kubectl apply -f monitoring/jaeger-instance.yaml
```

### Jaeger Configuration (`monitoring/jaeger-instance.yaml`)
```yaml
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: its-analytics-tracing
  namespace: monitoring
spec:
  strategy: production
  
  # Storage configuration for production
  storage:
    type: elasticsearch
    elasticsearch:
      server-urls: http://elasticsearch-master:9200
      index-prefix: jaeger
      
  # Collector configuration
  collector:
    resources:
      requests:
        cpu: "500m"
        memory: "1Gi"
      limits:
        cpu: "1000m"
        memory: "2Gi"
    
    # High-throughput configuration
    config: |
      processors:
        batch:
          timeout: 1s
          send_batch_size: 1024
          send_batch_max_size: 2048
      
      exporters:
        jaeger:
          endpoint: jaeger-collector:14250
          tls:
            insecure: true
  
  # Query service configuration
  query:
    resources:
      requests:
        cpu: "200m"
        memory: "512Mi"
      limits:
        cpu: "500m"
        memory: "1Gi"
    
    # UI configuration
    config: |
      query:
        max-clock-skew-adjustment: 0
        max-trace-retention-time: 168h  # 7 days
```

### Application Instrumentation

#### Python OpenTelemetry Setup
```python
# requirements-monitoring.txt
opentelemetry-api==1.15.0
opentelemetry-sdk==1.15.0
opentelemetry-instrumentation-fastapi==0.36b0
opentelemetry-instrumentation-asyncpg==0.36b0
opentelemetry-instrumentation-redis==0.36b0
opentelemetry-exporter-jaeger==1.15.0
opentelemetry-exporter-prometheus==1.12.0rc1

# monitoring/tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

def setup_tracing(service_name: str = "its-analytics-service"):
    """Setup distributed tracing with Jaeger."""
    
    # Create tracer provider
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="jaeger-agent.monitoring.svc.cluster.local",
        agent_port=6831,
    )
    
    # Add span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Auto-instrument frameworks
    FastAPIInstrumentor.instrument()
    AsyncPGInstrumentor.instrument()
    RedisInstrumentor.instrument()
    
    return tracer

# Usage in analytics service
tracer = setup_tracing("its-analytics-service")

@tracer.start_as_current_span("process_ml_batch")
async def process_ml_batch(batch_results, frame_metadata):
    """Process ML batch with tracing."""
    
    with tracer.start_as_current_span("convert_ml_outputs") as span:
        span.set_attribute("batch_size", len(batch_results))
        detection_results = await self._convert_ml_outputs(batch_results, frame_metadata)
        span.set_attribute("detections_count", len(detection_results))
    
    with tracer.start_as_current_span("group_by_camera") as span:
        camera_groups = self._group_by_camera(detection_results)
        span.set_attribute("cameras_count", len(camera_groups))
    
    # Continue with traced operations...
```

## GPU Monitoring Setup

### NVIDIA GPU Exporter
```bash
# Deploy GPU metrics exporter
kubectl apply -f monitoring/gpu-exporter.yaml
```

### GPU Exporter Configuration (`monitoring/gpu-exporter.yaml`)
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: gpu-exporter
  namespace: monitoring
  labels:
    app: gpu-exporter
spec:
  selector:
    matchLabels:
      app: gpu-exporter
  template:
    metadata:
      labels:
        app: gpu-exporter
    spec:
      nodeSelector:
        gpu: "true"
      containers:
      - name: gpu-exporter
        image: utkuozdemir/nvidia_gpu_exporter:1.1.0
        ports:
        - containerPort: 9835
          name: metrics
        securityContext:
          privileged: true
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
        env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: all
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 200m
            memory: 256Mi
      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
      hostNetwork: true
      hostPID: true

---
apiVersion: v1
kind: Service
metadata:
  name: gpu-exporter
  namespace: monitoring
  labels:
    app: gpu-exporter
spec:
  ports:
  - port: 9835
    targetPort: 9835
    name: metrics
  selector:
    app: gpu-exporter

---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: gpu-exporter
  namespace: monitoring
  labels:
    app: gpu-exporter
spec:
  selector:
    matchLabels:
      app: gpu-exporter
  endpoints:
  - port: metrics
    interval: 5s
    path: /metrics
```

## Custom Metrics Implementation

### Analytics Service Metrics (`src/its_camera_ai/monitoring/metrics.py`)
```python
from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps
from typing import Callable, Any

# Define custom metrics for analytics service
ANALYTICS_REQUESTS_TOTAL = Counter(
    'analytics_requests_total',
    'Total number of analytics requests',
    ['method', 'status', 'camera_id']
)

ANALYTICS_REQUEST_DURATION = Histogram(
    'analytics_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'camera_id'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

ANALYTICS_QUEUE_DEPTH = Gauge(
    'analytics_queue_depth',
    'Current number of items in processing queue'
)

ANALYTICS_FRAMES_PROCESSED = Counter(
    'analytics_frames_processed_total',
    'Total number of frames processed',
    ['camera_id', 'model_version']
)

GPU_UTILIZATION = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['device_id']
)

GPU_MEMORY_UTILIZATION = Gauge(
    'gpu_memory_utilization_percent',
    'GPU memory utilization percentage',
    ['device_id']
)

ML_MODEL_ACCURACY = Gauge(
    'ml_model_accuracy',
    'Current ML model accuracy',
    ['model_name', 'model_version']
)

ML_MODEL_DRIFT_SCORE = Gauge(
    'ml_model_drift_score',
    'ML model drift detection score',
    ['model_name']
)

CACHE_HIT_RATE = Gauge(
    'cache_hit_rate',
    'Cache hit rate percentage',
    ['cache_level', 'cache_type']
)

DATABASE_CONNECTION_POOL_ACTIVE = Gauge(
    'db_connection_pool_active',
    'Active database connections'
)

DATABASE_CONNECTION_POOL_MAX = Gauge(
    'db_connection_pool_max',
    'Maximum database connections'
)

def track_request_metrics(method: str = "unknown"):
    """Decorator to track request metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            camera_id = kwargs.get('camera_id', 'unknown')
            
            start_time = time.time()
            status = 'success'
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                
                # Record metrics
                ANALYTICS_REQUESTS_TOTAL.labels(
                    method=method,
                    status=status,
                    camera_id=camera_id
                ).inc()
                
                ANALYTICS_REQUEST_DURATION.labels(
                    method=method,
                    camera_id=camera_id
                ).observe(duration)
        
        return wrapper
    return decorator

class MetricsCollector:
    """Centralized metrics collection for analytics service."""
    
    def __init__(self):
        self.gpu_monitor_active = False
        self.system_monitor_active = False
        
    async def start_collection(self):
        """Start all metrics collection."""
        import asyncio
        
        self.gpu_monitor_active = True
        self.system_monitor_active = True
        
        # Start background collection tasks
        asyncio.create_task(self._collect_gpu_metrics())
        asyncio.create_task(self._collect_system_metrics())
    
    async def _collect_gpu_metrics(self):
        """Collect GPU utilization metrics."""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            
            while self.gpu_monitor_active:
                for i in range(device_count):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        
                        # GPU utilization
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        GPU_UTILIZATION.labels(device_id=str(i)).set(utilization.gpu)
                        
                        # GPU memory utilization
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        memory_util = (memory_info.used / memory_info.total) * 100
                        GPU_MEMORY_UTILIZATION.labels(device_id=str(i)).set(memory_util)
                        
                    except Exception as e:
                        logger.warning(f"GPU metrics collection failed for device {i}: {e}")
                
                await asyncio.sleep(5)  # Collect every 5 seconds
                
        except ImportError:
            logger.warning("pynvml not available, GPU metrics disabled")
        except Exception as e:
            logger.error(f"GPU metrics collection error: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        while self.system_monitor_active:
            try:
                # Update queue depth (would be injected from actual queue)
                # ANALYTICS_QUEUE_DEPTH.set(current_queue_size)
                
                # Update connection pool metrics (would be injected from pool)
                # DATABASE_CONNECTION_POOL_ACTIVE.set(active_connections)
                # DATABASE_CONNECTION_POOL_MAX.set(max_connections)
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
    
    def update_model_metrics(self, model_name: str, model_version: str, 
                           accuracy: float, drift_score: float):
        """Update ML model metrics."""
        ML_MODEL_ACCURACY.labels(
            model_name=model_name,
            model_version=model_version
        ).set(accuracy)
        
        ML_MODEL_DRIFT_SCORE.labels(model_name=model_name).set(drift_score)
    
    def update_cache_metrics(self, cache_level: str, cache_type: str, hit_rate: float):
        """Update cache performance metrics."""
        CACHE_HIT_RATE.labels(
            cache_level=cache_level,
            cache_type=cache_type
        ).set(hit_rate)
    
    def record_frame_processing(self, camera_id: str, model_version: str, count: int = 1):
        """Record frame processing metrics."""
        ANALYTICS_FRAMES_PROCESSED.labels(
            camera_id=camera_id,
            model_version=model_version
        ).inc(count)

# Global metrics collector instance
metrics_collector = MetricsCollector()
```

## Deployment Commands

### Complete Monitoring Stack Deployment
```bash
#!/bin/bash
# deploy-monitoring.sh - Complete monitoring stack deployment

set -e

echo "🚀 Deploying ITS Analytics Monitoring Stack..."

# Create namespaces
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace observability-system --dry-run=client -o yaml | kubectl apply -f -

# Add Helm repositories
echo "📦 Adding Helm repositories..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add elastic https://helm.elastic.co
helm repo update

# Deploy Prometheus + Grafana + AlertManager
echo "📊 Deploying Prometheus stack..."
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --values monitoring/prometheus-values.yaml \
  --wait --timeout=10m

# Deploy Elasticsearch
echo "📝 Deploying Elasticsearch..."
helm upgrade --install elasticsearch elastic/elasticsearch \
  --namespace monitoring \
  --values monitoring/elasticsearch-values.yaml \
  --wait --timeout=15m

# Deploy Logstash
echo "🔄 Deploying Logstash..."
helm upgrade --install logstash elastic/logstash \
  --namespace monitoring \
  --values monitoring/logstash-values.yaml \
  --wait --timeout=10m

# Deploy Kibana
echo "🔍 Deploying Kibana..."
helm upgrade --install kibana elastic/kibana \
  --namespace monitoring \
  --values monitoring/kibana-values.yaml \
  --wait --timeout=10m

# Deploy Jaeger operator
echo "🔍 Deploying Jaeger tracing..."
kubectl apply -n observability-system -f https://github.com/jaegertracing/jaeger-operator/releases/download/v1.35.0/jaeger-operator.yaml
kubectl wait --for=condition=available --timeout=300s deployment/jaeger-operator -n observability-system

# Deploy Jaeger instance
kubectl apply -f monitoring/jaeger-instance.yaml

# Deploy GPU exporter
echo "🖥️ Deploying GPU metrics exporter..."
kubectl apply -f monitoring/gpu-exporter.yaml

# Deploy alert rules
echo "🚨 Deploying alert rules..."
kubectl apply -f monitoring/alert-rules.yaml

# Configure Grafana dashboards
echo "📈 Configuring Grafana dashboards..."
kubectl create configmap analytics-dashboards \
  --from-file=monitoring/dashboards/ \
  --namespace monitoring \
  --dry-run=client -o yaml | kubectl apply -f -

# Wait for all services to be ready
echo "⏳ Waiting for services to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/prometheus-grafana -n monitoring
kubectl wait --for=condition=ready --timeout=300s pod -l app=elasticsearch-master -n monitoring

echo "✅ Monitoring stack deployed successfully!"

# Print access information
echo ""
echo "🌐 Access URLs:"
echo "Grafana: kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80"
echo "Prometheus: kubectl port-forward -n monitoring svc/prometheus-server 9090:80"
echo "Kibana: kubectl port-forward -n monitoring svc/kibana-kibana 5601:5601"
echo "Jaeger: kubectl port-forward -n monitoring svc/its-analytics-tracing-query 16686:16686"

echo ""
echo "🔑 Grafana credentials:"
echo "Username: admin"
echo "Password: $(kubectl get secret prometheus-grafana -n monitoring -o jsonpath='{.data.admin-password}' | base64 -d)"
```

### Verification Script
```bash
#!/bin/bash
# verify-monitoring.sh - Verify monitoring stack health

echo "🔍 Verifying ITS Analytics Monitoring Stack..."

# Check Prometheus
echo "📊 Checking Prometheus..."
if kubectl get pods -n monitoring -l app.kubernetes.io/name=prometheus | grep Running; then
  echo "✅ Prometheus is running"
else
  echo "❌ Prometheus is not running"
  exit 1
fi

# Check Grafana
echo "📈 Checking Grafana..."
if kubectl get pods -n monitoring -l app.kubernetes.io/name=grafana | grep Running; then
  echo "✅ Grafana is running"
else
  echo "❌ Grafana is not running"
  exit 1
fi

# Check Elasticsearch
echo "📝 Checking Elasticsearch..."
if kubectl get pods -n monitoring -l app=elasticsearch-master | grep Running; then
  echo "✅ Elasticsearch is running"
else
  echo "❌ Elasticsearch is not running"
  exit 1
fi

# Test metrics endpoint
echo "🔗 Testing metrics endpoint..."
if kubectl exec -n monitoring deployment/prometheus-server -- wget -qO- http://its-analytics-service.its-camera-ai:8000/metrics | grep analytics_ > /dev/null; then
  echo "✅ Analytics metrics available"
else
  echo "⚠️ Analytics metrics not available (service may not be deployed)"
fi

# Check alerts
echo "🚨 Checking alert rules..."
ALERT_COUNT=$(kubectl exec -n monitoring deployment/prometheus-server -- wget -qO- http://localhost:9090/api/v1/rules | jq '.data.groups[].rules[] | select(.type=="alerting") | .name' | wc -l)
echo "📊 $ALERT_COUNT alert rules configured"

echo "✅ Monitoring stack verification complete!"
```

This comprehensive monitoring setup provides:

- **Real-time performance monitoring** with sub-second granularity
- **Comprehensive alerting** for all critical system metrics
- **Distributed tracing** for request flow analysis
- **Log aggregation** for troubleshooting and analysis
- **GPU monitoring** for ML workload optimization
- **Automated deployment** scripts for production environments

The monitoring stack is designed to handle high-throughput analytics workloads while maintaining minimal overhead and providing actionable insights for system optimization.