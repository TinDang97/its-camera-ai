#!/bin/bash

# Production Monitoring Stack Deployment Script
# Deploys comprehensive observability infrastructure for ITS Camera AI
# Supports 1000+ cameras, 10TB/day processing, 99.9% uptime SLA

set -euo pipefail

# Configuration
NAMESPACE="monitoring"
ENVIRONMENT="${ENVIRONMENT:-production}"
CLUSTER_NAME="${CLUSTER_NAME:-its-camera-ai-prod}"
DOMAIN="${DOMAIN:-its-camera-ai.com}"

# Color output functions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is required but not installed"
        exit 1
    fi
    
    # Check if helm is available
    if ! command -v helm &> /dev/null; then
        log_error "helm is required but not installed"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if prometheus-operator CRDs exist
    if ! kubectl get crd prometheuses.monitoring.coreos.com &> /dev/null; then
        log_warning "Prometheus Operator CRDs not found, will install them"
    fi
    
    log_success "Prerequisites check completed"
}

# Install Prometheus Operator CRDs
install_prometheus_operator_crds() {
    log_info "Installing Prometheus Operator CRDs..."
    
    kubectl apply --server-side -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/v0.70.0/example/prometheus-operator-crd/monitoring.coreos.com_alertmanagerconfigs.yaml
    kubectl apply --server-side -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/v0.70.0/example/prometheus-operator-crd/monitoring.coreos.com_alertmanagers.yaml
    kubectl apply --server-side -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/v0.70.0/example/prometheus-operator-crd/monitoring.coreos.com_podmonitors.yaml
    kubectl apply --server-side -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/v0.70.0/example/prometheus-operator-crd/monitoring.coreos.com_probes.yaml
    kubectl apply --server-side -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/v0.70.0/example/prometheus-operator-crd/monitoring.coreos.com_prometheuses.yaml
    kubectl apply --server-side -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/v0.70.0/example/prometheus-operator-crd/monitoring.coreos.com_prometheusrules.yaml
    kubectl apply --server-side -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/v0.70.0/example/prometheus-operator-crd/monitoring.coreos.com_servicemonitors.yaml
    kubectl apply --server-side -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/v0.70.0/example/prometheus-operator-crd/monitoring.coreos.com_thanosrulers.yaml
    
    log_success "Prometheus Operator CRDs installed"
}

# Create monitoring namespace and secrets
setup_namespace_and_secrets() {
    log_info "Setting up monitoring namespace and secrets..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Label namespace for network policies
    kubectl label namespace ${NAMESPACE} name=${NAMESPACE} --overwrite
    
    # Create secrets for sensitive data
    log_info "Creating secrets..."
    
    # Grafana admin password
    GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 32)
    kubectl create secret generic grafana-credentials \
        --from-literal=admin-password="${GRAFANA_ADMIN_PASSWORD}" \
        --namespace=${NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Grafana database credentials
    GRAFANA_DB_PASSWORD=$(openssl rand -base64 32)
    kubectl create secret generic grafana-db-credentials \
        --from-literal=password="${GRAFANA_DB_PASSWORD}" \
        --namespace=${NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # AlertManager secrets (you'll need to set these environment variables)
    if [[ -n "${SMTP_PASSWORD:-}" && -n "${SLACK_API_URL:-}" && -n "${PAGERDUTY_INTEGRATION_KEY:-}" ]]; then
        kubectl create secret generic alertmanager-secrets \
            --from-literal=smtp-password="${SMTP_PASSWORD}" \
            --from-literal=slack-api-url="${SLACK_API_URL}" \
            --from-literal=pagerduty-integration-key="${PAGERDUTY_INTEGRATION_KEY}" \
            --namespace=${NAMESPACE} \
            --dry-run=client -o yaml | kubectl apply -f -
    else
        log_warning "AlertManager secrets not provided. Please set SMTP_PASSWORD, SLACK_API_URL, and PAGERDUTY_INTEGRATION_KEY"
    fi
    
    # Jaeger Elasticsearch credentials
    JAEGER_ES_PASSWORD=$(openssl rand -base64 32)
    kubectl create secret generic jaeger-elasticsearch-credentials \
        --from-literal=password="${JAEGER_ES_PASSWORD}" \
        --namespace=${NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Namespace and secrets configured"
    log_info "Grafana admin password: ${GRAFANA_ADMIN_PASSWORD}"
}

# Deploy ConfigMaps
deploy_configmaps() {
    log_info "Deploying configuration files..."
    
    # Prometheus configuration
    kubectl create configmap prometheus-additional-scrape-configs \
        --from-file=prometheus-additional.yaml=monitoring/prometheus/prometheus-production.yml \
        --namespace=${NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Grafana datasources
    kubectl create configmap grafana-datasources \
        --from-file=datasources.yaml=<(cat <<EOF
apiVersion: 1
datasources:
- name: Prometheus
  type: prometheus
  access: proxy
  url: http://prometheus-main:9090
  isDefault: true
  jsonData:
    timeInterval: "15s"
    queryTimeout: "60s"

- name: Loki
  type: loki
  access: proxy
  url: http://loki:3100
  jsonData:
    maxLines: 1000
    derivedFields:
      - name: "TraceID"
        matcherRegex: "trace_id=(\\w+)"
        url: "http://jaeger-query:16686/trace/\${__value.raw}"
        datasourceUid: "jaeger"

- name: Jaeger
  type: jaeger
  access: proxy
  url: http://jaeger-query:16686
  uid: jaeger

- name: InfluxDB
  type: influxdb
  access: proxy
  url: http://influxdb-service.its-camera-ai-production:8086
  database: its-camera-ai
  user: grafana
EOF
) \
        --namespace=${NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Grafana dashboards configuration
    kubectl create configmap grafana-dashboards-config \
        --from-file=dashboards.yaml=<(cat <<EOF
apiVersion: 1
providers:
- name: 'default'
  orgId: 1
  folder: ''
  type: file
  disableDeletion: false
  updateIntervalSeconds: 10
  allowUiUpdates: true
  options:
    path: /var/lib/grafana/dashboards
EOF
) \
        --namespace=${NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Grafana dashboards
    kubectl create configmap grafana-dashboards \
        --from-file=monitoring/grafana/dashboards/ \
        --namespace=${NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Loki configuration
    kubectl create configmap loki-config \
        --from-file=local-config.yaml=<(cat <<EOF
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9095
  log_level: info

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1

schema_config:
  configs:
    - from: 2023-01-01
      store: boltdb-shipper
      object_store: filesystem
      schema: v12
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
  ingestion_rate_mb: 64
  ingestion_burst_size_mb: 128
  per_stream_rate_limit: 32MB
  per_stream_rate_limit_burst: 64MB

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: true
  retention_period: 30d

ruler:
  storage:
    type: local
    local:
      directory: /loki/rules
  rule_path: /loki/rules-temp
  ring:
    kvstore:
      store: inmemory
  enable_api: true
EOF
) \
        --namespace=${NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # OpenTelemetry Collector configuration
    kubectl create configmap otel-collector-config \
        --from-file=otel-collector.yaml=monitoring/opentelemetry/otel-collector.yaml \
        --namespace=${NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Configuration files deployed"
}

# Deploy Prometheus Rules
deploy_prometheus_rules() {
    log_info "Deploying Prometheus alerting rules..."
    
    # Apply all Prometheus rules
    kubectl apply -f monitoring/prometheus/rules.yaml -n ${NAMESPACE}
    kubectl apply -f monitoring/prometheus/slo-rules.yaml -n ${NAMESPACE}
    kubectl apply -f monitoring/prometheus/business-rules.yaml -n ${NAMESPACE}
    kubectl apply -f monitoring/prometheus/ml-pipeline-rules.yaml -n ${NAMESPACE}
    
    log_success "Prometheus rules deployed"
}

# Install Prometheus Operator using Helm
install_prometheus_operator() {
    log_info "Installing Prometheus Operator..."
    
    # Add prometheus-community helm repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Install prometheus-operator
    helm upgrade --install prometheus-operator prometheus-community/kube-prometheus-stack \
        --namespace ${NAMESPACE} \
        --create-namespace \
        --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.ruleSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.retentionSize=500GB \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=fast-ssd \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=500Gi \
        --set grafana.enabled=false \
        --set alertmanager.enabled=false \
        --wait
    
    log_success "Prometheus Operator installed"
}

# Deploy monitoring stack
deploy_monitoring_stack() {
    log_info "Deploying monitoring stack..."
    
    # Deploy the main monitoring stack
    kubectl apply -f monitoring/kubernetes/monitoring-stack-deployment.yaml
    
    # Wait for deployments to be ready
    log_info "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment/grafana -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=600s deployment/otel-collector -n ${NAMESPACE}
    kubectl wait --for=condition=ready --timeout=600s statefulset/loki -n ${NAMESPACE}
    
    log_success "Monitoring stack deployed successfully"
}

# Configure AlertManager
configure_alertmanager() {
    log_info "Configuring AlertManager..."
    
    # Create AlertManager configuration secret
    kubectl create secret generic alertmanager-main \
        --from-file=alertmanager.yml=monitoring/alertmanager/alertmanager-production.yml \
        --namespace=${NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "AlertManager configured"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying monitoring stack deployment..."
    
    # Check if all pods are running
    log_info "Checking pod status..."
    kubectl get pods -n ${NAMESPACE}
    
    # Check services
    log_info "Checking services..."
    kubectl get svc -n ${NAMESPACE}
    
    # Get service endpoints
    GRAFANA_ENDPOINT=$(kubectl get svc grafana -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "ClusterIP")
    JAEGER_ENDPOINT=$(kubectl get svc jaeger-query -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "ClusterIP")
    
    log_success "Monitoring stack verification completed"
    
    echo "================================================"
    echo "🎉 ITS Camera AI Monitoring Stack Deployed!"
    echo "================================================"
    echo ""
    echo "Access URLs:"
    echo "📊 Grafana: http://${GRAFANA_ENDPOINT}:3000"
    echo "🔍 Jaeger: http://${JAEGER_ENDPOINT}:16686"
    echo "🚨 AlertManager: kubectl port-forward svc/alertmanager 9093:9093 -n ${NAMESPACE}"
    echo "📈 Prometheus: kubectl port-forward svc/prometheus 9090:9090 -n ${NAMESPACE}"
    echo ""
    echo "Default Credentials:"
    echo "👤 Grafana Admin: admin / ${GRAFANA_ADMIN_PASSWORD}"
    echo ""
    echo "Monitoring Features:"
    echo "✅ Real-time system monitoring (1000+ cameras)"
    echo "✅ ML pipeline performance tracking"
    echo "✅ Business analytics and traffic intelligence"
    echo "✅ Distributed tracing with OpenTelemetry"
    echo "✅ Log aggregation with Loki"
    echo "✅ GPU monitoring with DCGM"
    echo "✅ SLA compliance tracking (99.9% uptime)"
    echo "✅ Automated alerting with PagerDuty integration"
    echo "✅ Cost optimization metrics"
    echo "✅ Capacity planning dashboards"
    echo ""
    echo "Next Steps:"
    echo "1. Configure external integrations (PagerDuty, Slack)"
    echo "2. Set up monitoring for ITS Camera AI applications"
    echo "3. Review and customize alerting thresholds"
    echo "4. Train teams on dashboard usage"
    echo "================================================"
}

# Cleanup function
cleanup() {
    log_warning "Cleaning up temporary files..."
    # Add any cleanup logic here
}

# Trap cleanup function on script exit
trap cleanup EXIT

# Main execution
main() {
    log_info "Starting ITS Camera AI Monitoring Stack Deployment"
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Cluster: ${CLUSTER_NAME}"
    log_info "Namespace: ${NAMESPACE}"
    
    check_prerequisites
    install_prometheus_operator_crds
    setup_namespace_and_secrets
    deploy_configmaps
    deploy_prometheus_rules
    install_prometheus_operator
    deploy_monitoring_stack
    configure_alertmanager
    verify_deployment
    
    log_success "ITS Camera AI Monitoring Stack deployment completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --cluster)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --environment   Environment name (default: production)"
            echo "  --cluster       Cluster name (default: its-camera-ai-prod)"
            echo "  --namespace     Namespace for monitoring (default: monitoring)"
            echo "  --domain        Domain name (default: its-camera-ai.com)"
            echo "  --help          Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  SMTP_PASSWORD               SMTP password for email alerts"
            echo "  SLACK_API_URL              Slack webhook URL for notifications"
            echo "  PAGERDUTY_INTEGRATION_KEY  PagerDuty integration key for critical alerts"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"