#!/bin/bash

# Deploy Prometheus and NVIDIA DCGM Monitoring Stack for ITS Camera AI
# This script deploys comprehensive monitoring with GPU metrics collection

set -euo pipefail

# Configuration
CLUSTER_NAME="its-camera-ai-prod"
REGION="us-west-2"
MONITORING_NAMESPACE="monitoring"

# Colors for output
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
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check cluster access
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot access Kubernetes cluster. Please check kubeconfig."
        exit 1
    fi
    
    # Check if GPU nodes are available
    GPU_NODES=$(kubectl get nodes -l accelerator=nvidia-tesla-v100 --no-headers 2>/dev/null | wc -l || echo "0")
    if [ "$GPU_NODES" -eq 0 ]; then
        log_warning "No GPU nodes found with label 'accelerator=nvidia-tesla-v100'. DCGM exporter will not be deployed."
        log_warning "Please ensure GPU nodes are properly labeled for GPU monitoring."
    else
        log_success "Found $GPU_NODES GPU nodes available for monitoring"
    fi
    
    log_success "Prerequisites check passed"
}

# Deploy monitoring namespace
deploy_namespace() {
    log_info "Creating monitoring namespace..."
    kubectl apply -f k8s/monitoring/namespace.yaml
    log_success "Monitoring namespace created"
}

# Deploy Prometheus
deploy_prometheus() {
    log_info "Deploying Prometheus server..."
    
    # Apply Prometheus configuration
    kubectl apply -f k8s/monitoring/prometheus.yaml
    
    # Wait for Prometheus to be ready
    log_info "Waiting for Prometheus to be ready..."
    kubectl wait --for=condition=available deployment/prometheus -n $MONITORING_NAMESPACE --timeout=600s
    
    log_success "Prometheus deployed successfully"
}

# Deploy NVIDIA DCGM Exporter
deploy_dcgm_exporter() {
    log_info "Deploying NVIDIA DCGM Exporter for GPU monitoring..."
    
    # Check if GPU nodes exist before deploying
    GPU_NODES=$(kubectl get nodes -l accelerator=nvidia-tesla-v100 --no-headers 2>/dev/null | wc -l || echo "0")
    if [ "$GPU_NODES" -eq 0 ]; then
        log_warning "Skipping DCGM exporter deployment - no GPU nodes available"
        return 0
    fi
    
    # Apply DCGM exporter configuration
    kubectl apply -f k8s/monitoring/dcgm-exporter.yaml
    
    # Wait for DCGM exporter to be ready
    log_info "Waiting for DCGM exporter to be ready..."
    kubectl rollout status daemonset/dcgm-exporter -n $MONITORING_NAMESPACE --timeout=300s
    
    # Wait for Node exporter to be ready
    log_info "Waiting for Node exporter to be ready..."
    kubectl rollout status daemonset/node-exporter -n $MONITORING_NAMESPACE --timeout=300s
    
    log_success "DCGM and Node exporters deployed successfully"
}

# Deploy Grafana
deploy_grafana() {
    log_info "Deploying Grafana with pre-configured dashboards..."
    
    # Apply Grafana configuration
    kubectl apply -f k8s/monitoring/grafana.yaml
    
    # Wait for Grafana to be ready
    log_info "Waiting for Grafana to be ready..."
    kubectl wait --for=condition=available deployment/grafana -n $MONITORING_NAMESPACE --timeout=600s
    
    log_success "Grafana deployed successfully"
}

# Deploy AlertManager
deploy_alertmanager() {
    log_info "Deploying AlertManager with PagerDuty integration..."
    
    # Apply AlertManager configuration
    kubectl apply -f k8s/monitoring/alertmanager.yaml
    
    # Wait for AlertManager to be ready
    log_info "Waiting for AlertManager to be ready..."
    kubectl wait --for=condition=available deployment/alertmanager -n $MONITORING_NAMESPACE --timeout=600s
    
    log_success "AlertManager deployed successfully"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying monitoring deployment..."
    
    echo
    log_info "=== Monitoring Stack Status ==="
    
    # Check namespace
    if kubectl get namespace $MONITORING_NAMESPACE &> /dev/null; then
        log_success "Namespace: $MONITORING_NAMESPACE exists"
    else
        log_error "Namespace: $MONITORING_NAMESPACE not found"
    fi
    
    # Check pods
    log_info "Pod status:"
    kubectl get pods -n $MONITORING_NAMESPACE -o wide
    
    # Check services
    log_info "Service status:"
    kubectl get svc -n $MONITORING_NAMESPACE
    
    # Check PVCs
    log_info "Storage status:"
    kubectl get pvc -n $MONITORING_NAMESPACE
    
    # Test Prometheus connectivity
    log_info "Testing Prometheus connectivity..."
    
    # Port forward to Prometheus (in background)
    kubectl port-forward svc/prometheus 9090:9090 -n $MONITORING_NAMESPACE &
    PORT_FORWARD_PID=$!
    
    # Wait for port forward to establish
    sleep 5
    
    # Test Prometheus health
    if command -v curl &> /dev/null; then
        log_info "Testing Prometheus health endpoint..."
        if curl -s http://localhost:9090/-/healthy | grep -q "Prometheus is Healthy"; then
            log_success "Prometheus is healthy and responding"
        else
            log_warning "Prometheus health check failed"
        fi
        
        log_info "Testing Prometheus targets..."
        TARGETS=$(curl -s http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets | length' 2>/dev/null || echo "unknown")
        log_info "Prometheus is monitoring $TARGETS active targets"
    else
        log_warning "curl not found. Skipping connectivity test."
    fi
    
    # Kill port forward
    kill $PORT_FORWARD_PID 2>/dev/null || true
    
    echo
    log_info "=== Connection Information ==="
    echo "Prometheus UI: http://prometheus-external:9090 (via LoadBalancer)"
    echo "Grafana UI: http://grafana-external:3000 (via LoadBalancer) - admin/its-camera-ai-grafana-admin-password"
    echo "AlertManager UI: http://alertmanager-external:9093 (via LoadBalancer)"
    echo "Internal Prometheus: http://prometheus.monitoring.svc.cluster.local:9090"
    echo "Internal Grafana: http://grafana.monitoring.svc.cluster.local:3000"
    echo "Internal AlertManager: http://alertmanager.monitoring.svc.cluster.local:9093"
    echo "DCGM Metrics: http://dcgm-exporter.monitoring.svc.cluster.local:9400/metrics"
    echo "Node Metrics: http://node-exporter.monitoring.svc.cluster.local:9100/metrics"
    echo
    
    log_info "=== Monitoring Endpoints ==="
    echo "GPU Metrics: dcgm-exporter.monitoring.svc.cluster.local:9400/metrics"
    echo "System Metrics: node-exporter.monitoring.svc.cluster.local:9100/metrics"
    echo "Database Metrics: postgres-exporter-coordinator.postgresql-cluster.svc.cluster.local:9187/metrics"
    echo "Connection Pool: pgbouncer.postgresql-cluster.svc.cluster.local:9127/metrics"
    echo
}

# Test GPU monitoring
test_gpu_monitoring() {
    log_info "Testing GPU monitoring capabilities..."
    
    GPU_NODES=$(kubectl get nodes -l accelerator=nvidia-tesla-v100 --no-headers 2>/dev/null | wc -l || echo "0")
    if [ "$GPU_NODES" -eq 0 ]; then
        log_warning "No GPU nodes available for testing"
        return 0
    fi
    
    # Port forward to DCGM exporter (if available)
    DCGM_POD=$(kubectl get pods -n $MONITORING_NAMESPACE -l app=dcgm-exporter -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [ -n "$DCGM_POD" ]; then
        log_info "Testing DCGM exporter metrics..."
        kubectl port-forward pod/$DCGM_POD 9400:9400 -n $MONITORING_NAMESPACE &
        PORT_FORWARD_PID=$!
        
        # Wait for port forward to establish
        sleep 5
        
        if command -v curl &> /dev/null; then
            log_info "Fetching GPU metrics sample..."
            GPU_METRICS=$(curl -s http://localhost:9400/metrics 2>/dev/null | grep "DCGM_FI_DEV_GPU_UTIL" | head -3 || echo "No metrics found")
            echo "Sample GPU utilization metrics:"
            echo "$GPU_METRICS"
        fi
        
        # Kill port forward
        kill $PORT_FORWARD_PID 2>/dev/null || true
        
        log_success "GPU monitoring test completed"
    else
        log_warning "DCGM exporter pod not found. GPU monitoring may not be working properly."
    fi
}

# Show next steps
show_next_steps() {
    echo
    log_success "Monitoring stack deployment completed successfully!"
    echo
    log_info "Next steps:"
    echo "  1. Access Grafana UI via LoadBalancer endpoint (admin/its-camera-ai-grafana-admin-password)"
    echo "  2. Explore pre-configured dashboards: GPU Performance, ML Inference, Database, Camera Analytics"
    echo "  3. Configure PagerDuty integration keys in AlertManager secrets"
    echo "  4. Set up Slack webhooks for team notifications"
    echo "  5. Test alert routing and escalation policies"
    echo "  6. Configure long-term storage (Thanos/Cortex) if needed"
    echo
    log_info "Key monitoring capabilities:"
    echo "  • GPU utilization, memory, temperature monitoring via DCGM"
    echo "  • System metrics (CPU, memory, disk, network) via Node Exporter"
    echo "  • Database performance metrics via PostgreSQL exporters"
    echo "  • Connection pool monitoring via PgBouncer exporters"
    echo "  • Kubernetes cluster metrics via kube-state-metrics"
    echo "  • Custom application metrics from ITS Camera AI services"
    echo "  • Critical alerting with PagerDuty, Slack, and email notifications"
    echo "  • Multi-channel alert routing based on severity and component"
    echo
    log_info "Prometheus query examples:"
    echo "  • GPU utilization: DCGM_FI_DEV_GPU_UTIL"
    echo "  • GPU memory usage: DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL * 100"
    echo "  • Database connections: pg_stat_database_numbackends"
    echo "  • Node CPU usage: 100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)"
    echo
}

# Main deployment function
main() {
    log_info "Starting monitoring stack deployment..."
    echo "Cluster: $CLUSTER_NAME"
    echo "Region: $REGION"
    echo "Namespace: $MONITORING_NAMESPACE"
    echo
    
    check_prerequisites
    deploy_namespace
    deploy_prometheus
    deploy_dcgm_exporter
    deploy_grafana
    deploy_alertmanager
    verify_deployment
    test_gpu_monitoring
    show_next_steps
}

# Handle script interruption
trap 'log_error "Script interrupted"; exit 1' INT TERM

# Run main function
main "$@"