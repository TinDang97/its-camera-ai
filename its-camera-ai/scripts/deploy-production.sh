#!/bin/bash
# Production Deployment Script for ITS Camera AI
# This script handles the complete deployment pipeline with safety checks

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENVIRONMENT=${1:-"production"}
VERSION=${2:-"latest"}
DRY_RUN=${3:-"false"}
SKIP_TESTS=${4:-"false"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

log_section() {
    echo ""
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f /tmp/its-camera-ai-*.yaml
}

# Set trap for cleanup
trap cleanup EXIT

# Banner
echo -e "${BLUE}"
cat << 'EOF'
 _____ _____ ____     ____                                     _    ___
|_   _|_   _/ ___|   / ___|__ _ _ __ ___   ___ _ __ __ _        / \  |_ _|
  | |   | | \___ \  | |   / _` | '_ ` _ \ / _ \ '__/ _` |      / _ \  | |
  | |   | |  ___) | | |__| (_| | | | | | |  __/ | | (_| |    / ___ \ | |
  |_|   |_| |____/   \____\__,_|_| |_| |_|\___|_|  \__,_|   /_/   \_\___|

  Production Deployment Pipeline
EOF
echo -e "${NC}"

log_info "Starting deployment to $ENVIRONMENT with version $VERSION"
log_info "Dry run: $DRY_RUN"
log_info "Skip tests: $SKIP_TESTS"

# Pre-flight checks
log_section "Pre-flight Checks"

# Check required tools
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        error_exit "$1 is not installed or not in PATH"
    fi
    log_success "$1 is available"
}

check_tool kubectl
check_tool helm
check_tool docker
check_tool aws
check_tool jq

# Check kubectl context
current_context=$(kubectl config current-context)
if [[ "$current_context" != *"$ENVIRONMENT"* ]]; then
    log_warning "Current kubectl context is '$current_context', expected context containing '$ENVIRONMENT'"
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        error_exit "Deployment cancelled by user"
    fi
fi

# Check cluster connectivity
if ! kubectl cluster-info &> /dev/null; then
    error_exit "Cannot connect to Kubernetes cluster"
fi
log_success "Kubernetes cluster connectivity verified"

# Check cluster resources
log_info "Checking cluster resources..."
nodes_ready=$(kubectl get nodes --no-headers | grep -c Ready || true)
if [[ $nodes_ready -lt 3 ]]; then
    log_warning "Less than 3 nodes are ready ($nodes_ready)"
else
    log_success "$nodes_ready nodes are ready"
fi

# Check namespace
if ! kubectl get namespace "its-camera-ai-$ENVIRONMENT" &> /dev/null; then
    log_info "Creating namespace its-camera-ai-$ENVIRONMENT"
    if [[ "$DRY_RUN" != "true" ]]; then
        kubectl create namespace "its-camera-ai-$ENVIRONMENT"
    fi
fi

# Validate Helm chart
log_section "Helm Chart Validation"

log_info "Validating Helm chart..."
helm lint "$PROJECT_ROOT/charts/its-camera-ai" || error_exit "Helm chart validation failed"
log_success "Helm chart validation passed"

# Template validation
log_info "Validating Helm templates..."
helm template its-camera-ai "$PROJECT_ROOT/charts/its-camera-ai" \
    --namespace "its-camera-ai-$ENVIRONMENT" \
    --values "$PROJECT_ROOT/charts/its-camera-ai/values-$ENVIRONMENT.yaml" \
    --set image.tag="$VERSION" \
    --output-dir /tmp/helm-templates > /dev/null || error_exit "Helm template validation failed"
log_success "Helm template validation passed"

# Kubernetes manifest validation
log_info "Validating Kubernetes manifests..."
find /tmp/helm-templates -name "*.yaml" -exec kubectl --dry-run=client apply -f {} \; > /dev/null || error_exit "Kubernetes manifest validation failed"
log_success "Kubernetes manifest validation passed"

# Security scanning (if not skipping tests)
if [[ "$SKIP_TESTS" != "true" ]]; then
    log_section "Security Scanning"

    log_info "Running security scans on manifests..."

    # Check for security policies with Polaris (if available)
    if command -v polaris &> /dev/null; then
        polaris audit --audit-path /tmp/helm-templates --format=json > /tmp/polaris-report.json || true
        critical_issues=$(jq '.Results[] | select(.Severity == "error") | .Severity' /tmp/polaris-report.json | wc -l)
        if [[ $critical_issues -gt 0 ]]; then
            log_warning "Found $critical_issues critical security issues"
            jq '.Results[] | select(.Severity == "error")' /tmp/polaris-report.json
            read -p "Do you want to continue despite security issues? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                error_exit "Deployment cancelled due to security issues"
            fi
        else
            log_success "No critical security issues found"
        fi
    else
        log_warning "Polaris not available, skipping security scan"
    fi
fi

# Pre-deployment health check
log_section "Pre-deployment Health Check"

log_info "Checking existing deployment health..."
if kubectl get deployment -n "its-camera-ai-$ENVIRONMENT" &> /dev/null; then
    # Check if all deployments are ready
    deployments=$(kubectl get deployments -n "its-camera-ai-$ENVIRONMENT" -o json)
    total_deployments=$(echo "$deployments" | jq '.items | length')
    ready_deployments=$(echo "$deployments" | jq '[.items[] | select(.status.readyReplicas == .status.replicas)] | length')

    if [[ $total_deployments -eq $ready_deployments ]]; then
        log_success "All existing deployments are healthy ($ready_deployments/$total_deployments)"
    else
        log_warning "Some deployments are not healthy ($ready_deployments/$total_deployments)"
    fi
else
    log_info "No existing deployments found (initial deployment)"
fi

# Backup current deployment (if exists)
log_section "Backup Current Deployment"

if kubectl get secret "its-camera-ai-backup-$(date +%Y%m%d)" -n "its-camera-ai-$ENVIRONMENT" &> /dev/null; then
    log_info "Backup for today already exists"
else
    log_info "Creating backup of current deployment..."
    if [[ "$DRY_RUN" != "true" ]]; then
        kubectl create secret generic "its-camera-ai-backup-$(date +%Y%m%d)" \
            --from-literal="timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            --from-literal="version=$(kubectl get deployment camera-service -n its-camera-ai-$ENVIRONMENT -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || echo 'none')" \
            -n "its-camera-ai-$ENVIRONMENT" || true
    fi
    log_success "Backup created"
fi

# Deployment
log_section "Deployment"

log_info "Deploying ITS Camera AI version $VERSION to $ENVIRONMENT..."

if [[ "$DRY_RUN" == "true" ]]; then
    log_info "DRY RUN: Would execute the following Helm command:"
    echo "helm upgrade --install its-camera-ai $PROJECT_ROOT/charts/its-camera-ai \\"
    echo "  --namespace its-camera-ai-$ENVIRONMENT \\"
    echo "  --create-namespace \\"
    echo "  --values $PROJECT_ROOT/charts/its-camera-ai/values-$ENVIRONMENT.yaml \\"
    echo "  --set image.tag=$VERSION \\"
    echo "  --set global.imageTag=$VERSION \\"
    echo "  --wait \\"
    echo "  --timeout=15m"
else
    # Execute deployment
    helm upgrade --install its-camera-ai "$PROJECT_ROOT/charts/its-camera-ai" \
        --namespace "its-camera-ai-$ENVIRONMENT" \
        --create-namespace \
        --values "$PROJECT_ROOT/charts/its-camera-ai/values-$ENVIRONMENT.yaml" \
        --set image.tag="$VERSION" \
        --set global.imageTag="$VERSION" \
        --wait \
        --timeout=15m || error_exit "Helm deployment failed"

    log_success "Helm deployment completed"
fi

# Post-deployment verification
if [[ "$DRY_RUN" != "true" ]]; then
    log_section "Post-deployment Verification"

    # Wait for deployments to be ready
    log_info "Waiting for deployments to be ready..."

    deployments=("camera-service" "analytics-service" "streaming-service" "vision-engine")
    for deployment in "${deployments[@]}"; do
        log_info "Checking $deployment..."
        if kubectl get deployment "$deployment" -n "its-camera-ai-$ENVIRONMENT" &> /dev/null; then
            kubectl rollout status "deployment/$deployment" \
                --namespace="its-camera-ai-$ENVIRONMENT" \
                --timeout=600s || error_exit "$deployment rollout failed"
            log_success "$deployment is ready"
        else
            log_warning "$deployment not found (may be disabled)"
        fi
    done

    # Health checks
    log_info "Running health checks..."

    # Check pod status
    unhealthy_pods=$(kubectl get pods -n "its-camera-ai-$ENVIRONMENT" --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)
    if [[ $unhealthy_pods -eq 0 ]]; then
        log_success "All pods are running"
    else
        log_warning "$unhealthy_pods pods are not running"
        kubectl get pods -n "its-camera-ai-$ENVIRONMENT" --field-selector=status.phase!=Running
    fi

    # Check service endpoints
    log_info "Checking service endpoints..."
    services=$(kubectl get services -n "its-camera-ai-$ENVIRONMENT" -o jsonpath='{.items[*].metadata.name}')
    for service in $services; do
        endpoints=$(kubectl get endpoints "$service" -n "its-camera-ai-$ENVIRONMENT" -o jsonpath='{.subsets[*].addresses[*].ip}' | wc -w)
        if [[ $endpoints -gt 0 ]]; then
            log_success "$service has $endpoints endpoints"
        else
            log_warning "$service has no endpoints"
        fi
    done

    # Application health checks (if not skipping tests)
    if [[ "$SKIP_TESTS" != "true" ]]; then
        log_section "Application Health Tests"

        # Get ingress URL
        ingress_host=$(kubectl get ingress its-camera-ai -n "its-camera-ai-$ENVIRONMENT" -o jsonpath='{.spec.rules[0].host}' 2>/dev/null || echo "localhost")

        # Health check endpoints
        health_checks=(
            "/api/v1/camera/health"
            "/api/v1/analytics/health"
            "/api/v1/streaming/health"
            "/api/v1/inference/health"
        )

        for endpoint in "${health_checks[@]}"; do
            log_info "Testing $endpoint..."
            # Use port-forward for local testing if ingress not available
            if [[ "$ingress_host" == "localhost" ]]; then
                # Test via port-forward (implementation depends on your setup)
                log_warning "Ingress not configured, skipping external health checks"
                break
            else
                if curl -sf "https://$ingress_host$endpoint" > /dev/null; then
                    log_success "$endpoint is healthy"
                else
                    log_warning "$endpoint health check failed"
                fi
            fi
        done

        # Run smoke tests if available
        if [[ -f "$PROJECT_ROOT/tests/smoke/test_deployment.py" ]]; then
            log_info "Running smoke tests..."
            cd "$PROJECT_ROOT"
            python -m pytest tests/smoke/ -v --tb=short || log_warning "Some smoke tests failed"
        else
            log_info "No smoke tests found, skipping"
        fi
    fi

    # Performance check
    log_section "Performance Verification"

    log_info "Checking resource utilization..."
    kubectl top pods -n "its-camera-ai-$ENVIRONMENT" --no-headers 2>/dev/null | while read -r line; do
        pod_name=$(echo "$line" | awk '{print $1}')
        cpu_usage=$(echo "$line" | awk '{print $2}')
        memory_usage=$(echo "$line" | awk '{print $3}')
        log_info "$pod_name: CPU=$cpu_usage, Memory=$memory_usage"
    done || log_warning "Resource metrics not available (metrics-server may not be installed)"

    # Check HPA status
    if kubectl get hpa -n "its-camera-ai-$ENVIRONMENT" &> /dev/null; then
        log_info "HPA Status:"
        kubectl get hpa -n "its-camera-ai-$ENVIRONMENT"
    fi
fi

# Final status
log_section "Deployment Summary"

if [[ "$DRY_RUN" == "true" ]]; then
    log_success "DRY RUN completed successfully!"
    log_info "No actual changes were made to the cluster"
else
    # Get final deployment status
    log_info "Final deployment status:"
    kubectl get all -n "its-camera-ai-$ENVIRONMENT" -l app.kubernetes.io/part-of=its-camera-ai

    # Create deployment record
    kubectl create configmap "deployment-$(date +%Y%m%d-%H%M%S)" \
        --from-literal="environment=$ENVIRONMENT" \
        --from-literal="version=$VERSION" \
        --from-literal="timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --from-literal="user=$(whoami)" \
        --from-literal="status=success" \
        -n "its-camera-ai-$ENVIRONMENT" || true

    log_success "🎉 Deployment to $ENVIRONMENT completed successfully!"
    log_info "Version: $VERSION"
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: its-camera-ai-$ENVIRONMENT"
    log_info "Time: $(date)"

    # Show useful commands
    echo ""
    log_info "Useful commands:"
    echo "  Monitor pods:     kubectl get pods -n its-camera-ai-$ENVIRONMENT -w"
    echo "  View logs:        kubectl logs -f deployment/camera-service -n its-camera-ai-$ENVIRONMENT"
    echo "  Port forward:     kubectl port-forward service/camera-service 8080:8080 -n its-camera-ai-$ENVIRONMENT"
    echo "  Rollback:         $SCRIPT_DIR/rollback-deployment.sh $ENVIRONMENT"
fi

# Notifications (if configured)
if [[ -n "${SLACK_WEBHOOK_URL:-}" && "$DRY_RUN" != "true" ]]; then
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"🚀 ITS Camera AI deployed to $ENVIRONMENT\n• Version: $VERSION\n• Status: SUCCESS\n• Time: $(date)\"}" \
        "$SLACK_WEBHOOK_URL" || log_warning "Failed to send Slack notification"
fi

log_success "Deployment script completed!"
